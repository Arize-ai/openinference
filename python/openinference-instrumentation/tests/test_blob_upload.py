import base64
from typing import Any, Dict, List, Optional

import pytest

from openinference.instrumentation import (
    Blob,
    BlobUploader,
    TraceConfig,
    _blob_upload,
    decode_base64_data_uri_to_blob,
    load_blob_uploader,
    parse_base64_data_uri,
)
from openinference.instrumentation.config import (
    OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH,
    OPENINFERENCE_BLOB_UPLOADER,
    REDACTED_VALUE,
)
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)

PNG_BYTES = b"\x89PNG\r\n" + bytes(range(256)) * 40  # ~10KB of fake image data

PNG_DATA_URI = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()

INPUT_IMAGE_URL_KEY = (
    f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.1."
    f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
)
OUTPUT_IMAGE_URL_KEY = (
    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.0."
    f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
)


@pytest.fixture(autouse=True)
def _clear_uploader_cache() -> None:
    _blob_upload._UPLOADER_CACHE.clear()


class InMemoryUploader:
    """A minimal BlobUploader implementation for tests — OpenInference ships
    no implementation of its own."""

    def __init__(self) -> None:
        self.store: Dict[str, bytes] = {}
        self.blobs: List[Blob] = []
        self.accepting = True

    def upload(self, blob: Blob) -> Optional[str]:
        if not self.accepting:
            return None
        uri = f"memory://{blob.content_sha256}"
        self.store[uri] = blob.data
        self.blobs.append(blob)
        return uri

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        self.accepting = False


def test_blob_derives_modality_and_digest() -> None:
    image_blob = Blob(data=PNG_BYTES, mime_type="image/png")
    assert image_blob.modality == "image"
    assert len(image_blob.content_sha256) == 64
    # The Blob contract is media-agnostic even though only images offload today.
    assert Blob(data=b"RIFF", mime_type="audio/wav").modality == "audio"
    assert Blob(data=b"%PDF-1.4", mime_type="application/pdf").modality == "document"
    assert Blob(data=b"", mime_type="video/mp4").modality == "video"


def test_in_memory_uploader_satisfies_protocol() -> None:
    assert isinstance(InMemoryUploader(), BlobUploader)


@pytest.mark.parametrize("key", [INPUT_IMAGE_URL_KEY, OUTPUT_IMAGE_URL_KEY])
def test_mask_uploads_oversized_image(key: str) -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    masked = config.mask(key, PNG_DATA_URI)
    assert isinstance(masked, str) and masked.startswith("memory://")
    assert uploader.store[masked] == PNG_BYTES


def test_mask_passes_blob_context_to_uploader() -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI)
    assert len(uploader.blobs) == 1
    assert uploader.blobs[0].mime_type == "image/png"
    assert uploader.blobs[0].modality == "image"
    assert uploader.blobs[0].attribute_key == INPUT_IMAGE_URL_KEY


def test_mask_redacts_oversized_image_without_uploader() -> None:
    config = TraceConfig(base64_image_max_length=100)
    assert config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI) == REDACTED_VALUE


def test_mask_redacts_when_uploader_rejects() -> None:
    uploader = InMemoryUploader()
    uploader.shutdown()  # upload() now returns None
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    assert config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI) == REDACTED_VALUE


def test_mask_keeps_small_image_inline() -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(
        blob_uploader=uploader,
        base64_image_max_length=len(PNG_DATA_URI) + 1,
    )
    assert config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI) == PNG_DATA_URI
    assert not uploader.store


def test_mask_keeps_external_image_url() -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=10)
    url = "https://example.com/image.png"
    assert config.mask(INPUT_IMAGE_URL_KEY, url) == url
    assert not uploader.store


def test_hide_takes_precedence_over_upload() -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(
        blob_uploader=uploader,
        hide_input_images=True,
        base64_image_max_length=100,
    )
    assert config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI) is None
    # Hidden content must never reach storage.
    assert not uploader.store


def test_non_recording_span_skips_upload() -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    from openinference.instrumentation import OITracer

    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    tracer = OITracer(
        TracerProvider(sampler=ALWAYS_OFF).get_tracer(__name__),
        config=config,
    )
    span = tracer.start_span("llm")
    span.set_attribute(INPUT_IMAGE_URL_KEY, PNG_DATA_URI)
    span.end()
    # No blob was uploaded for the sampled-out span.
    assert not uploader.store


class _FakeEntryPoint:
    def __init__(self, name: str, target: Any) -> None:
        self.name = name
        self._target = target

    def load(self) -> Any:
        return self._target


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch, entry_points_list: List[_FakeEntryPoint]
) -> None:
    def fake_entry_points(*, group: str) -> List[_FakeEntryPoint]:
        assert group == _blob_upload.BLOB_UPLOADER_ENTRY_POINT_GROUP
        return entry_points_list

    monkeypatch.setattr(_blob_upload, "entry_points", fake_entry_points)


def test_load_blob_uploader_from_instance_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploader = InMemoryUploader()
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("mem", uploader)])
    assert load_blob_uploader("mem") is uploader


def test_load_blob_uploader_instantiates_class_entry_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("mem", InMemoryUploader)])
    loaded = load_blob_uploader("mem")
    assert isinstance(loaded, InMemoryUploader)


def test_load_blob_uploader_unknown_name_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("mem", InMemoryUploader)])
    assert load_blob_uploader("nope") is None


def test_load_blob_uploader_rejects_non_uploader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("bad", object())])
    assert load_blob_uploader("bad") is None


def test_blob_uploader_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("mem", InMemoryUploader)])
    monkeypatch.setenv(OPENINFERENCE_BLOB_UPLOADER, "mem")
    monkeypatch.setenv(OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH, "100")
    config = TraceConfig()
    assert isinstance(config.blob_uploader, InMemoryUploader)
    masked = config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI)
    assert isinstance(masked, str) and masked.startswith("memory://")


def test_blob_uploader_env_var_unset_leaves_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPENINFERENCE_BLOB_UPLOADER, raising=False)
    config = TraceConfig()
    assert config.blob_uploader is None


def test_parse_and_decode_base64_data_uri() -> None:
    parsed = parse_base64_data_uri(PNG_DATA_URI)
    assert parsed is not None
    mime_type, payload = parsed
    assert mime_type == "image/png"
    assert base64.b64decode(payload) == PNG_BYTES
    assert parse_base64_data_uri("https://example.com/a.png") is None
    blob = decode_base64_data_uri_to_blob(PNG_DATA_URI, attribute_key="k")
    assert blob is not None
    assert blob.data == PNG_BYTES
    assert blob.mime_type == "image/png"
    assert blob.attribute_key == "k"
    assert decode_base64_data_uri_to_blob("data:image/png;base64,%%%") is None


@pytest.mark.parametrize(
    "uri,valid",
    [
        ("https://bucket.example.com/a.png", True),
        ("gs://bucket/prefix/a.png", True),
        ("s3://bucket/a.png", True),
        ("file:///tmp/a.png", True),
        ("memory://abc123", True),
        ("internal_docs/blob_store/a.png", False),  # bare file path — no scheme
        ("/tmp/a.png", False),
        ("", False),
        ("https://example.com/a b.png", False),  # whitespace
    ],
)
def test_is_valid_reference_uri(uri: str, valid: bool) -> None:
    from openinference.instrumentation._blob_upload import is_valid_reference_uri

    assert is_valid_reference_uri(uri) is valid


def test_mask_redacts_when_uploader_returns_invalid_uri() -> None:
    class PathReturningUploader(InMemoryUploader):
        def upload(self, blob: Blob) -> Optional[str]:
            return "blob_store/not-a-uri.png"  # bare relative path

    config = TraceConfig(blob_uploader=PathReturningUploader(), base64_image_max_length=100)
    assert config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI) == REDACTED_VALUE


def test_mask_externalizes_callable_value() -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    masked = config.mask(INPUT_IMAGE_URL_KEY, lambda: PNG_DATA_URI)
    assert isinstance(masked, str) and masked.startswith("memory://")
    assert uploader.store[masked] == PNG_BYTES


def test_mask_redacts_callable_value_without_uploader() -> None:
    config = TraceConfig(base64_image_max_length=100)
    assert config.mask(INPUT_IMAGE_URL_KEY, lambda: PNG_DATA_URI) == REDACTED_VALUE


def test_hidden_callable_value_is_never_evaluated() -> None:
    def boom() -> str:
        raise AssertionError("hidden lazy value must not be evaluated")

    config = TraceConfig(hide_input_images=True)
    assert config.mask(INPUT_IMAGE_URL_KEY, boom) is None


def test_mask_externalize_false_redacts_without_upload() -> None:
    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    assert config.mask(INPUT_IMAGE_URL_KEY, PNG_DATA_URI, externalize=False) == REDACTED_VALUE
    assert not uploader.store


@pytest.mark.parametrize(
    "bad_uploader",
    ["mem", InMemoryUploader, object()],
    ids=["string", "class", "wrong-object"],
)
def test_trace_config_rejects_invalid_blob_uploader(bad_uploader: object) -> None:
    with pytest.raises(TypeError, match="blob_uploader"):
        TraceConfig(blob_uploader=bad_uploader)  # type: ignore[arg-type]


def test_load_blob_uploader_is_memoized(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("mem", InMemoryUploader)])
    first = load_blob_uploader("mem")
    second = load_blob_uploader("mem")
    assert first is not None and first is second


def test_env_var_configs_share_one_uploader(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_entry_points(monkeypatch, [_FakeEntryPoint("mem", InMemoryUploader)])
    monkeypatch.setenv(OPENINFERENCE_BLOB_UPLOADER, "mem")
    config_a = TraceConfig()
    config_b = TraceConfig()
    assert config_a.blob_uploader is not None
    assert config_a.blob_uploader is config_b.blob_uploader


def _make_tracer_provider() -> "tuple[object, object]":
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_suppressed_span_skips_upload() -> None:
    from openinference.instrumentation import OITracer, suppress_tracing

    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    provider, _ = _make_tracer_provider()
    tracer = OITracer(provider.get_tracer(__name__), config=config)  # type: ignore[attr-defined]
    with suppress_tracing():
        span = tracer.start_span("llm", attributes={INPUT_IMAGE_URL_KEY: PNG_DATA_URI})
        span.end()
    assert not uploader.store


def test_sampled_out_start_attributes_skip_upload() -> None:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

    from openinference.instrumentation import OITracer

    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    tracer = OITracer(TracerProvider(sampler=ALWAYS_OFF).get_tracer(__name__), config=config)
    span = tracer.start_span("llm", attributes={INPUT_IMAGE_URL_KEY: PNG_DATA_URI})
    span.end()
    assert not uploader.store


def test_start_span_attributes_upload_exactly_once() -> None:
    from openinference.instrumentation import OITracer

    uploader = InMemoryUploader()
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=100)
    provider, exporter = _make_tracer_provider()
    tracer = OITracer(provider.get_tracer(__name__), config=config)  # type: ignore[attr-defined]
    span = tracer.start_span("llm", attributes={INPUT_IMAGE_URL_KEY: PNG_DATA_URI})
    span.end()
    assert len(uploader.blobs) == 1  # not once for sampling and once for the span
    (finished,) = exporter.get_finished_spans()  # type: ignore[attr-defined]
    attribute = (finished.attributes or {})[INPUT_IMAGE_URL_KEY]
    assert isinstance(attribute, str) and attribute.startswith("memory://")


def test_ended_span_attribute_write_still_warns(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    from openinference.instrumentation import OITracer

    provider, _ = _make_tracer_provider()
    tracer = OITracer(provider.get_tracer(__name__), config=TraceConfig())  # type: ignore[attr-defined]
    span = tracer.start_span("llm")
    span.end()
    with caplog.at_level(logging.WARNING, logger="opentelemetry.sdk.trace"):
        span.set_attribute("llm.token_count.total", 7)
    assert any("ended span" in record.message for record in caplog.records)
