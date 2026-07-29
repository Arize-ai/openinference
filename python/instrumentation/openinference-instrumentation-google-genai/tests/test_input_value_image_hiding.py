import json
from typing import Any, cast

import pytest
from google.genai import types
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import (
    REDACTED_VALUE,
    OITracer,
    TraceConfig,
)
from openinference.instrumentation import (
    get_input_attributes as get_oi_input_attributes,
)
from openinference.instrumentation.google_genai._context import (
    CapturedRequestScope,
    _CapturedRequestWrapper,
    get_input_attributes,
)
from openinference.instrumentation.google_genai._image_utils import (
    redact_images_from_request_parameters,
)
from openinference.instrumentation.google_genai._wrappers import (
    _SyncCreateCachesWrapper,
    _SyncCreateInteractionWrapper,
    _SyncGenerateContent,
)
from openinference.semconv.trace import SpanAttributes


def _get_input_value(
    request: dict[str, Any],
    config: TraceConfig,
) -> dict[str, Any]:
    def wrapped(*args: Any, **kwargs: Any) -> None:
        return None

    with CapturedRequestScope():
        _CapturedRequestWrapper()(
            wrapped,
            None,
            ("POST", "/models/gemini:generateContent", request),
            {},
        )
        attributes = dict(get_input_attributes(config))

    input_value = cast(str, attributes[SpanAttributes.INPUT_VALUE])
    return cast(dict[str, Any], json.loads(input_value))


# GenerateContent request capture
def test_generate_content_input_value_hides_inline_and_file_images() -> None:
    request: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {"text": "Describe the images."},
                    {
                        "inlineData": {
                            "data": "aW1hZ2U=",
                            "mimeType": "image/png",
                        }
                    },
                    {
                        "fileData": {
                            "fileUri": "https://example.com/image.jpg",
                            "mimeType": "image/jpeg",
                        }
                    },
                    {
                        "inlineData": {
                            "data": "YXVkaW8=",
                            "mimeType": "audio/wav",
                        }
                    },
                ]
            }
        ]
    }

    input_value = _get_input_value(
        request,
        TraceConfig(hide_input_images=True),
    )
    parts = input_value["contents"][0]["parts"]

    assert parts[0]["text"] == "Describe the images."
    assert parts[1]["inline_data"]["data"] == REDACTED_VALUE
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert parts[2]["file_data"]["file_uri"] == REDACTED_VALUE
    assert parts[2]["file_data"]["mime_type"] == "image/jpeg"
    assert parts[3]["inline_data"]["data"] == "YXVkaW8="
    assert request["contents"][0]["parts"][1]["inlineData"]["data"] == "aW1hZ2U="
    assert (
        request["contents"][0]["parts"][2]["fileData"]["fileUri"] == "https://example.com/image.jpg"
    )


@pytest.mark.parametrize(
    "maximum_length, expected_data",
    [
        pytest.param(
            0,
            REDACTED_VALUE,
            id="zero-limit",
        ),
        pytest.param(
            len("data:image/png;base64,") + len("aW1hZ2U=") - 1,
            REDACTED_VALUE,
            id="over-limit",
        ),
        pytest.param(
            len("data:image/png;base64,") + len("aW1hZ2U="),
            "aW1hZ2U=",
            id="at-limit",
        ),
    ],
)
def test_generate_content_input_value_respects_base64_image_max_length(
    maximum_length: int,
    expected_data: str,
) -> None:
    request: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "data": "aW1hZ2U=",
                            "mimeType": "image/png",
                        }
                    }
                ]
            }
        ]
    }

    input_value = _get_input_value(
        request,
        TraceConfig(
            hide_input_images=False,
            base64_image_max_length=maximum_length,
        ),
    )

    assert input_value["contents"][0]["parts"][0]["inline_data"]["data"] == expected_data


def test_generate_content_input_value_preserves_images_when_unmasked() -> None:
    request: dict[str, Any] = {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "data": "aW1hZ2U=",
                            "mimeType": "image/png",
                        }
                    },
                    {
                        "fileData": {
                            "fileUri": "https://example.com/image.jpg",
                            "mimeType": "image/jpeg",
                        }
                    },
                ]
            }
        ]
    }

    input_value = _get_input_value(
        request,
        TraceConfig(
            hide_input_images=False,
            base64_image_max_length=1_000,
        ),
    )
    parts = input_value["contents"][0]["parts"]

    assert parts[0]["inline_data"]["data"] == "aW1hZ2U="
    assert parts[1]["file_data"]["file_uri"] == "https://example.com/image.jpg"


# GenerateContent finished-span wiring
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(TraceConfig(hide_input_images=True), id="hide-input-images"),
        pytest.param(
            TraceConfig(hide_input_images=False, base64_image_max_length=1),
            id="base64-image-max-length",
        ),
    ],
)
def test_generate_content_writes_redacted_input_value_to_finished_span(
    config: TraceConfig,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    request = {
        "contents": [
            {
                "parts": [
                    {
                        "inlineData": {
                            "data": "aW1hZ2U=",
                            "mimeType": "image/png",
                        }
                    }
                ]
            }
        ]
    }

    def api_request(*args: Any, **kwargs: Any) -> None:
        return None

    def generate_content(*, model: str, contents: Any) -> dict[str, str]:
        _CapturedRequestWrapper()(
            api_request,
            None,
            ("POST", "/models/gemini:generateContent", request),
            {},
        )
        return {"text": "response"}

    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
    _SyncGenerateContent(tracer=tracer)(
        generate_content,
        None,
        (),
        {"model": "gemini-2.0-flash", "contents": "describe this"},
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    input_value = cast(str, attributes[SpanAttributes.INPUT_VALUE])
    input_payload = cast(dict[str, Any], json.loads(input_value))
    assert input_payload["contents"][0]["parts"][0]["inline_data"]["data"] == REDACTED_VALUE


# Interactions finished-span wiring
@pytest.mark.parametrize(
    "config",
    [
        pytest.param(TraceConfig(hide_input_images=True), id="hide-input-images"),
        pytest.param(
            TraceConfig(hide_input_images=False, base64_image_max_length=1),
            id="base64-image-max-length",
        ),
    ],
)
def test_create_interaction_writes_redacted_input_value_to_finished_span(
    config: TraceConfig,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    interaction_input = [
        {"type": "text", "text": "Describe the image."},
        {
            "type": "image",
            "data": "aW1hZ2U=",
            "mime_type": "image/png",
        },
    ]

    def create_interaction(*, model: str, input: Any) -> None:
        return None

    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
    _SyncCreateInteractionWrapper(tracer=tracer)(
        create_interaction,
        None,
        (),
        {
            "model": "gemini-2.5-flash",
            "input": interaction_input,
        },
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    input_value = cast(str, attributes[SpanAttributes.INPUT_VALUE])
    input_payload = cast(list[dict[str, Any]], json.loads(input_value))
    assert input_payload[0]["text"] == "Describe the image."
    assert input_payload[1]["data"] == REDACTED_VALUE


def test_create_interaction_hides_uri_and_type_discriminated_image(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    interaction_input = [
        {
            "type": "image",
            "uri": "https://ex.com/a.png",
            "mime_type": "image/png",
        },
        {
            "type": "image",
            "data": "a@WhZ2UU==",
        },
    ]

    def create_interaction(*, model: str, input: Any) -> None:
        return None

    tracer = OITracer(
        tracer_provider.get_tracer(__name__),
        config=TraceConfig(hide_input_images=True),
    )
    _SyncCreateInteractionWrapper(tracer=tracer)(
        create_interaction,
        None,
        (),
        {
            "model": "gemini-2.5-flash",
            "input": interaction_input,
        },
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    input_value = cast(str, attributes[SpanAttributes.INPUT_VALUE])
    input_payload = cast(list[dict[str, Any]], json.loads(input_value))

    assert input_payload[0]["uri"] == REDACTED_VALUE
    assert input_payload[1]["data"] == REDACTED_VALUE


def test_create_interaction_limits_mime_less_data_and_base64_uri(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    interaction_input = [
        {
            "type": "image",
            "data": "a@WhZ2UU==",
        },
        {
            "type": "image",
            "uri": "data:image/png;base64,aW1hZ2U=",
            "mime_type": "image/png",
        },
    ]

    def create_interaction(*, model: str, input: Any) -> None:
        return None

    tracer = OITracer(
        tracer_provider.get_tracer(__name__),
        config=TraceConfig(
            hide_input_images=False,
            base64_image_max_length=1,
        ),
    )
    _SyncCreateInteractionWrapper(tracer=tracer)(
        create_interaction,
        None,
        (),
        {
            "model": "gemini-2.5-flash",
            "input": interaction_input,
        },
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    input_value = cast(str, attributes[SpanAttributes.INPUT_VALUE])
    input_payload = cast(list[dict[str, Any]], json.loads(input_value))

    assert input_payload[0]["data"] == REDACTED_VALUE
    assert input_payload[1]["uri"] == REDACTED_VALUE


def test_hides_tuple_interaction_input() -> None:
    image = {
        "type": "image",
        "data": "c2Vuc2l0aXZlLWltYWdl",
        "mime_type": "image/png",
    }
    request = {"model": "gemini-2.5-flash", "input": (image,)}

    sanitized = redact_images_from_request_parameters(
        request,
        hide_input_images=True,
        base64_image_max_length=32_000,
    )
    assert sanitized["input"][0]["data"] == REDACTED_VALUE


def test_applies_image_length_limit_when_mime_type_is_omitted() -> None:
    image = {"type": "image", "data": "A" * 100}
    request = {"model": "gemini-2.5-flash", "input": [image]}

    sanitized = redact_images_from_request_parameters(
        request,
        hide_input_images=False,
        base64_image_max_length=1,
    )
    assert sanitized["input"][0]["data"] == REDACTED_VALUE


# Cache Pydantic input
@pytest.mark.parametrize(
    "config, expected_data",
    [
        pytest.param(
            TraceConfig(hide_input_images=True),
            REDACTED_VALUE,
            id="hide-input-images",
        ),
        pytest.param(
            TraceConfig(hide_input_images=False, base64_image_max_length=1),
            REDACTED_VALUE,
            id="base64-image-max-length",
        ),
        pytest.param(
            TraceConfig(hide_input_images=False, base64_image_max_length=1_000),
            "aW1hZ2U=",
            id="preserve-under-limit",
        ),
    ],
)
def test_create_cache_handles_pydantic_image_input(
    config: TraceConfig,
    expected_data: str,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    cache_config = types.CreateCachedContentConfig(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=b"image", mime_type="image/png")],
            )
        ]
    )

    def create_cache(
        *,
        model: str,
        config: types.CreateCachedContentConfig,
    ) -> types.CachedContent:
        return types.CachedContent()

    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
    _SyncCreateCachesWrapper(tracer=tracer)(
        create_cache,
        None,
        (),
        {
            "model": "gemini-2.5-flash",
            "config": cache_config,
        },
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    input_value = cast(str, attributes[SpanAttributes.INPUT_VALUE])
    input_payload = cast(dict[str, Any], json.loads(input_value))
    image_data = input_payload["config"]["contents"][0]["parts"][0]["inline_data"]["data"]
    assert image_data == expected_data


def test_create_cache_hides_file_image_and_preserves_inline_audio(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    cache_config = types.CreateCachedContentConfig(
        contents=[
            types.Content(
                role="user",
                parts=[
                    # File-backed images have no base64 payload, so hiding input
                    # images must redact the URI itself.
                    types.Part(
                        file_data=types.FileData(
                            file_uri="gs://example-bucket/image.png",
                            mime_type="image/png",
                        )
                    ),
                    # Audio uses the same Blob/data representation as an inline
                    # image, but image-specific hiding and size limits must not
                    # redact it.
                    types.Part.from_bytes(data=b"\xfb\xff", mime_type="audio/wav"),
                ],
            )
        ]
    )

    def create_cache(
        *,
        model: str,
        config: types.CreateCachedContentConfig,
    ) -> types.CachedContent:
        return types.CachedContent()

    tracer = OITracer(
        tracer_provider.get_tracer(__name__),
        config=TraceConfig(
            hide_input_images=True,
            base64_image_max_length=1,
        ),
    )
    _SyncCreateCachesWrapper(tracer=tracer)(
        create_cache,
        None,
        (),
        {
            "model": "gemini-2.5-flash",
            "config": cache_config,
        },
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    input_value = cast(str, attributes[SpanAttributes.INPUT_VALUE])
    input_payload = cast(dict[str, Any], json.loads(input_value))
    parts = input_payload["config"]["contents"][0]["parts"]

    assert parts[0]["file_data"]["file_uri"] == REDACTED_VALUE
    assert parts[0]["file_data"]["mime_type"] == "image/png"
    assert parts[1]["inline_data"]["data"] == "-_8="
    assert parts[1]["inline_data"]["mime_type"] == "audio/wav"


# Serialization-preservation regressions
def test_unchanged_cache_input_preserves_pydantic_serialization() -> None:
    cache_config = types.CreateCachedContentConfig(
        display_name="multimodal cache",
        ttl="300s",
        system_instruction="Describe the supplied context.",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(text="An image below the configured size limit."),
                    types.Part.from_bytes(data=b"image", mime_type="image/png"),
                    types.Part.from_bytes(data=b"\xfb\xff", mime_type="audio/wav"),
                ],
            )
        ],
    )
    request_parameters = {
        "model": "gemini-2.5-flash",
        "config": cache_config,
    }

    sanitized_parameters = redact_images_from_request_parameters(
        request_parameters,
        hide_input_images=False,
        base64_image_max_length=1_000,
    )

    # No image was redacted, so retain the original Pydantic object and its
    # JSON-mode serializers instead of tracing a model_dump(mode="python") tree.
    assert sanitized_parameters["config"] is cache_config
    assert get_oi_input_attributes(sanitized_parameters) == get_oi_input_attributes(
        request_parameters
    )
