import asyncio
import base64
import copy
import json
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    cast,
)
from urllib.parse import urljoin

import pytest
from httpx import AsyncByteStream, Response
from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from opentelemetry.trace import Link, SpanKind, TraceState
from opentelemetry.util.types import Attributes, AttributeValue
from respx import MockRouter

from openinference.instrumentation import (
    REDACTED_VALUE,
    Blob,
    TraceConfig,
    suppress_tracing,
)
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.openai._image_utils import serialize_request_input
from openinference.semconv.trace import SpanAttributes

_BASE_URL = "https://api.openai.com/v1/"
_IMAGE_BYTES = b"openinference-image"
_DATA_URI = "data:image/png;base64," + base64.b64encode(_IMAGE_BYTES).decode()
_UPLOADED_URI = "memory://input-image"


class _Uploader:
    def __init__(self, result: Optional[str] = _UPLOADED_URI) -> None:
        self.result = result
        self.blobs: List[Blob] = []

    def upload(self, blob: Blob) -> Optional[str]:
        self.blobs.append(blob)
        return self.result

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        pass


class _RaisingUploader(_Uploader):
    def upload(self, blob: Blob) -> Optional[str]:
        self.blobs.append(blob)
        raise RuntimeError("upload failed")


class _CapturingSampler(Sampler):
    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        self.attributes: Dict[str, AttributeValue] = {}

    def should_sample(
        self,
        parent_context: Optional[Context],
        trace_id: int,
        name: str,
        kind: Optional[SpanKind] = None,
        attributes: Attributes = None,
        links: Optional[Sequence[Link]] = None,
        trace_state: Optional[TraceState] = None,
    ) -> SamplingResult:
        self.attributes = dict(attributes or {})
        return SamplingResult(self.decision)

    def get_description(self) -> str:
        return "capturing sampler"


class _MockStream(AsyncByteStream):
    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks = chunks

    def __iter__(self) -> Iterator[bytes]:
        yield from self._chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def _request(family: str, image_url: str = _DATA_URI) -> Dict[str, Any]:
    if family == "chat":
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url, "detail": "low"},
                        },
                    ],
                }
            ],
        }
    return {
        "model": "gpt-4o-mini",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "describe"},
                    {"type": "input_image", "image_url": image_url, "detail": "low"},
                ],
            }
        ],
    }


def _image_url(request: Mapping[str, Any], family: str) -> str:
    key = "messages" if family == "chat" else "input"
    image = request[key][0]["content"][1]
    if family == "chat":
        return cast(str, image["image_url"]["url"])
    return cast(str, image["image_url"])


@pytest.mark.parametrize("family", ["chat", "responses"])
def test_serialize_request_input_externalizes_oversized_images(family: str) -> None:
    request = _request(family)
    original = copy.deepcopy(request)
    uploader = _Uploader()
    config = TraceConfig(
        blob_uploader=uploader,
        base64_image_max_length=len(_DATA_URI) - 1,
    )

    serialized = serialize_request_input(request, config)

    assert _image_url(json.loads(serialized), family) == _UPLOADED_URI
    assert request == original
    assert uploader.blobs == [
        Blob(
            data=_IMAGE_BYTES,
            mime_type="image/png",
            modality="image",
            attribute_key=SpanAttributes.INPUT_VALUE,
        )
    ]


@pytest.mark.parametrize(
    ("budget", "expected_url", "upload_count"),
    [
        (len(_DATA_URI) - 1, _UPLOADED_URI, 1),
        (len(_DATA_URI), _DATA_URI, 0),
        (len(_DATA_URI) + 1, _DATA_URI, 0),
    ],
)
def test_serialize_request_input_uses_strict_full_string_budget(
    budget: int,
    expected_url: str,
    upload_count: int,
) -> None:
    uploader = _Uploader()
    result = json.loads(
        serialize_request_input(
            _request("chat"),
            TraceConfig(blob_uploader=uploader, base64_image_max_length=budget),
        )
    )

    assert _image_url(result, "chat") == expected_url
    assert len(uploader.blobs) == upload_count


@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.parametrize("image_url", [_DATA_URI, "https://example.com/image.png"])
def test_serialize_request_input_hides_images_without_upload(
    family: str,
    image_url: str,
) -> None:
    uploader = _RaisingUploader()
    serialized = serialize_request_input(
        _request(family, image_url),
        TraceConfig(
            blob_uploader=uploader,
            hide_input_images=True,
            base64_image_max_length=0,
        ),
    )

    assert _image_url(json.loads(serialized), family) == REDACTED_VALUE
    assert uploader.blobs == []


@pytest.mark.parametrize(
    "uploader",
    [None, _Uploader(None), _RaisingUploader(), _Uploader("relative/image.png")],
    ids=["missing", "rejected", "exception", "invalid-uri"],
)
def test_serialize_request_input_redacts_failed_upload(uploader: Optional[_Uploader]) -> None:
    config = TraceConfig(blob_uploader=uploader, base64_image_max_length=0)
    request = _request("responses")
    request["input"][0]["content"].append(
        {
            "type": "input_image",
            "image_url": "https://example.com/remote.png",
        }
    )
    result = json.loads(serialize_request_input(request, config))
    assert _image_url(result, "responses") == REDACTED_VALUE
    assert result["input"][0]["content"][2]["image_url"] == "https://example.com/remote.png"


def test_serialize_request_input_redacts_malformed_base64_leaf() -> None:
    request = _request("chat", "data:image/png;base64,%%%")
    result = json.loads(
        serialize_request_input(
            request,
            TraceConfig(blob_uploader=_Uploader(), base64_image_max_length=0),
        )
    )
    assert _image_url(result, "chat") == REDACTED_VALUE


def test_serialize_request_input_hides_inputs_before_copying() -> None:
    class _Uncopyable:
        def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
            raise AssertionError("hidden input was copied")

    assert (
        serialize_request_input(
            {"messages": _Uncopyable()},
            TraceConfig(hide_inputs=True),
        )
        == REDACTED_VALUE
    )


@pytest.fixture
def custom_instrumentation(
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[Callable[[TraceConfig, TracerProvider], None]]:
    OpenAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()

    def instrument(config: TraceConfig, provider: TracerProvider) -> None:
        OpenAIInstrumentor().instrument(config=config, tracer_provider=provider)

    yield instrument
    OpenAIInstrumentor().uninstrument()


def _provider(
    exporter: InMemorySpanExporter,
    sampler: Optional[Sampler] = None,
) -> TracerProvider:
    provider = TracerProvider(sampler=sampler)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def _mock_response(respx_mock: MockRouter, family: str) -> None:
    if family == "chat":
        respx_mock.post(urljoin(_BASE_URL, "chat/completions")).mock(
            return_value=Response(
                200,
                json={
                    "id": "chat-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "model": "gpt-4o-mini",
                },
            )
        )
        return
    respx_mock.post(urljoin(_BASE_URL, "responses")).mock(
        return_value=Response(
            200,
            json={
                "id": "response-1",
                "object": "response",
                "status": "completed",
                "output": [],
                "model": "gpt-4o-mini",
            },
        )
    )


def _openinference_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = [
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope
        and span.instrumentation_scope.name == "openinference.instrumentation.openai"
    ]
    assert len(spans) == 1
    return spans[0]


@pytest.mark.parametrize("family", ["chat", "responses"])
@pytest.mark.parametrize("is_async", [False, True])
def test_openai_call_externalizes_input_value_image(
    family: str,
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    custom_instrumentation: Callable[[TraceConfig, TracerProvider], None],
) -> None:
    import openai

    _mock_response(respx_mock, family)
    uploader = _Uploader()
    custom_instrumentation(
        TraceConfig(blob_uploader=uploader, base64_image_max_length=0),
        _provider(in_memory_span_exporter),
    )
    request = _request(family)
    original = copy.deepcopy(request)

    async def call_async() -> Any:
        client = openai.AsyncOpenAI(api_key="sk-", base_url=_BASE_URL)
        if family == "chat":
            return await client.chat.completions.create(**request)
        return await client.responses.create(**request)

    if is_async:
        response = asyncio.run(call_async())
    else:
        client = openai.OpenAI(api_key="sk-", base_url=_BASE_URL)
        response = (
            client.chat.completions.create(**request)
            if family == "chat"
            else client.responses.create(**request)
        )

    assert response.model == "gpt-4o-mini"
    assert request == original
    span = _openinference_span(in_memory_span_exporter)
    assert span.attributes is not None
    input_value = span.attributes[SpanAttributes.INPUT_VALUE]
    assert isinstance(input_value, str)
    assert _image_url(json.loads(input_value), family) == _UPLOADED_URI
    assert len(_input_blobs(uploader)) == 1


def test_openai_sampler_sees_redacted_input_value(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    custom_instrumentation: Callable[[TraceConfig, TracerProvider], None],
) -> None:
    import openai

    _mock_response(respx_mock, "chat")
    sampler = _CapturingSampler(Decision.RECORD_AND_SAMPLE)
    custom_instrumentation(TraceConfig(), _provider(in_memory_span_exporter, sampler))

    response = openai.OpenAI(api_key="sk-", base_url=_BASE_URL).chat.completions.create(
        **_request("chat")
    )

    assert response.model == "gpt-4o-mini"
    assert sampler.attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE


@pytest.mark.parametrize("suppressed", [False, True], ids=["sampled-out", "suppressed"])
def test_openai_skips_upload_when_span_will_not_record(
    suppressed: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    custom_instrumentation: Callable[[TraceConfig, TracerProvider], None],
) -> None:
    import openai

    _mock_response(respx_mock, "chat")
    uploader = _Uploader()
    sampler = _CapturingSampler(Decision.RECORD_AND_SAMPLE if suppressed else Decision.DROP)
    custom_instrumentation(
        TraceConfig(blob_uploader=uploader, base64_image_max_length=0),
        _provider(in_memory_span_exporter, sampler),
    )
    client = openai.OpenAI(api_key="sk-", base_url=_BASE_URL)

    if suppressed:
        with suppress_tracing():
            response = client.chat.completions.create(**_request("chat"))
    else:
        response = client.chat.completions.create(**_request("chat"))

    assert response.model == "gpt-4o-mini"
    assert uploader.blobs == []
    assert _openinference_spans(in_memory_span_exporter) == []


def test_input_serialization_failure_does_not_affect_openai_call(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    custom_instrumentation: Callable[[TraceConfig, TracerProvider], None],
) -> None:
    import openai

    from openinference.instrumentation.openai import _request as request_module

    _mock_response(respx_mock, "chat")
    custom_instrumentation(TraceConfig(), _provider(in_memory_span_exporter))

    def fail_serialization(request_parameters: Mapping[str, Any], config: TraceConfig) -> str:
        raise ValueError("cannot serialize")

    monkeypatch.setattr(request_module, "serialize_request_input", fail_serialization)
    response = openai.OpenAI(api_key="sk-", base_url=_BASE_URL).chat.completions.create(
        **_request("chat")
    )

    assert response.model == "gpt-4o-mini"
    span = _openinference_span(in_memory_span_exporter)
    assert span.attributes is not None
    assert span.attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE


@pytest.mark.parametrize("is_async", [False, True])
def test_streaming_input_is_externalized_once_before_iteration(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    custom_instrumentation: Callable[[TraceConfig, TracerProvider], None],
) -> None:
    import openai

    chunks = [
        b'data: {"id":"chat-1","object":"chat.completion.chunk","created":0,'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant",'
        b'"content":"ok"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chat-1","object":"chat.completion.chunk","created":0,'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},'
        b'"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    respx_mock.post(urljoin(_BASE_URL, "chat/completions")).mock(
        return_value=Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_MockStream(chunks),
        )
    )
    uploader = _Uploader()
    custom_instrumentation(
        TraceConfig(blob_uploader=uploader, base64_image_max_length=0),
        _provider(in_memory_span_exporter),
    )
    request = {**_request("chat"), "stream": True}

    async def call_async() -> None:
        stream = await openai.AsyncOpenAI(
            api_key="sk-", base_url=_BASE_URL
        ).chat.completions.create(**request)
        assert len(_input_blobs(uploader)) == 1
        async for _ in stream:
            pass

    if is_async:
        asyncio.run(call_async())
    else:
        stream = openai.OpenAI(api_key="sk-", base_url=_BASE_URL).chat.completions.create(**request)
        assert len(_input_blobs(uploader)) == 1
        for _ in stream:
            pass

    assert len(_input_blobs(uploader)) == 1
    span = _openinference_span(in_memory_span_exporter)
    assert span.attributes is not None
    input_value = span.attributes[SpanAttributes.INPUT_VALUE]
    assert isinstance(input_value, str)
    assert _image_url(json.loads(input_value), "chat") == _UPLOADED_URI


def _input_blobs(uploader: _Uploader) -> List[Blob]:
    return [blob for blob in uploader.blobs if blob.attribute_key == SpanAttributes.INPUT_VALUE]


def _openinference_spans(exporter: InMemorySpanExporter) -> List[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope
        and span.instrumentation_scope.name == "openinference.instrumentation.openai"
    ]
