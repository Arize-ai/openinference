import base64
import json
from importlib import import_module
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, cast
from urllib.parse import urljoin

import pytest
from httpx import Response
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
from respx import MockRouter

from openinference.instrumentation import REDACTED_VALUE, TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.instrumentation.openai._attributes._responses_api import _ResponsesApiAttributes
from openinference.instrumentation.openai._image_utils import get_attributes_from_image_files
from openinference.instrumentation.openai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.semconv.trace import (
    ImageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_OPENAI_BASE_URL = "https://api.openai.com/v1/"
_OPENINFERENCE_SCOPE = "openinference.instrumentation.openai"

# Leading bytes matter: the media type is sniffed from them when the caller
# doesn't say what the file is.
_PNG_BYTES = b"\x89PNG\r\n\x1a\nimage"
_JPEG_BYTES = b"\xff\xd8\xffimage"
_WEBP_BYTES = b"RIFF\x00\x00\x00\x00WEBPimage"


def _client(is_async: bool) -> Any:
    openai = import_module("openai")
    if is_async:
        return openai.AsyncOpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL)
    return openai.OpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL)


def _data_url(media_type: str, data: bytes) -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def _input_image(index: int = 0) -> str:
    return f"{SpanAttributes.INPUT_IMAGES}.{index}.{ImageAttributes.IMAGE_URL}"


def _output_image(index: int = 0) -> str:
    return f"{SpanAttributes.OUTPUT_IMAGES}.{index}.{ImageAttributes.IMAGE_URL}"


def _openinference_span(
    exporter: InMemorySpanExporter, name: str = "ImagesResponse"
) -> ReadableSpan:
    spans = tuple(
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope is not None
        and span.instrumentation_scope.name == _OPENINFERENCE_SCOPE
    )
    assert len(spans) == 1
    span = spans[0]
    assert span.name == name
    return span


def _image_span_attributes(
    exporter: InMemorySpanExporter,
    expected_input: Mapping[str, Any],
) -> dict[str, AttributeValue]:
    """Pop the attributes every Images API span carries and return the rest."""
    span = _openinference_span(exporter)
    assert span.status.is_ok
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "openai"
    assert attributes.pop(SpanAttributes.LLM_SYSTEM) == "openai"
    assert json.loads(cast(str, attributes.pop(SpanAttributes.INPUT_VALUE))) == expected_input
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value
    assert (
        json.loads(cast(str, attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS)))
        == expected_input
    )
    assert isinstance(attributes.pop(SpanAttributes.OUTPUT_VALUE), str)
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value
    return attributes


@pytest.fixture
def instrument_with_config(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[Callable[[TraceConfig], None]]:
    OpenAIInstrumentor().uninstrument()

    def _instrument(config: TraceConfig) -> None:
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)

    yield _instrument

    OpenAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("response_image", "response_format", "request_format", "expected_url"),
    [
        pytest.param(
            {"url": "https://example.com/generated.png"},
            None,
            None,
            "https://example.com/generated.png",
            id="hosted-url",
        ),
        pytest.param(
            {"b64_json": "aW1hZ2U="},
            "webp",
            None,
            "data:image/webp;base64,aW1hZ2U=",
            id="format-from-response",
        ),
        pytest.param(
            {"b64_json": "aW1hZ2U="},
            None,
            "jpeg",
            "data:image/jpeg;base64,aW1hZ2U=",
            id="format-from-request",
        ),
    ],
)
async def test_image_generation_records_the_generated_image(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    is_async: bool,
    response_image: Mapping[str, str],
    response_format: str | None,
    request_format: str | None,
    expected_url: str,
) -> None:
    response_json: dict[str, Any] = {"created": 1, "data": [response_image]}
    if response_format:
        response_json["output_format"] = response_format
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "images/generations")).mock(
        return_value=Response(status_code=200, json=response_json)
    )
    request: dict[str, Any] = {"prompt": "a lighthouse in a storm"}
    if request_format:
        request["output_format"] = request_format

    if is_async:
        await _client(is_async).images.generate(**request)
    else:
        _client(is_async).images.generate(**request)

    attributes = _image_span_attributes(in_memory_span_exporter, request)
    assert attributes.pop(_output_image()) == expected_url
    assert attributes == {}


@pytest.mark.parametrize("is_async", [False, True])
async def test_image_edit_records_the_source_the_mask_and_every_result(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    is_async: bool,
) -> None:
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "images/edits")).mock(
        return_value=Response(
            status_code=200,
            json={
                "created": 1,
                "data": [
                    {"url": "https://example.com/edited-0.png"},
                    {"url": "https://example.com/edited-1.png"},
                ],
            },
        )
    )

    call = _client(is_async).images.edit(
        image=("source.webp", _WEBP_BYTES, "image/webp"),
        mask=("mask.png", _PNG_BYTES, "image/png"),
        prompt="remove the background",
    )
    if is_async:
        await call

    attributes = _image_span_attributes(
        in_memory_span_exporter,
        {"prompt": "remove the background"},
    )
    assert attributes.pop(_input_image(0)) == _data_url("image/webp", _WEBP_BYTES)
    assert attributes.pop(_input_image(1)) == _data_url("image/png", _PNG_BYTES)
    assert attributes.pop(_output_image(0)) == "https://example.com/edited-0.png"
    assert attributes.pop(_output_image(1)) == "https://example.com/edited-1.png"
    assert attributes == {}


def test_image_edit_records_every_image_in_an_array(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "images/edits")).mock(
        return_value=Response(
            status_code=200,
            json={"created": 1, "data": [{"url": "https://example.com/edited.png"}]},
        )
    )

    _client(False).images.edit(
        image=[("first.png", _PNG_BYTES), ("second.jpg", _JPEG_BYTES)],
        prompt="combine these",
    )

    attributes = _image_span_attributes(in_memory_span_exporter, {"prompt": "combine these"})
    assert attributes.pop(_input_image(0)) == _data_url("image/png", _PNG_BYTES)
    assert attributes.pop(_input_image(1)) == _data_url("image/jpeg", _JPEG_BYTES)
    assert attributes.pop(_output_image()) == "https://example.com/edited.png"
    assert attributes == {}


def test_image_edit_reads_an_image_given_as_a_path(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    tmp_path: Path,
) -> None:
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "images/edits")).mock(
        return_value=Response(
            status_code=200,
            json={"created": 1, "data": [{"url": "https://example.com/edited.webp"}]},
        )
    )
    source_path = tmp_path / "source.webp"
    source_path.write_bytes(_WEBP_BYTES)

    _client(False).images.edit(image=source_path, prompt="remove the background")

    attributes = _image_span_attributes(
        in_memory_span_exporter,
        {"prompt": "remove the background"},
    )
    assert attributes.pop(_input_image()) == _data_url("image/webp", _WEBP_BYTES)
    assert attributes.pop(_output_image()) == "https://example.com/edited.webp"
    assert attributes == {}


def test_image_variation_records_raw_bytes_input_and_base64_output(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "images/variations")).mock(
        return_value=Response(
            status_code=200,
            json={"created": 1, "data": [{"b64_json": "dmFyaWF0aW9u"}]},
        )
    )

    _client(False).images.create_variation(image=_PNG_BYTES, response_format="b64_json")

    attributes = _image_span_attributes(in_memory_span_exporter, {"response_format": "b64_json"})
    assert attributes.pop(_input_image()) == _data_url("image/png", _PNG_BYTES)
    assert attributes.pop(_output_image()) == "data:image/png;base64,dmFyaWF0aW9u"
    assert attributes == {}


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("endpoint", "event", "call", "expected_request", "expected_input_url"),
    [
        pytest.param(
            "images/generations",
            {
                "type": "image_generation.completed",
                "b64_json": "c3RyZWFtZWQ=",
                "output_format": "webp",
            },
            lambda client: client.images.generate(prompt="a lighthouse", stream=True),
            {"prompt": "a lighthouse", "stream": True},
            None,
            id="generation",
        ),
        pytest.param(
            "images/edits",
            {"type": "image_edit.completed", "b64_json": "ZWRpdGVk", "output_format": "jpeg"},
            lambda client: client.images.edit(image=_PNG_BYTES, prompt="edit this", stream=True),
            {"prompt": "edit this", "stream": True},
            _data_url("image/png", _PNG_BYTES),
            id="edit",
        ),
    ],
)
async def test_streaming_records_the_completed_image(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    is_async: bool,
    endpoint: str,
    event: Mapping[str, str],
    call: Callable[[Any], Any],
    expected_request: Mapping[str, Any],
    expected_input_url: str | None,
) -> None:
    respx_mock.post(urljoin(_OPENAI_BASE_URL, endpoint)).mock(
        return_value=Response(
            status_code=200,
            text=f"data: {json.dumps(event)}\n\n",
            headers={"content-type": "text/event-stream"},
        )
    )

    stream = call(_client(is_async))
    if is_async:
        async for _ in await stream:
            pass
    else:
        for _ in stream:
            pass

    attributes = _image_span_attributes(in_memory_span_exporter, expected_request)
    assert attributes.pop(_output_image()) == (
        f"data:image/{event['output_format']};base64,{event['b64_json']}"
    )
    if expected_input_url:
        assert attributes.pop(_input_image()) == expected_input_url
    assert attributes == {}


@pytest.mark.parametrize(
    ("config", "expected_input_image", "expected_output_image"),
    [
        pytest.param(
            TraceConfig(hide_inputs=True),
            None,
            "data:image/png;base64,b3V0cHV0",
            id="hide-inputs",
        ),
        pytest.param(
            TraceConfig(hide_input_images=True),
            None,
            "data:image/png;base64,b3V0cHV0",
            id="hide-input-images",
        ),
        pytest.param(
            TraceConfig(hide_outputs=True),
            _data_url("image/png", _PNG_BYTES),
            None,
            id="hide-outputs",
        ),
        pytest.param(
            TraceConfig(base64_image_max_length=1),
            REDACTED_VALUE,
            REDACTED_VALUE,
            id="base64-limit",
        ),
    ],
)
def test_image_attributes_respect_trace_config(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    instrument_with_config: Callable[[TraceConfig], None],
    config: TraceConfig,
    expected_input_image: str | None,
    expected_output_image: str | None,
) -> None:
    instrument_with_config(config)
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "images/edits")).mock(
        return_value=Response(
            status_code=200,
            json={"created": 1, "data": [{"b64_json": "b3V0cHV0"}]},
        )
    )

    _client(False).images.edit(
        image=_PNG_BYTES,
        prompt="edit this",
        response_format="b64_json",
    )

    span = _openinference_span(in_memory_span_exporter)
    assert span.status.is_ok
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(_input_image()) == expected_input_image
    assert attributes.get(_output_image()) == expected_output_image


def test_responses_image_generation_records_completed_results_only(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "responses")).mock(
        return_value=Response(
            status_code=200,
            json={
                "id": "resp_123",
                "status": "completed",
                "object": "response",
                "model": "gpt-image-1",
                "usage": None,
                "output": [
                    {"id": "ig_1", "type": "image_generation_call", "status": "failed"},
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Generated two images.",
                                "annotations": [],
                            }
                        ],
                    },
                    {
                        "id": "ig_2",
                        "type": "image_generation_call",
                        "status": "completed",
                        "result": "c2Vjb25k",
                    },
                    {
                        "id": "ig_3",
                        "type": "image_generation_call",
                        "status": "completed",
                        "result": "dGhpcmQ=",
                    },
                ],
            },
        )
    )
    image_tool = {"type": "image_generation", "output_format": "webp"}
    request = {
        "input": "Draw two lighthouses",
        "model": "gpt-image-1",
        "tools": [image_tool],
    }

    _client(False).responses.create(**request)

    span = _openinference_span(in_memory_span_exporter, "Response")
    assert span.status.is_ok
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "openai"
    assert attributes.pop(SpanAttributes.LLM_SYSTEM) == "openai"
    assert json.loads(cast(str, attributes.pop(SpanAttributes.INPUT_VALUE))) == request
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value
    assert json.loads(cast(str, attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS))) == {
        "model": "gpt-image-1"
    }
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-image-1"
    assert attributes.pop("llm.input_messages.1.message.role") == "user"
    assert attributes.pop("llm.input_messages.1.message.content") == "Draw two lighthouses"
    assert attributes.pop("llm.output_messages.1.message.role") == "assistant"
    assert attributes.pop("llm.output_messages.1.message.contents.0.message_content.type") == "text"
    assert attributes.pop("llm.output_messages.1.message.contents.0.message_content.text") == (
        "Generated two images."
    )
    assert json.loads(cast(str, attributes.pop("llm.tools.0.tool.json_schema"))) == image_tool
    assert isinstance(attributes.pop(SpanAttributes.OUTPUT_VALUE), str)
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value
    # The failed call is skipped, and the two results that came back are numbered
    # from zero rather than by their position in the output list.
    assert attributes.pop(_output_image(0)) == "data:image/webp;base64,c2Vjb25k"
    assert attributes.pop(_output_image(1)) == "data:image/webp;base64,dGhpcmQ="
    assert attributes == {}


def test_responses_item_format_wins_over_the_requested_tool_format() -> None:
    image_call = SimpleNamespace(
        type="image_generation_call",
        result="aW1hZ2U=",
        output_format="jpeg",
    )

    attributes = dict(
        _ResponsesApiAttributes._get_attributes_from_response_output_item(
            image_call,
            image_format="webp",
        )
    )

    assert attributes == {_output_image(): "data:image/jpeg;base64,aW1hZ2U="}


def test_paginated_responses_are_not_mistaken_for_images(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    respx_mock.get(urljoin(_OPENAI_BASE_URL, "models")).mock(
        return_value=Response(
            status_code=200,
            json={
                "object": "list",
                "data": [{"id": "gpt-4o", "object": "model", "created": 1, "owned_by": "openai"}],
            },
        )
    )

    _client(False).models.list()

    span = _openinference_span(in_memory_span_exporter, "SyncPage[Model]")
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert not [key for key in attributes if key.startswith(SpanAttributes.OUTPUT_IMAGES)]


def test_images_response_from_another_module_keeps_compatibility() -> None:
    class ImagesResponse:
        data = [{"b64_json": "aW1hZ2U="}]
        output_format = "jpeg"

    openai = import_module("openai")
    attributes = dict(
        _ResponseAttributesExtractor(openai).get_attributes_from_response(ImagesResponse(), {})
    )

    assert attributes == {_output_image(): "data:image/jpeg;base64,aW1hZ2U="}


def test_image_files_leave_the_stream_where_they_found_it() -> None:
    source = BytesIO(_PNG_BYTES)

    attributes = dict(get_attributes_from_image_files([("image", source)]))

    assert source.tell() == 0
    assert attributes == {_input_image(): _data_url("image/png", _PNG_BYTES)}


def test_image_files_rewind_the_stream_when_reading_fails() -> None:
    class FailingRead(BytesIO):
        def read(self, *args: Any, **kwargs: Any) -> bytes:
            super().read(1)
            raise OSError("read failed")

    source = FailingRead(_PNG_BYTES)

    attributes = dict(get_attributes_from_image_files([("image", source)]))

    assert source.tell() == 0
    assert attributes == {}


def test_image_files_that_cannot_be_read_leave_no_gap_in_the_indexes() -> None:
    class NonSeekable:
        def seekable(self) -> bool:
            return False

        def read(self) -> bytes:
            raise AssertionError("non-seekable input should not be read")

    attributes = dict(
        get_attributes_from_image_files([("image[]", NonSeekable()), ("image[]", _PNG_BYTES)])
    )

    assert attributes == {_input_image(): _data_url("image/png", _PNG_BYTES)}
