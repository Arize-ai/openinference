import base64
import json
from importlib import import_module
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, cast
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


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("response_image", "response_format", "request_format", "expected_url"),
    [
        pytest.param(
            {"url": "https://example.com/generated.png"},
            None,
            None,
            "https://example.com/generated.png",
            id="url",
        ),
        pytest.param(
            {"b64_json": "aW1hZ2U="},
            "webp",
            None,
            "data:image/webp;base64,aW1hZ2U=",
            id="base64-response-format",
        ),
        pytest.param(
            {"b64_json": "aW1hZ2U="},
            None,
            "jpeg",
            "data:image/jpeg;base64,aW1hZ2U=",
            id="base64-request-format",
        ),
    ],
)
async def test_image_generation_output_images(
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
    openai = import_module("openai")
    client = (
        openai.AsyncOpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL)
        if is_async
        else openai.OpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL)
    )
    kwargs: dict[str, Any] = {"prompt": "a lighthouse in a storm"}
    if request_format:
        kwargs["output_format"] = request_format

    if is_async:
        await client.images.generate(**kwargs)
    else:
        client.images.generate(**kwargs)

    attributes = _image_span_attributes(in_memory_span_exporter, kwargs)
    assert (
        attributes.pop(f"{SpanAttributes.OUTPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}")
        == expected_url
    )
    assert attributes == {}


@pytest.mark.parametrize("is_async", [False, True])
async def test_image_edit_input_and_output_images(
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
    source = b"RIFF\x00\x00\x00\x00WEBPsource"
    mask = b"\x89PNG\r\n\x1a\nmask"
    openai = import_module("openai")
    client = (
        openai.AsyncOpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL)
        if is_async
        else openai.OpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL)
    )

    request = client.images.edit(
        image=("source.webp", source, "image/webp"),
        mask=("mask.png", mask, "image/png"),
        prompt="remove the background",
    )
    if is_async:
        await request

    attributes = _image_span_attributes(
        in_memory_span_exporter,
        {"prompt": "remove the background"},
    )
    assert attributes.pop(f"{SpanAttributes.INPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}") == (
        f"data:image/webp;base64,{base64.b64encode(source).decode('ascii')}"
    )
    assert attributes.pop(f"{SpanAttributes.INPUT_IMAGES}.1.{ImageAttributes.IMAGE_URL}") == (
        f"data:image/png;base64,{base64.b64encode(mask).decode('ascii')}"
    )
    assert attributes.pop(f"{SpanAttributes.OUTPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}") == (
        "https://example.com/edited-0.png"
    )
    assert attributes.pop(f"{SpanAttributes.OUTPUT_IMAGES}.1.{ImageAttributes.IMAGE_URL}") == (
        "https://example.com/edited-1.png"
    )
    assert attributes == {}


def test_image_variation_input_image_and_base64_output(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "images/variations")).mock(
        return_value=Response(
            status_code=200,
            json={"created": 1, "data": [{"b64_json": "dmFyaWF0aW9u"}]},
        )
    )
    source = b"\x89PNG\r\n\x1a\nsource"
    openai = import_module("openai")
    client = openai.OpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL)

    client.images.create_variation(image=source, response_format="b64_json")

    attributes = _image_span_attributes(
        in_memory_span_exporter,
        {"response_format": "b64_json"},
    )
    assert attributes.pop(f"{SpanAttributes.INPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}") == (
        f"data:image/png;base64,{base64.b64encode(source).decode('ascii')}"
    )
    assert attributes.pop(f"{SpanAttributes.OUTPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}") == (
        "data:image/png;base64,dmFyaWF0aW9u"
    )
    assert attributes == {}


def test_image_edit_image_array_inputs(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "images/edits")).mock(
        return_value=Response(
            status_code=200,
            json={"created": 1, "data": [{"url": "https://example.com/edited.png"}]},
        )
    )
    first = b"\x89PNG\r\n\x1a\nfirst"
    second = b"\xff\xd8\xffsecond"
    openai = import_module("openai")

    openai.OpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL).images.edit(
        image=[("first.png", first), ("second.jpg", second)],
        prompt="combine these",
    )

    attributes = _image_span_attributes(in_memory_span_exporter, {"prompt": "combine these"})
    assert attributes.pop(f"{SpanAttributes.INPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}") == (
        f"data:image/png;base64,{base64.b64encode(first).decode('ascii')}"
    )
    assert attributes.pop(f"{SpanAttributes.INPUT_IMAGES}.1.{ImageAttributes.IMAGE_URL}") == (
        f"data:image/jpeg;base64,{base64.b64encode(second).decode('ascii')}"
    )
    assert attributes.pop(f"{SpanAttributes.OUTPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}") == (
        "https://example.com/edited.png"
    )
    assert attributes == {}


def test_image_file_attribute_extraction_restores_file_position() -> None:
    source_bytes = b"\x89PNG\r\n\x1a\nsource"
    source = BytesIO(source_bytes)
    source.seek(0)

    attributes = dict(get_attributes_from_image_files([("image", source)]))
    expected_url = f"data:image/png;base64,{base64.b64encode(source_bytes).decode('ascii')}"

    assert source.tell() == 0
    assert attributes == {
        f"{SpanAttributes.INPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}": (expected_url)
    }


@pytest.mark.parametrize("is_async", [False, True])
async def test_streaming_image_generation_output_image(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    is_async: bool,
) -> None:
    completed_event = {
        "type": "image_generation.completed",
        "b64_json": "c3RyZWFtZWQ=",
        "background": "opaque",
        "created_at": 1,
        "output_format": "webp",
        "quality": "medium",
        "size": "1024x1024",
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"image_tokens": 0, "text_tokens": 1},
            "output_tokens": 1,
            "total_tokens": 2,
        },
    }
    content = f"data: {json.dumps(completed_event)}\n\n"
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "images/generations")).mock(
        return_value=Response(
            status_code=200, text=content, headers={"content-type": "text/event-stream"}
        )
    )
    openai = import_module("openai")
    client = (
        openai.AsyncOpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL)
        if is_async
        else openai.OpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL)
    )

    stream = (
        await client.images.generate(prompt="a lighthouse", stream=True)
        if is_async
        else client.images.generate(prompt="a lighthouse", stream=True)
    )
    if is_async:
        async for _ in stream:
            pass
    else:
        for _ in stream:
            pass

    attributes = _image_span_attributes(
        in_memory_span_exporter,
        {"prompt": "a lighthouse", "stream": True},
    )
    assert attributes.pop(f"{SpanAttributes.OUTPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}") == (
        "data:image/webp;base64,c3RyZWFtZWQ="
    )
    assert attributes == {}


@pytest.mark.parametrize("is_async", [False, True])
async def test_streaming_image_edit_output_image(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    is_async: bool,
) -> None:
    completed_event = {
        "type": "image_edit.completed",
        "b64_json": "ZWRpdGVk",
        "background": "opaque",
        "created_at": 1,
        "output_format": "jpeg",
        "quality": "medium",
        "size": "1024x1024",
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"image_tokens": 0, "text_tokens": 1},
            "output_tokens": 1,
            "total_tokens": 2,
        },
    }
    content = f"data: {json.dumps(completed_event)}\n\n"
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "images/edits")).mock(
        return_value=Response(
            status_code=200, text=content, headers={"content-type": "text/event-stream"}
        )
    )
    source = b"\x89PNG\r\n\x1a\nsource"
    openai = import_module("openai")
    client = (
        openai.AsyncOpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL)
        if is_async
        else openai.OpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL)
    )

    stream = client.images.edit(image=source, prompt="edit this", stream=True)
    if is_async:
        async for _ in await stream:
            pass
    else:
        for _ in stream:
            pass

    attributes = _image_span_attributes(
        in_memory_span_exporter,
        {"prompt": "edit this", "stream": True},
    )
    assert attributes.pop(f"{SpanAttributes.OUTPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}") == (
        "data:image/jpeg;base64,ZWRpdGVk"
    )
    assert attributes.pop(f"{SpanAttributes.INPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}")
    assert attributes == {}


def test_image_edit_path_input(
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
    source = b"RIFF\x00\x00\x00\x00WEBPsource"
    source_path = tmp_path / "source.webp"
    source_path.write_bytes(source)
    openai = import_module("openai")

    openai.OpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL).images.edit(
        image=source_path,
        prompt="remove the background",
    )

    attributes = _image_span_attributes(
        in_memory_span_exporter,
        {"prompt": "remove the background"},
    )
    assert (
        attributes.pop(f"{SpanAttributes.INPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}")
        == f"data:image/webp;base64,{base64.b64encode(source).decode('ascii')}"
    )
    assert (
        attributes.pop(f"{SpanAttributes.OUTPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}")
        == "https://example.com/edited.webp"
    )
    assert attributes == {}


def test_tuple_wrapped_path_input(tmp_path: Path) -> None:
    source = b"RIFF\x00\x00\x00\x00WEBPsource"
    source_path = tmp_path / "source.webp"
    source_path.write_bytes(source)

    attributes = dict(
        get_attributes_from_image_files([("image", ("source.webp", source_path, "image/webp"))])
    )

    assert attributes == {
        f"{SpanAttributes.INPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}": (
            f"data:image/webp;base64,{base64.b64encode(source).decode('ascii')}"
        )
    }


def test_skipped_image_inputs_keep_indexes_dense() -> None:
    class NonSeekable:
        def seekable(self) -> bool:
            return False

        def read(self) -> bytes:
            raise AssertionError("non-seekable input should not be read")

    valid = b"\x89PNG\r\n\x1a\nvalid"
    attributes = dict(
        get_attributes_from_image_files([("image[]", NonSeekable()), ("image[]", valid)])
    )

    assert attributes == {
        f"{SpanAttributes.INPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}": (
            f"data:image/png;base64,{base64.b64encode(valid).decode('ascii')}"
        )
    }


@pytest.mark.parametrize(
    ("config", "expected_input_image", "expected_output_image"),
    [
        pytest.param(TraceConfig(hide_inputs=True), None, "output", id="hide-inputs"),
        pytest.param(TraceConfig(hide_input_images=True), None, "output", id="hide-input-images"),
        pytest.param(TraceConfig(hide_outputs=True), "input", None, id="hide-outputs"),
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
    tracer_provider: trace_api.TracerProvider,
    config: TraceConfig,
    expected_input_image: str | None,
    expected_output_image: str | None,
) -> None:
    OpenAIInstrumentor().uninstrument()
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "images/edits")).mock(
        return_value=Response(
            status_code=200,
            json={"created": 1, "data": [{"b64_json": "b3V0cHV0"}]},
        )
    )
    source = b"\x89PNG\r\n\x1a\ninput"
    openai = import_module("openai")

    openai.OpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL).images.edit(
        image=source,
        prompt="edit this",
        response_format="b64_json",
    )

    span = _openinference_span(in_memory_span_exporter)
    assert span.status.is_ok
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "openai"
    assert attributes.pop(SpanAttributes.LLM_SYSTEM) == "openai"
    assert json.loads(cast(str, attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS))) == {
        "prompt": "edit this",
        "response_format": "b64_json",
    }
    input_image_key = f"{SpanAttributes.INPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}"
    output_image_key = f"{SpanAttributes.OUTPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}"
    if config.hide_inputs:
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == REDACTED_VALUE
        assert SpanAttributes.INPUT_MIME_TYPE not in attributes
    else:
        assert isinstance(attributes.pop(SpanAttributes.INPUT_VALUE), str)
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    if expected_input_image is None:
        assert input_image_key not in attributes
    elif expected_input_image == REDACTED_VALUE:
        assert attributes.pop(input_image_key) == REDACTED_VALUE
    else:
        assert cast(str, attributes.pop(input_image_key)).startswith("data:image/png;base64,")
    if config.hide_outputs:
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == REDACTED_VALUE
        assert SpanAttributes.OUTPUT_MIME_TYPE not in attributes
    else:
        assert isinstance(attributes.pop(SpanAttributes.OUTPUT_VALUE), str)
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    if expected_output_image is None:
        assert output_image_key not in attributes
    elif expected_output_image == REDACTED_VALUE:
        assert attributes.pop(output_image_key) == REDACTED_VALUE
    else:
        assert cast(str, attributes.pop(output_image_key)).startswith("data:image/png;base64,")
    assert attributes == {}
    OpenAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


def test_responses_image_generation_output_images(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    response_json = {
        "id": "resp_123",
        "status": "completed",
        "object": "response",
        "output": [
            {
                "id": "ig_1",
                "type": "image_generation_call",
                "status": "failed",
                "result": None,
            },
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
        "model": "gpt-image-1",
        "usage": None,
    }
    respx_mock.post(urljoin(_OPENAI_BASE_URL, "responses")).mock(
        return_value=Response(status_code=200, json=response_json)
    )
    image_tool = {"type": "image_generation", "output_format": "webp"}
    request = {
        "input": "Draw two lighthouses",
        "model": "gpt-image-1",
        "tools": [image_tool],
    }
    openai = import_module("openai")

    openai.OpenAI(api_key="sk-", base_url=_OPENAI_BASE_URL).responses.create(**request)

    span = _openinference_span(in_memory_span_exporter, "Response")
    assert span.status.is_ok
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "openai"
    assert attributes.pop(SpanAttributes.LLM_SYSTEM) == "openai"
    assert json.loads(cast(str, attributes.pop(SpanAttributes.INPUT_VALUE))) == request
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
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
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert (
        attributes.pop(f"{SpanAttributes.OUTPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}")
        == "data:image/webp;base64,c2Vjb25k"
    )
    assert (
        attributes.pop(f"{SpanAttributes.OUTPUT_IMAGES}.1.{ImageAttributes.IMAGE_URL}")
        == "data:image/webp;base64,dGhpcmQ="
    )
    assert attributes == {}


def test_responses_image_generation_item_format_takes_precedence() -> None:
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

    assert attributes == {
        f"{SpanAttributes.OUTPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}": (
            "data:image/jpeg;base64,aW1hZ2U="
        )
    }


def test_images_response_shape_fallback_extracts_base64() -> None:
    class ImagesResponse:
        data = [{"b64_json": "aW1hZ2U="}]
        output_format = "jpeg"

    openai = import_module("openai")
    attributes = dict(
        _ResponseAttributesExtractor(openai).get_attributes_from_response(ImagesResponse(), {})
    )

    assert attributes == {
        f"{SpanAttributes.OUTPUT_IMAGES}.0.{ImageAttributes.IMAGE_URL}": (
            "data:image/jpeg;base64,aW1hZ2U="
        )
    }
