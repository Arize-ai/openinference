import json
import subprocess
import sys
from typing import Any

import httpx
import pytest
from google import genai
from google.genai import types
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

_RESPONSE = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "reply"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
}


def _input_messages(attributes: dict[str, Any]) -> dict[str, Any]:
    prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}."
    return {key: value for key, value in attributes.items() if key.startswith(prefix)}


def _message(index: int, role: str, *texts: str) -> dict[str, str]:
    prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}."
    result = {f"{prefix}{MessageAttributes.MESSAGE_ROLE}": role}
    if len(texts) == 1:
        result[f"{prefix}{MessageAttributes.MESSAGE_CONTENT}"] = texts[0]
    else:
        for part_index, text in enumerate(texts):
            part_prefix = f"{prefix}{MessageAttributes.MESSAGE_CONTENTS}.{part_index}."
            result[f"{part_prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] = "text"
            result[f"{part_prefix}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] = text
    return result


async def _call(client: genai.Client, asynchronous: bool, streaming: bool, **kwargs: Any) -> None:
    if asynchronous:
        if streaming:
            stream = await client.aio.models.generate_content_stream(**kwargs)
            chunks = [chunk.text async for chunk in stream]
            assert chunks == ["reply"]
        else:
            response = await client.aio.models.generate_content(**kwargs)
            assert response.text == "reply"
    elif streaming:
        sync_stream = client.models.generate_content_stream(**kwargs)
        assert [chunk.text for chunk in sync_stream] == ["reply"]
    else:
        assert client.models.generate_content(**kwargs).text == "reply"


@pytest.fixture
def mock_client() -> tuple[genai.Client, list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if request.url.path.endswith(":streamGenerateContent"):
            return httpx.Response(200, text=f"data: {json.dumps(_RESPONSE)}\n\n")
        return httpx.Response(200, json=_RESPONSE)

    transport = httpx.MockTransport(respond)
    client = genai.Client(
        api_key="test-key",
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=transport),
            httpx_async_client=httpx.AsyncClient(transport=transport),
        ),
    )
    return client, requests


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("streaming", [False, True], ids=["non_streaming", "streaming"])
@pytest.mark.parametrize(
    "contents, expected_messages, expected_wire_contents",
    [
        pytest.param(
            ("hello",),
            _message(0, "user", "hello"),
            [{"role": "user", "parts": [{"text": "hello"}]}],
            id="single_text_tuple",
        ),
        pytest.param(
            (
                "hello",
                types.Part.from_text(text="world"),
            ),
            _message(0, "user", "hello", "world"),
            [{"role": "user", "parts": [{"text": "hello"}, {"text": "world"}]}],
            id="mixed_text_and_part_tuple",
        ),
        pytest.param(
            (("hello", "world"),),
            _message(0, "user", "hello", "world"),
            [{"role": "user", "parts": [{"text": "hello"}, {"text": "world"}]}],
            id="nested_tuple",
        ),
        pytest.param(
            [("hello", "world")],
            _message(0, "user", "hello", "world"),
            [{"role": "user", "parts": [{"text": "hello"}, {"text": "world"}]}],
            id="list_of_tuple",
        ),
    ],
)
async def test_tuple_contents_across_call_paths(
    contents: Any,
    expected_messages: dict[str, str],
    expected_wire_contents: list[dict[str, Any]],
    asynchronous: bool,
    streaming: bool,
    mock_client: tuple[genai.Client, list[dict[str, Any]]],
    in_memory_span_exporter: InMemorySpanExporter,
    setup_google_genai_instrumentation: None,
) -> None:
    """Tuple contents retain their wire form and input messages on every call path."""
    client, requests = mock_client
    try:
        await _call(client, asynchronous, streaming, model="gemini-2.0-flash", contents=contents)
    except ValueError as exc:
        # google-genai 2.0.1 sends the request but rejects tuples during stream assembly.
        assert streaming and "Unsupported content" in str(exc)

    assert len(requests) == 1
    assert requests[0]["contents"] == expected_wire_contents
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert _input_messages(dict(span.attributes or {})) == expected_messages


def _text_and_image_message(text: str, image_url: str) -> dict[str, str]:
    prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0."
    text_prefix = f"{prefix}{MessageAttributes.MESSAGE_CONTENTS}.0."
    image_prefix = f"{prefix}{MessageAttributes.MESSAGE_CONTENTS}.1."
    image_key = f"{image_prefix}{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}"
    image_url_key = f"{image_key}.{ImageAttributes.IMAGE_URL}"
    return {
        f"{prefix}{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{text_prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}": "text",
        f"{text_prefix}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}": text,
        f"{image_prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}": "image",
        image_url_key: image_url,
    }


@pytest.mark.parametrize(
    "part, expected_wire_key, expected_image_url",
    [
        pytest.param(
            {"inline_data": {"mime_type": "image/png", "data": "aGk="}},
            "inlineData",
            "data:image/png;base64,aGk=",
            id="snake_case_inline_data",
        ),
        pytest.param(
            {"inlineData": {"mimeType": "image/png", "data": "aGk="}},
            "inlineData",
            "data:image/png;base64,aGk=",
            id="camel_case_inline_data",
        ),
        pytest.param(
            {"fileData": {"mimeType": "image/png", "fileUri": "gs://bucket/cat.png"}},
            "fileData",
            "gs://bucket/cat.png",
            id="camel_case_file_data",
        ),
    ],
)
def test_dict_part_key_casing_is_recorded(
    part: dict[str, Any],
    expected_wire_key: str,
    expected_image_url: str,
    mock_client: tuple[genai.Client, list[dict[str, Any]]],
    in_memory_span_exporter: InMemorySpanExporter,
    setup_google_genai_instrumentation: None,
) -> None:
    """Dict image parts are recorded as sent: camelCase keys kept, string data not re-encoded."""
    client, requests = mock_client
    contents: Any = [{"role": "user", "parts": [{"text": "see"}, part]}]
    client.models.generate_content(model="gemini-2.0-flash", contents=contents)

    assert list(requests[0]["contents"][0]["parts"][1]) == [expected_wire_key]
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert _input_messages(dict(span.attributes or {})) == _text_and_image_message(
        "see", expected_image_url
    )


def test_camel_case_function_call_dict_part_is_recorded(
    mock_client: tuple[genai.Client, list[dict[str, Any]]],
    in_memory_span_exporter: InMemorySpanExporter,
    setup_google_genai_instrumentation: None,
) -> None:
    client, requests = mock_client
    contents: Any = [
        {"role": "user", "parts": [{"text": "weather?"}]},
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}],
        },
    ]
    client.models.generate_content(model="gemini-2.0-flash", contents=contents)

    assert requests[0]["contents"][1]["parts"][0] == {
        "functionCall": {"name": "get_weather", "args": {"city": "Paris"}}
    }
    (span,) = in_memory_span_exporter.get_finished_spans()
    tool_call_prefix = (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
    )
    assert _input_messages(dict(span.attributes or {})) == {
        **_message(0, "user", "weather?"),
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{tool_call_prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}": "get_weather",
        f"{tool_call_prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}": json.dumps(
            {"city": "Paris"}
        ),
    }


def test_pil_image_in_tuple_contents_is_recorded(
    mock_client: tuple[genai.Client, list[dict[str, Any]]],
    in_memory_span_exporter: InMemorySpanExporter,
    setup_google_genai_instrumentation: None,
) -> None:
    pil_image = pytest.importorskip("PIL.Image")
    client, requests = mock_client
    contents: Any = ("describe", pil_image.new("RGB", (1, 1)))
    client.models.generate_content(model="gemini-2.0-flash", contents=contents)

    wire_image = requests[0]["contents"][0]["parts"][1]["inlineData"]
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert _input_messages(dict(span.attributes or {})) == _text_and_image_message(
        "describe", f"data:image/png;base64,{wire_image['data']}"
    )


def test_instrumentor_instruments_when_pillow_is_installed() -> None:
    """A fresh interpreter can import and instrument even when Pillow is importable."""
    pytest.importorskip("PIL.Image")
    script = (
        "from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor\n"
        "GoogleGenAIInstrumentor().instrument()\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
