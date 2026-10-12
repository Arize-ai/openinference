import json
from typing import Any

import httpx
import pytest
from google import genai
from google.genai import types
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import MessageAttributes, MessageContentAttributes, SpanAttributes

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
        chunks = client.models.generate_content_stream(**kwargs)
        assert [chunk.text for chunk in chunks] == ["reply"]
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
    ],
)
async def test_tuple_contents_across_call_paths(
    contents: tuple[Any, ...],
    expected_messages: dict[str, str],
    expected_wire_contents: list[dict[str, Any]],
    asynchronous: bool,
    streaming: bool,
    mock_client: tuple[genai.Client, list[dict[str, Any]]],
    in_memory_span_exporter: InMemorySpanExporter,
    setup_google_genai_instrumentation: None,
) -> None:
    client, requests = mock_client
    try:
        await _call(client, asynchronous, streaming, model="gemini-2.0-flash", contents=contents)
    except ValueError as exc:
        # google-genai 2.0.1 sends the request but rejects tuples during stream assembly.
        assert streaming and "Unsupported content part type" in str(exc)

    assert len(requests) == 1
    assert requests[0]["contents"] == expected_wire_contents
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert _input_messages(dict(span.attributes or {})) == expected_messages


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("streaming", [False, True], ids=["non_streaming", "streaming"])
async def test_empty_tuple_contents_across_call_paths(
    asynchronous: bool,
    streaming: bool,
    mock_client: tuple[genai.Client, list[dict[str, Any]]],
    in_memory_span_exporter: InMemorySpanExporter,
    setup_google_genai_instrumentation: None,
) -> None:
    client, requests = mock_client
    with pytest.raises(ValueError, match="contents are required"):
        await _call(client, asynchronous, streaming, model="gemini-2.0-flash", contents=())

    assert requests == []
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert _input_messages(dict(span.attributes or {})) == {}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("streaming", [False, True], ids=["non_streaming", "streaming"])
@pytest.mark.parametrize(
    "system_instruction, expected_messages",
    [
        pytest.param(
            ("Be concise.", types.Part.from_text(text="Use plain language.")),
            {
                **_message(0, "system", "Be concise.", "Use plain language."),
                **_message(1, "user", "hello"),
            },
            id="mixed_system_tuple",
        ),
        pytest.param((), _message(0, "user", "hello"), id="empty_system_tuple"),
    ],
)
async def test_tuple_system_instruction_across_call_paths(
    system_instruction: tuple[Any, ...],
    expected_messages: dict[str, str],
    asynchronous: bool,
    streaming: bool,
    mock_client: tuple[genai.Client, list[dict[str, Any]]],
    in_memory_span_exporter: InMemorySpanExporter,
    setup_google_genai_instrumentation: None,
) -> None:
    client, requests = mock_client
    kwargs = {
        "model": "gemini-2.0-flash",
        "contents": ["hello"],
        "config": {"system_instruction": system_instruction},
    }
    if not system_instruction:
        with pytest.raises(ValueError, match="content parts are required"):
            await _call(client, asynchronous, streaming, **kwargs)
        assert requests == []
    else:
        await _call(client, asynchronous, streaming, **kwargs)
        assert len(requests) == 1
        assert requests[0]["systemInstruction"]["parts"] == [
            {"text": "Be concise."},
            {"text": "Use plain language."},
        ]
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert _input_messages(dict(span.attributes or {})) == expected_messages
