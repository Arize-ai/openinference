from importlib import import_module
from typing import Any, Dict

import pytest
from httpx import Response
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respx import MockRouter

from openinference.instrumentation import suppress_tracing

_OPENAI_BASE_URL = "https://api.openai.com/v1/"


@pytest.fixture
def openai_client() -> Any:
    openai = import_module("openai")
    return openai.OpenAI(api_key="sk-test", base_url=_OPENAI_BASE_URL)


@pytest.fixture
def mock_chat_response() -> Dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_suppress_tracing_prevents_span_creation(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    openai_client: Any,
    mock_chat_response: Dict[str, Any],
) -> None:
    respx_mock.post(f"{_OPENAI_BASE_URL}chat/completions").mock(
        return_value=Response(200, json=mock_chat_response)
    )

    with suppress_tracing():
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert response.choices[0].message.content == "Hello!"
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 0


@pytest.mark.asyncio
async def test_suppress_tracing_async_prevents_span_creation(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    mock_chat_response: Dict[str, Any],
) -> None:
    openai = import_module("openai")
    async_client = openai.AsyncOpenAI(api_key="sk-test", base_url=_OPENAI_BASE_URL)

    respx_mock.post(f"{_OPENAI_BASE_URL}chat/completions").mock(
        return_value=Response(200, json=mock_chat_response)
    )

    with suppress_tracing():
        response = await async_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
        )

    assert response.choices[0].message.content == "Hello!"
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 0


def test_tracing_works_outside_suppress_context(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    openai_client: Any,
    mock_chat_response: Dict[str, Any],
) -> None:
    respx_mock.post(f"{_OPENAI_BASE_URL}chat/completions").mock(
        return_value=Response(200, json=mock_chat_response)
    )

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert response.choices[0].message.content == "Hello!"
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0
