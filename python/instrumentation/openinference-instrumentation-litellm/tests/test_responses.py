import os
from typing import Any, Dict, Iterator

import litellm
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation.litellm import LiteLLMInstrumentor


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
) -> Iterator[None]:
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    yield


@pytest.mark.vcr
def test_responses_simple_input(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    response = litellm.responses(
        model="openai/gpt-4.1",
        api_key=os.getenv("OPENAI_API_KEY", "sk-"),
        input="Write a poem about a boy and his first pet dog.",
        max_tokens=100,
    )
    assert response is not None
    assert response.output is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "responses"
    attributes: Dict[str, Any] = dict(span.attributes or {})
    assert attributes is not None
    assert attributes.pop("input.mime_type") == "text/plain"
    assert attributes.pop("input.value") == "Write a poem about a boy and his first pet dog."
    assert "Write a poem about a boy" in attributes.pop("llm.input_messages.0.message.content")
    assert attributes.pop("llm.input_messages.0.message.role") == "user"
    assert attributes.pop("llm.invocation_parameters") == '{"max_tokens": 100}'
    assert attributes.pop("llm.model_name") == "gpt-4.1-2025-04-14"
    o = "llm.output_messages.0.message"
    assert attributes.pop(f"{o}.contents.0.message_content.text").startswith("In a sunlit")
    assert attributes.pop(f"{o}.contents.0.message_content.type") == "text"
    assert attributes.pop(f"{o}.role") == "assistant"
    assert attributes.pop("llm.provider") == "openai"
    assert attributes.pop("llm.token_count.completion") == 253
    assert attributes.pop("llm.token_count.completion_details.reasoning") == 0
    assert attributes.pop("llm.token_count.prompt") == 19
    cache_read = attributes.pop("llm.token_count.prompt_details.cache_read", None)
    if cache_read is not None:
        assert cache_read == 0
    assert attributes.pop("llm.token_count.total") == 272
    assert attributes.pop("openinference.span.kind") == "LLM"
    assert attributes.pop("output.mime_type") == "application/json"
    assert attributes.pop("output.value").startswith('[{"role": "assistant"')
    assert attributes.pop("llm.cost.total") > 0
    assert attributes == {}


@pytest.mark.vcr
def test_responses_simple_input_stream(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    response = litellm.responses(
        model="openai/gpt-4.1",
        api_key=os.getenv("OPENAI_API_KEY", "sk-"),
        input="Write a poem about a boy and his first pet dog.",
        max_tokens=100,
        stream=True,
    )
    response = list(response)
    assert response is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "responses"
    attributes: Dict[str, Any] = dict(span.attributes or {})
    assert attributes is not None
    assert attributes.pop("input.mime_type") == "text/plain"
    assert attributes.pop("input.value") == "Write a poem about a boy and his first pet dog."
    assert "Write a poem about a boy" in attributes.pop("llm.input_messages.0.message.content")
    assert attributes.pop("llm.input_messages.0.message.role") == "user"
    assert attributes.pop("llm.invocation_parameters") == '{"max_tokens": 100, "stream": true}'
    assert attributes.pop("llm.model_name") == "gpt-4.1-2025-04-14"
    o = "llm.output_messages.0.message"
    assert "Underneath a summer sky" in attributes.pop(f"{o}.contents.0.message_content.text")
    assert attributes.pop(f"{o}.contents.0.message_content.type") == "text"
    assert attributes.pop(f"{o}.role") == "assistant"
    assert attributes.pop("llm.provider") == "openai"
    assert attributes.pop("llm.token_count.completion") == 262
    assert attributes.pop("llm.token_count.completion_details.reasoning") == 0
    assert attributes.pop("llm.token_count.prompt") == 19
    cache_read = attributes.pop("llm.token_count.prompt_details.cache_read", None)
    if cache_read is not None:
        assert cache_read == 0
    assert attributes.pop("llm.token_count.total") == 281
    assert attributes.pop("openinference.span.kind") == "LLM"
    assert attributes.pop("output.mime_type") == "application/json"
    assert attributes.pop("output.value").startswith('[{"role": "assistant"')
    # streams will not have cost unless litellm.include_cost_in_streaming_usage is set to True
    attributes.pop("llm.cost.total", None)
    assert attributes == {}


@pytest.mark.vcr
def test_responses_websearch_input(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    response = litellm.responses(
        model="openai/gpt-4.1",
        api_key=os.getenv("OPENAI_API_KEY", "sk-"),
        tools=[{"type": "web_search_preview"}],
        input="What was a positive news story from today?",
    )
    response = list(response)
    assert response is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "responses"
    attributes: Dict[str, Any] = dict(span.attributes or {})
    assert attributes is not None
    assert attributes.pop("input.mime_type") == "text/plain"
    assert attributes.pop("input.value") == "What was a positive news story from today?"
    assert "What was a positive news" in attributes.pop("llm.input_messages.0.message.content")
    assert attributes.pop("llm.input_messages.0.message.role") == "user"
    assert attributes.pop("llm.invocation_parameters") == "{}"
    assert attributes.pop("llm.model_name") == "gpt-4.1-2025-04-14"
    lom = "llm.output_messages"
    wot = "message.tool_calls"
    assert attributes.pop(f"{lom}.0.message.role") == "assistant"
    assert attributes.pop(f"{lom}.0.{wot}.0.tool_call.function.name") == "web_search_call"
    assert "ws_008" in attributes.pop(f"{lom}.0.{wot}.0.tool_call.id")
    assert attributes.pop(f"{lom}.1.message.contents.0.message_content.text").startswith(
        "As of February 25, 2026, a notable positive development"
    )
    assert attributes.pop(f"{lom}.1.message.contents.0.message_content.type") == "text"
    assert attributes.pop(f"{lom}.1.message.role") == "assistant"
    assert attributes.pop("llm.provider") == "openai"
    assert attributes.pop("llm.token_count.completion") == 290
    assert attributes.pop("llm.token_count.completion_details.reasoning") == 0
    assert attributes.pop("llm.token_count.prompt") == 310
    cache_read = attributes.pop("llm.token_count.prompt_details.cache_read", None)
    if cache_read is not None:
        assert cache_read == 0
    assert attributes.pop("llm.token_count.total") == 600
    assert attributes.pop("llm.tools.0.tool.json_schema") == '{"type": "web_search_preview"}'
    assert attributes.pop("openinference.span.kind") == "LLM"
    assert attributes.pop("output.mime_type") == "application/json"
    assert attributes.pop("output.value").startswith('[{"role": "assistant"')
    assert attributes.pop("llm.cost.total") > 0
    assert attributes == {}


@pytest.mark.vcr
async def test_responses_websearch_input_async(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    response = await litellm.aresponses(
        model="openai/gpt-4.1",
        api_key=os.getenv("OPENAI_API_KEY", "sk-"),
        tools=[{"type": "web_search_preview"}],
        input="What was a positive news story from today?",
    )
    response = list(response)
    assert response is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "aresponses"
    attributes: Dict[str, Any] = dict(span.attributes or {})
    assert attributes is not None
    assert attributes.pop("input.mime_type") == "text/plain"
    assert attributes.pop("input.value") == "What was a positive news story from today?"
    assert "What was a positive news" in attributes.pop("llm.input_messages.0.message.content")
    assert attributes.pop("llm.input_messages.0.message.role") == "user"
    assert attributes.pop("llm.invocation_parameters") == "{}"
    assert attributes.pop("llm.model_name") == "gpt-4.1-2025-04-14"
    lom = "llm.output_messages"
    wot = "message.tool_calls"
    assert attributes.pop(f"{lom}.0.message.role") == "assistant"
    assert attributes.pop(f"{lom}.0.{wot}.0.tool_call.function.name") == "web_search_call"
    assert "ws_010" in attributes.pop(f"{lom}.0.{wot}.0.tool_call.id")
    assert attributes.pop(f"{lom}.1.message.contents.0.message_content.text").startswith(
        "As of February 25, 2026, a notable positive news story"
    )
    assert attributes.pop(f"{lom}.1.message.contents.0.message_content.type") == "text"
    assert attributes.pop(f"{lom}.1.message.role") == "assistant"
    assert attributes.pop("llm.provider") == "openai"
    assert attributes.pop("llm.token_count.completion") == 364
    assert attributes.pop("llm.token_count.completion_details.reasoning") == 0
    assert attributes.pop("llm.token_count.prompt") == 310
    cache_read = attributes.pop("llm.token_count.prompt_details.cache_read", None)
    if cache_read is not None:
        assert cache_read == 0
    assert attributes.pop("llm.token_count.total") == 674
    assert attributes.pop("llm.tools.0.tool.json_schema") == '{"type": "web_search_preview"}'
    assert attributes.pop("openinference.span.kind") == "LLM"
    assert attributes.pop("output.mime_type") == "application/json"
    assert attributes.pop("output.value").startswith('[{"role": "assistant"')
    assert attributes.pop("llm.cost.total") > 0
    assert attributes == {}
