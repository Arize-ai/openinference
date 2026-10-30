import asyncio
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
    assert "llm.cost.total" not in attributes
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


def test_responses_input_typed_reasoning_item_summary_text(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from openai.types.responses import ResponseReasoningItem
    from openai.types.responses.response_reasoning_item import Summary

    from openinference.instrumentation.litellm import LiteLLMInstrumentor

    reasoning_item = ResponseReasoningItem(
        id="rs_123",
        type="reasoning",
        summary=[Summary(type="summary_text", text="The user asks about France.")],
        encrypted_content="enc-abc",
    )

    original_func = LiteLLMInstrumentor.original_litellm_funcs["responses"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["responses"] = lambda *args, **kwargs: None
        litellm.responses(
            model="openai/gpt-4.1",
            api_key=os.getenv("OPENAI_API_KEY", "sk-"),
            input=[reasoning_item],
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["responses"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes: Dict[str, Any] = dict(spans[0].attributes or {})
    assert not any(key.endswith(".message_content.id") for key in attributes)
    prefix = "llm.input_messages.0.message.contents.0.message_content"
    assert attributes.pop(f"{prefix}.type") == "reasoning"
    assert attributes.pop(f"{prefix}.text") == "The user asks about France."
    assert attributes.pop(f"{prefix}.encrypted_content") == "enc-abc"
    assert attributes.pop("llm.input_messages.0.message.role") == "assistant"
    assert attributes.pop("input.mime_type") == "application/json"
    assert isinstance(attributes.pop("input.value"), str)
    assert attributes.pop("llm.invocation_parameters") == "{}"
    assert attributes.pop("llm.model_name") == "openai/gpt-4.1"
    assert attributes.pop("llm.provider") == "openai"
    assert attributes.pop("openinference.span.kind") == "LLM"
    assert attributes == {}


async def test_responses_input_typed_reasoning_item_summary_text_async(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    from openai.types.responses import ResponseReasoningItem
    from openai.types.responses.response_reasoning_item import Summary

    from openinference.instrumentation.litellm import LiteLLMInstrumentor

    reasoning_item = ResponseReasoningItem(
        id="rs_123",
        type="reasoning",
        summary=[Summary(type="summary_text", text="The user asks about France.")],
        encrypted_content="enc-abc",
    )

    original_func = LiteLLMInstrumentor.original_litellm_funcs["aresponses"]

    async def mock_aresponses(*args: Any, **kwargs: Any) -> None:
        return None

    try:
        LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = mock_aresponses
        await litellm.aresponses(
            model="openai/gpt-4.1",
            api_key=os.getenv("OPENAI_API_KEY", "sk-"),
            input=[reasoning_item],
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "aresponses"
    attributes: Dict[str, Any] = dict(spans[0].attributes or {})
    assert not any(key.endswith(".message_content.id") for key in attributes)
    prefix = "llm.input_messages.0.message.contents.0.message_content"
    assert attributes.pop(f"{prefix}.type") == "reasoning"
    assert attributes.pop(f"{prefix}.text") == "The user asks about France."
    assert attributes.pop(f"{prefix}.encrypted_content") == "enc-abc"
    assert attributes.pop("llm.input_messages.0.message.role") == "assistant"
    assert attributes.pop("input.mime_type") == "application/json"
    assert isinstance(attributes.pop("input.value"), str)
    assert attributes.pop("llm.invocation_parameters") == "{}"
    assert attributes.pop("llm.model_name") == "openai/gpt-4.1"
    assert attributes.pop("llm.provider") == "openai"
    assert attributes.pop("openinference.span.kind") == "LLM"
    assert attributes == {}


@pytest.mark.vcr
def test_responses_input_typed_reasoning_item(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    response1 = litellm.responses(
        model="openai/gpt-5-mini",
        api_key=os.getenv("OPENAI_API_KEY", "sk-"),
        input="What is 2 + 2?",
        reasoning={"summary": "auto"},
    )
    reasoning_item = next(
        item for item in response1.output if getattr(item, "type", None) == "reasoning"
    )
    assert reasoning_item.encrypted_content

    response2 = litellm.responses(
        model="openai/gpt-5-mini",
        api_key=os.getenv("OPENAI_API_KEY", "sk-"),
        input=list(response1.output) + [{"role": "user", "content": "Now multiply that by 10."}],
        reasoning={"summary": "auto"},
    )
    reasoning_item2 = next(
        item for item in response2.output if getattr(item, "type", None) == "reasoning"
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    attributes: Dict[str, Any] = dict(spans[1].attributes or {})
    assert not any(key.endswith(".message_content.id") for key in attributes)

    in_prefix = "llm.input_messages"
    assert attributes.pop(f"{in_prefix}.0.message.role") == "assistant"
    assert attributes.pop(f"{in_prefix}.0.message.contents.0.message_content.type") == "reasoning"
    assert (
        attributes.pop(f"{in_prefix}.0.message.contents.0.message_content.text")
        == reasoning_item.summary[0].text
    )
    assert (
        attributes.pop(f"{in_prefix}.0.message.contents.0.message_content.encrypted_content")
        == reasoning_item.encrypted_content
    )
    assert attributes.pop(f"{in_prefix}.1.message.role") == "assistant"
    assert attributes.pop(f"{in_prefix}.2.message.role") == "user"
    assert attributes.pop(f"{in_prefix}.2.message.content") == "Now multiply that by 10."

    assert attributes.pop("input.mime_type") == "application/json"
    assert isinstance(attributes.pop("input.value"), str)
    assert attributes.pop("llm.invocation_parameters") == '{"reasoning": {"summary": "auto"}}'
    assert attributes.pop("llm.model_name") == "gpt-5-mini-2025-08-07"
    assert attributes.pop("llm.provider") == "openai"
    assert attributes.pop("openinference.span.kind") == "LLM"

    out_prefix = "llm.output_messages"
    assert attributes.pop(f"{out_prefix}.0.message.role") == "assistant"
    assert attributes.pop(f"{out_prefix}.0.message.contents.0.message_content.type") == "reasoning"
    assert (
        attributes.pop(f"{out_prefix}.0.message.contents.0.message_content.text")
        == reasoning_item2.summary[0].text
    )
    assert (
        attributes.pop(f"{out_prefix}.0.message.contents.0.message_content.encrypted_content")
        == reasoning_item2.encrypted_content
    )
    assert attributes.pop(f"{out_prefix}.1.message.role") == "assistant"
    assert attributes.pop(f"{out_prefix}.1.message.contents.0.message_content.type") == "text"
    assert attributes.pop(f"{out_prefix}.1.message.contents.0.message_content.text") == "40"

    assert attributes.pop("llm.token_count.completion") == 82
    assert attributes.pop("llm.token_count.completion_details.reasoning") == 64
    assert attributes.pop("llm.token_count.prompt") == 20
    cache_read = attributes.pop("llm.token_count.prompt_details.cache_read", None)
    if cache_read is not None:
        assert cache_read == 0
    assert attributes.pop("llm.token_count.total") == 102
    assert attributes.pop("llm.cost.total") > 0
    assert attributes.pop("output.mime_type") == "application/json"
    assert isinstance(attributes.pop("output.value"), str)
    assert attributes == {}


def _bridged_completion_response() -> litellm.ModelResponse:
    # A ModelResponse mock (not a str) skips the Responses-level mock, so LiteLLM
    # bridges the call to chat completions as it does for providers without
    # native Responses API support.
    return litellm.ModelResponse(
        model="gemini-2.5-flash",
        choices=[
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello"},
            }
        ],
        usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    )


def test_responses_bridged_to_completion_creates_single_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    litellm.responses(
        model="gemini/gemini-2.5-flash",
        api_key="fake-key",
        input="hi",
        mock_response=_bridged_completion_response(),
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["responses"]
    attributes = dict(spans[0].attributes or {})
    assert attributes["llm.model_name"] == "gemini-2.5-flash"
    assert attributes["llm.token_count.total"] == 4


async def test_aresponses_bridged_to_acompletion_creates_single_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    await litellm.aresponses(
        model="gemini/gemini-2.5-flash",
        api_key="fake-key",
        input="hi",
        mock_response=_bridged_completion_response(),
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["aresponses"]
    attributes = dict(spans[0].attributes or {})
    assert attributes["llm.model_name"] == "gemini-2.5-flash"
    assert attributes["llm.token_count.total"] == 4


async def test_acompletion_after_bridged_aresponses_is_still_traced(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    await litellm.aresponses(
        model="gemini/gemini-2.5-flash",
        api_key="fake-key",
        input="hi",
        mock_response=_bridged_completion_response(),
    )
    await litellm.acompletion(
        model="gemini/gemini-2.5-flash",
        api_key="fake-key",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["aresponses", "acompletion"]


def test_streamed_responses_bridged_to_completion_creates_single_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    events = litellm.responses(
        model="gemini/gemini-2.5-flash",
        api_key="fake-key",
        input="hi",
        mock_response=_bridged_completion_response(),
        stream=True,
    )
    list(events)
    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["responses"]
    assert spans[0].status.status_code.name == "OK"
    assert (spans[0].attributes or {})["llm.token_count.total"] > 0


async def test_streamed_aresponses_bridged_to_acompletion_creates_single_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    events = await litellm.aresponses(
        model="gemini/gemini-2.5-flash",
        api_key="fake-key",
        input="hi",
        mock_response=_bridged_completion_response(),
        stream=True,
    )
    async for _ in events:
        pass
    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["aresponses"]
    assert spans[0].status.status_code.name == "OK"
    assert (spans[0].attributes or {})["llm.token_count.total"] > 0


def test_completion_after_failed_responses_is_still_traced(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_responses(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("responses failed")

    monkeypatch.setitem(LiteLLMInstrumentor.original_litellm_funcs, "responses", fail_responses)
    with pytest.raises(RuntimeError, match="responses failed"):
        litellm.responses(model="gemini/gemini-2.5-flash", input="hi")

    litellm.completion(
        model="gemini/gemini-2.5-flash",
        api_key="fake-key",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["responses", "completion"]


async def test_acompletion_after_failed_aresponses_is_still_traced(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_aresponses(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("aresponses failed")

    monkeypatch.setitem(LiteLLMInstrumentor.original_litellm_funcs, "aresponses", fail_aresponses)
    with pytest.raises(RuntimeError, match="aresponses failed"):
        await litellm.aresponses(model="gemini/gemini-2.5-flash", input="hi")

    await litellm.acompletion(
        model="gemini/gemini-2.5-flash",
        api_key="fake-key",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hello",
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["aresponses", "acompletion"]


async def test_concurrent_acompletion_is_traced_during_aresponses(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def paused_aresponses(*args: Any, **kwargs: Any) -> None:
        entered.set()
        await release.wait()

    monkeypatch.setitem(LiteLLMInstrumentor.original_litellm_funcs, "aresponses", paused_aresponses)
    response_task = asyncio.create_task(
        litellm.aresponses(model="gemini/gemini-2.5-flash", input="hi")
    )
    await entered.wait()
    try:
        await litellm.acompletion(
            model="gemini/gemini-2.5-flash",
            api_key="fake-key",
            messages=[{"role": "user", "content": "hi"}],
            mock_response="hello",
        )
    finally:
        release.set()
        await response_task

    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["acompletion", "aresponses"]
