from typing import Any, Callable, Dict, Optional

from groq import Groq
from groq.types.chat import ChatCompletion
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import REDACTED_VALUE, TraceConfig
from openinference.instrumentation.groq import GroqInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

REASONING = "The user wants the capital of France. Paris is the capital."
ANSWER = "Paris."
MODEL = "qwen-qwq-32b"
OUTPUT = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
CONTENTS = f"{OUTPUT}.{MessageAttributes.MESSAGE_CONTENTS}"
TOOL_CALLS = f"{OUTPUT}.{MessageAttributes.MESSAGE_TOOL_CALLS}"


def _make_post(
    message: Dict[str, Any], usage: Optional[Dict[str, Any]] = None
) -> Callable[..., Any]:
    """
    Builds a `_post` stub that returns a chat completion parsed the way the SDK parses a real
    response (`construct`), so on an old SDK unknown fields stay plain dicts/extras and on a
    new one they become typed models. No keyword-argument type ignores needed either way.
    """
    completion = ChatCompletion.construct(
        id="chat_comp_0",
        object="chat.completion",
        created=1722531851,
        model=MODEL,
        system_fingerprint="fp0",
        choices=[dict(finish_reason="stop", index=0, logprobs=None, message=message)],
        usage=usage or dict(completion_tokens=57, prompt_tokens=25, total_tokens=82),
    )

    def post(self: Any, *args: Any, **kwargs: Any) -> Any:
        return completion

    return post


def _run(client: Groq, post: Callable[..., Any]) -> None:
    client.chat.completions._post = post
    client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )


def _attributes(exporter: InMemorySpanExporter) -> Dict[str, Any]:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return dict(spans[0].attributes or {})


def test_reasoning_and_token_details(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_groq_instrumentation: Any,
) -> None:
    _run(
        Groq(api_key="fake-api-key"),
        _make_post(
            message=dict(role="assistant", content=ANSWER, reasoning=REASONING),
            usage=dict(
                completion_tokens=57,
                prompt_tokens=25,
                total_tokens=82,
                completion_tokens_details={"reasoning_tokens": 40},
                prompt_tokens_details={"cached_tokens": 12},
            ),
        ),
    )
    attributes = _attributes(in_memory_span_exporter)

    assert attributes[f"{OUTPUT}.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    assert (
        attributes[f"{CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "reasoning"
    )
    assert attributes[f"{CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == REASONING
    assert attributes[f"{CONTENTS}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "text"
    assert attributes[f"{CONTENTS}.1.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == ANSWER
    # With the answer carried as a content block, the plain content attribute is not repeated.
    assert f"{OUTPUT}.{MessageAttributes.MESSAGE_CONTENT}" not in attributes

    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 25
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 57
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 82
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 12
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] == 40


def test_reasoning_with_tool_call_keeps_ordered_view(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_groq_instrumentation: Any,
) -> None:
    tool_call = dict(
        id="call_0",
        type="function",
        function=dict(name="get_weather", arguments='{"city": "Paris"}'),
    )
    _run(
        Groq(api_key="fake-api-key"),
        _make_post(
            message=dict(
                role="assistant", content=None, reasoning=REASONING, tool_calls=[tool_call]
            )
        ),
    )
    attributes = _attributes(in_memory_span_exporter)

    assert (
        attributes[f"{CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "reasoning"
    )
    assert attributes[f"{CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == REASONING
    # No text block when there is no answer text; the tool call follows the reasoning directly.
    assert attributes[f"{CONTENTS}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "tool_use"
    assert attributes[f"{CONTENTS}.1.{ToolCallAttributes.TOOL_CALL_ID}"] == "call_0"
    assert attributes[f"{CONTENTS}.1.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_weather"
    assert (
        attributes[f"{CONTENTS}.1.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]
        == '{"city": "Paris"}'
    )
    # The flat tool call form is still emitted alongside.
    assert attributes[f"{TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_ID}"] == "call_0"
    assert (
        attributes[f"{TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_weather"
    )
    assert f"{OUTPUT}.{MessageAttributes.MESSAGE_CONTENT}" not in attributes


def test_reasoning_on_input_message(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_groq_instrumentation: Any,
) -> None:
    client = Groq(api_key="fake-api-key")
    client.chat.completions._post = _make_post(message=dict(role="assistant", content="Yes."))
    # Older SDKs do not type `reasoning` on assistant params, newer ones do.
    messages: Any = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": ANSWER, "reasoning": REASONING},
        {"role": "user", "content": "Are you sure?"},
    ]
    client.chat.completions.create(model=MODEL, messages=messages)
    attributes = _attributes(in_memory_span_exporter)

    assistant_turn = f"{SpanAttributes.LLM_INPUT_MESSAGES}.1"
    contents = f"{assistant_turn}.{MessageAttributes.MESSAGE_CONTENTS}"
    assert (
        attributes[f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "reasoning"
    )
    assert attributes[f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == REASONING
    assert attributes[f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "text"
    assert attributes[f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == ANSWER
    assert f"{assistant_turn}.{MessageAttributes.MESSAGE_CONTENT}" not in attributes
    # Plain turns keep the plain form.
    user_turn = f"{SpanAttributes.LLM_INPUT_MESSAGES}.2"
    assert attributes[f"{user_turn}.{MessageAttributes.MESSAGE_CONTENT}"] == "Are you sure?"


def test_message_without_reasoning_is_unchanged(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_groq_instrumentation: Any,
) -> None:
    _run(Groq(api_key="fake-api-key"), _make_post(message=dict(role="assistant", content=ANSWER)))
    attributes = _attributes(in_memory_span_exporter)

    assert attributes[f"{OUTPUT}.{MessageAttributes.MESSAGE_CONTENT}"] == ANSWER
    assert not any(MessageAttributes.MESSAGE_CONTENTS in key for key in attributes)
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING not in attributes


def test_reasoning_is_masked_with_hide_output_text(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    GroqInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_output_text=True),
    )
    try:
        _run(
            Groq(api_key="fake-api-key"),
            _make_post(message=dict(role="assistant", content=ANSWER, reasoning=REASONING)),
        )
    finally:
        GroqInstrumentor().uninstrument()
    attributes = _attributes(in_memory_span_exporter)

    assert (
        attributes[f"{CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "reasoning"
    )
    assert (
        attributes[f"{CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"]
        == REDACTED_VALUE
    )
    assert (
        attributes[f"{CONTENTS}.1.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"]
        == REDACTED_VALUE
    )
    assert REASONING not in attributes.values()
