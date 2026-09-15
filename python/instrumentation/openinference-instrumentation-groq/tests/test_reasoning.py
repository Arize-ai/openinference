from typing import Any, Dict, Optional, Type, Union, cast

from groq import Groq
from groq._base_client import _StreamT
from groq._types import Body, RequestFiles, RequestOptions, ResponseT
from groq.types import CompletionUsage
from groq.types.chat import ChatCompletion, ChatCompletionMessage
from groq.types.chat.chat_completion import Choice
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)

REASONING = "The user wants the capital of France. Paris is the capital."
ANSWER = "Paris."


def _mock_post(
    self: Any,
    path: str = "fake/url",
    *,
    cast_to: Type[ResponseT],
    body: Optional[Body] = None,
    options: RequestOptions = {},
    files: Optional[RequestFiles] = None,
    stream: bool = False,
    stream_cls: Optional[Type[_StreamT]] = None,
) -> Union[ResponseT, _StreamT]:
    # The extra fields are what groq>=0.20 returns for reasoning models; on the
    # oldest supported SDK the models keep them as extras, which is what a real
    # response parsed by that SDK looks like too.
    completion = ChatCompletion(
        id="chat_comp_0",
        choices=[
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content=ANSWER,
                    role="assistant",
                    reasoning=REASONING,  # type: ignore[call-arg]
                ),
            )
        ],
        created=1722531851,
        model="qwen-qwq-32b",
        object="chat.completion",
        system_fingerprint="fp0",
        usage=CompletionUsage(
            completion_tokens=57,
            prompt_tokens=25,
            total_tokens=82,
            completion_tokens_details={"reasoning_tokens": 40},  # type: ignore[call-arg]
            prompt_tokens_details={"cached_tokens": 12},
        ),
    )
    return cast(ResponseT, completion)


def test_reasoning_and_token_details(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_groq_instrumentation: Any,
) -> None:
    client = Groq(api_key="fake-api-key")
    client.chat.completions._post = _mock_post  # type: ignore[assignment]

    client.chat.completions.create(
        model="qwen-qwq-32b",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes: Dict[str, Any] = dict(spans[0].attributes or {})

    output = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    contents = f"{output}.{MessageAttributes.MESSAGE_CONTENTS}"
    assert attributes[f"{output}.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    assert (
        attributes[f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "reasoning"
    )
    assert attributes[f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == REASONING
    assert attributes[f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "text"
    assert attributes[f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == ANSWER
    # With the answer carried as a content block, the plain content attribute is not repeated.
    assert f"{output}.{MessageAttributes.MESSAGE_CONTENT}" not in attributes

    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 25
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 57
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 82
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 12
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] == 40


def test_message_without_reasoning_is_unchanged(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_groq_instrumentation: Any,
) -> None:
    def plain_post(self: Any, *args: Any, **kwargs: Any) -> Any:
        return ChatCompletion(
            id="chat_comp_1",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(content=ANSWER, role="assistant"),
                )
            ],
            created=1722531851,
            model="llama-3.1-8b-instant",
            object="chat.completion",
            usage=CompletionUsage(completion_tokens=3, prompt_tokens=25, total_tokens=28),
        )

    client = Groq(api_key="fake-api-key")
    client.chat.completions._post = plain_post  # type: ignore[assignment]
    client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    attributes: Dict[str, Any] = dict(spans[0].attributes or {})
    output = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes[f"{output}.{MessageAttributes.MESSAGE_CONTENT}"] == ANSWER
    assert not any(MessageAttributes.MESSAGE_CONTENTS in key for key in attributes)
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING not in attributes
