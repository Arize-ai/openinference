from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

import pytest
from google.genai import chats, models, types
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent, LLMCompletionEndEvent
from llama_index.core.llms import ChatResponse, CompletionResponse
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq  # type: ignore[import-untyped]
from llama_index.llms.openai import OpenAI
from llama_index.llms.vertex import Vertex
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.llama_index._handler import _get_token_counts, _Span
from openinference.semconv.trace import SpanAttributes

LLM_FINISH_REASON = SpanAttributes.LLM_FINISH_REASON
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO
LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING = (
    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING
)


class TestTokenCounts:
    @pytest.mark.vcr
    async def test_groq(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        result = await Groq(
            model="llama3-8b-8192",
            api_key="gsk_",
        ).astream_chat([ChatMessage(content="Hello!")])
        async for _ in result:
            pass
        span = in_memory_span_exporter.get_finished_spans()[0]
        assert span.attributes
        assert span.attributes.get(LLM_TOKEN_COUNT_TOTAL)
        assert span.attributes.get(LLM_TOKEN_COUNT_COMPLETION)
        assert span.attributes.get(LLM_TOKEN_COUNT_TOTAL)
        assert span.attributes[LLM_FINISH_REASON] == "stop"

    @pytest.mark.vcr
    def test_openai(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        llm = OpenAI(model="gpt-4o-mini", api_key="sk-")
        resp = llm.chat([ChatMessage(content="Hello!")])
        usage = resp.raw.usage

        span = in_memory_span_exporter.get_finished_spans()[0]
        attr = dict(span.attributes or {})

        assert attr.pop(LLM_TOKEN_COUNT_COMPLETION) == resp.raw.usage.completion_tokens
        assert attr.pop(LLM_TOKEN_COUNT_PROMPT) == resp.raw.usage.prompt_tokens
        assert attr.pop(LLM_TOKEN_COUNT_TOTAL) == resp.raw.usage.total_tokens
        assert (
            attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ)
            == usage.prompt_tokens_details.cached_tokens
        )
        assert (
            attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO)
            == usage.prompt_tokens_details.audio_tokens
        )
        assert (
            attr.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO)
            == usage.completion_tokens_details.audio_tokens
        )
        assert (
            attr.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING)
            == usage.completion_tokens_details.reasoning_tokens
        )

    @pytest.mark.vcr(
        match_on=["method", "scheme", "host", "port", "path"],
    )
    def test_anthropic(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        anthropic_model: str,
    ) -> None:
        llm = Anthropic(model=anthropic_model, api_key="sk-")
        resp = llm.chat([ChatMessage(content="Hello!")])
        span = in_memory_span_exporter.get_finished_spans()[0]
        attr = dict(span.attributes or {})

        usage = resp.raw.get("usage")
        assert (
            attr.pop(LLM_TOKEN_COUNT_PROMPT)
            == usage.input_tokens
            + usage.cache_creation_input_tokens
            + usage.cache_read_input_tokens
        )
        assert attr.pop(LLM_TOKEN_COUNT_COMPLETION) == usage.output_tokens
        assert attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == usage.cache_read_input_tokens
        assert (
            attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE)
            == usage.cache_creation_input_tokens
        )

    @pytest.mark.vcr(
        before_record_request=lambda request: (
            None if "oauth2" in request.uri else (request.headers.clear() or request)
        ),
        match_on=["method", "body"],
    )
    def test_vertex(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        llm = Vertex(model="gemini-1.5-flash-002")
        resp = llm.chat([ChatMessage(content="Hello!")])
        span = in_memory_span_exporter.get_finished_spans()[0]
        attr = dict(span.attributes or {})

        if not resp.raw:
            raise ValueError("No raw response")

        # Get usage_metadata from the raw response
        raw_response = resp.raw.get("_raw_response")

        if not raw_response:
            raise ValueError("No raw response")

        usage_metadata = raw_response.usage_metadata

        assert attr.pop(LLM_TOKEN_COUNT_PROMPT) == usage_metadata.prompt_token_count
        assert attr.pop(LLM_TOKEN_COUNT_COMPLETION) == usage_metadata.candidates_token_count
        assert attr.pop(LLM_TOKEN_COUNT_TOTAL) == usage_metadata.total_token_count


@pytest.mark.parametrize("response_type", ["chat", "completion"])
@pytest.mark.parametrize("raw_shape", ["google-genai", "vertex-dict", "vertex-object", "object"])
@pytest.mark.parametrize("tool_tokens", [0, 30])
@pytest.mark.parametrize("reported_total", [None, 1280, 2180, 3000])
def test_gemini_token_counts_on_exported_span(
    response_type: str,
    raw_shape: str,
    tool_tokens: int,
    reported_total: Any,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    usage = {
        "prompt_token_count": 1200,
        "candidates_token_count": 80,
        "thoughts_token_count": 900,
        "cached_content_token_count": 1024,
        "tool_use_prompt_token_count": tool_tokens,
        "total_token_count": reported_total,
    }
    raw: Any
    if raw_shape == "google-genai":
        raw = {"usage_metadata": usage}
    elif raw_shape == "vertex-dict":
        raw = {"_raw_response": {"usage_metadata": usage}}
    elif raw_shape == "vertex-object":
        raw = {"_raw_response": SimpleNamespace(usage_metadata=SimpleNamespace(**usage))}
    else:
        raw = SimpleNamespace(usage_metadata=SimpleNamespace(**usage))
    # LlamaIndex omits thinking/tool-use tokens from these convenience counts.
    kwargs = {"prompt_tokens": 1200, "completion_tokens": 80, "total_tokens": 1280}
    span = _Span(otel_span=tracer_provider.get_tracer(__name__).start_span("llm"), span_kind="LLM")
    if response_type == "chat":
        span.process_event(
            LLMChatEndEvent(
                messages=[],
                response=ChatResponse(
                    message=ChatMessage(content="Paris."), raw=raw, additional_kwargs=kwargs
                ),
            )
        )
    else:
        span.process_event(
            LLMCompletionEndEvent(
                prompt="Capital of France?",
                response=CompletionResponse(text="Paris.", raw=raw, additional_kwargs=kwargs),
            )
        )
    span.end()

    exported_span = in_memory_span_exporter.get_finished_spans()[0]
    attributes = exported_span.attributes or {}
    assert attributes[LLM_TOKEN_COUNT_PROMPT] == 1200 + tool_tokens
    assert attributes[LLM_TOKEN_COUNT_COMPLETION] == 980
    assert attributes[LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] == 900
    assert attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 1024
    assert attributes[LLM_TOKEN_COUNT_TOTAL] == max(2180 + tool_tokens, reported_total or 0)


@pytest.mark.parametrize(
    "usage, expected",
    [
        ({}, {}),
        ({"prompt_token_count": None, "thoughts_token_count": None}, {}),
        (
            {"prompt_token_count": 0, "candidates_token_count": 0, "total_token_count": 0},
            {LLM_TOKEN_COUNT_PROMPT: 0, LLM_TOKEN_COUNT_COMPLETION: 0, LLM_TOKEN_COUNT_TOTAL: 0},
        ),
        (
            {"thoughts_token_count": 900},
            {
                LLM_TOKEN_COUNT_COMPLETION: 900,
                LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING: 900,
                LLM_TOKEN_COUNT_TOTAL: 900,
            },
        ),
        (
            {"tool_use_prompt_token_count": 30, "cached_content_token_count": 1024},
            {
                LLM_TOKEN_COUNT_PROMPT: 30,
                LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 1024,
                LLM_TOKEN_COUNT_TOTAL: 30,
            },
        ),
        (
            {"prompt_token_count": "1200", "candidates_token_count": "80"},
            {
                LLM_TOKEN_COUNT_PROMPT: 1200,
                LLM_TOKEN_COUNT_COMPLETION: 80,
                LLM_TOKEN_COUNT_TOTAL: 1280,
            },
        ),
        (
            {"prompt_token_count": "invalid", "candidates_token_count": 80},
            {LLM_TOKEN_COUNT_COMPLETION: 80, LLM_TOKEN_COUNT_TOTAL: 80},
        ),
    ],
)
@pytest.mark.parametrize("as_object", [False, True])
def test_gemini_optional_token_counts(
    usage: dict[str, Any], expected: dict[str, int], as_object: bool
) -> None:
    assert dict(_get_token_counts(SimpleNamespace(**usage) if as_object else usage)) == expected


@pytest.mark.parametrize(
    "raw, expected_completion",
    [
        ({}, 80),
        ({"usage_metadata": {}}, 80),
        ({"usage_metadata": {"candidates_token_count": None}}, 80),
        ({"usage_metadata": {"candidates_token_count": "invalid"}}, 80),
        ({"usage_metadata": {"candidates_token_count": 0}}, 0),
        ({"usage": {"completion_tokens": 90}}, 90),
        ({"usage": {"output_tokens": 90}}, 90),
    ],
)
def test_additional_kwargs_token_counts_are_a_fallback(
    raw: Any,
    expected_completion: int,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    span = _Span(otel_span=tracer_provider.get_tracer(__name__).start_span("llm"), span_kind="LLM")
    span.process_event(
        LLMChatEndEvent(
            messages=[],
            response=ChatResponse(
                message=ChatMessage(content="Paris."),
                raw=raw,
                additional_kwargs={"prompt_tokens": 1200, "completion_tokens": 80},
            ),
        )
    )
    span.end()
    attributes = in_memory_span_exporter.get_finished_spans()[0].attributes or {}
    assert attributes[LLM_TOKEN_COUNT_PROMPT] == 1200
    assert attributes[LLM_TOKEN_COUNT_COMPLETION] == expected_completion


def test_google_genai_real_response_token_counts(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=1200,
        candidates_token_count=80,
        thoughts_token_count=900,
        cached_content_token_count=1024,
        tool_use_prompt_token_count=30,
        total_token_count=2210,
    )
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[types.Part(text="Paris.")]),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=usage,
        model_version="gemini-2.5-flash",
    )
    with (
        patch.object(
            models.Models,
            "get",
            return_value=types.Model(
                name="models/gemini-2.5-flash",
                output_token_limit=65536,
                input_token_limit=1048576,
            ),
        ),
        patch.object(chats.Chat, "send_message", return_value=response),
    ):
        llm = GoogleGenAI(model="gemini-2.5-flash", api_key="fake-key-not-used")
        result = llm.chat([ChatMessage(role="user", content="Capital of France?")])

    assert result.raw["usage_metadata"]["thoughts_token_count"] == 900
    assert result.additional_kwargs["completion_tokens"] == 80
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = spans[0].attributes or {}
    assert attributes[LLM_TOKEN_COUNT_PROMPT] == 1230
    assert attributes[LLM_TOKEN_COUNT_COMPLETION] == 980
    assert attributes[LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] == 900
    assert attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 1024
    assert attributes[LLM_TOKEN_COUNT_TOTAL] == 2210


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LlamaIndexInstrumentor().uninstrument()
