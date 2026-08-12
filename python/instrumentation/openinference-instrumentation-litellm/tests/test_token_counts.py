import json
import os
from types import SimpleNamespace
from typing import Generator, Iterator
from unittest.mock import MagicMock, patch

import litellm
import pytest
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation.litellm import (
    LiteLLMInstrumentor,
    _set_token_counts_from_usage,
)


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
) -> Iterator[None]:
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    yield


@pytest.fixture
def patch_tiktoken_encoding() -> Generator[None, None, None]:
    """Patch `tiktoken.get_encoding` for LiteLLM to avoid network calls."""

    with patch("tiktoken.get_encoding") as mock_get_encoding:
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3]
        mock_get_encoding.return_value = mock_encoding
        yield


@pytest.mark.usefixtures("patch_tiktoken_encoding")
class TestTokenCounts:
    @pytest.mark.vcr
    def test_openai(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        messages = [{"role": "user", "content": "Hello!"}]
        resp = litellm.completion(
            model="openai/gpt-4o-mini",
            messages=messages,
            api_key=os.getenv("OPENAI_API_KEY", "sk-"),
            temperature=0.7,
        )
        usage = resp.usage

        span = in_memory_span_exporter.get_finished_spans()[0]
        attr = dict(span.attributes or {})
        # make sure we are not leaking any sensitive information
        params_str = attr.get(LLM_INVOCATION_PARAMETERS)
        if params_str is not None:
            params = json.loads(str(params_str))
            assert isinstance(params, dict)
            assert "api_key" not in params

        assert attr.pop(LLM_TOKEN_COUNT_COMPLETION) == usage.completion_tokens
        assert attr.pop(LLM_TOKEN_COUNT_PROMPT) == usage.prompt_tokens
        assert attr.pop(LLM_TOKEN_COUNT_TOTAL) == usage.total_tokens

        # Check for detailed token stats if available in the response
        if hasattr(usage, "prompt_tokens_details"):
            assert (
                attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ)
                == usage.prompt_tokens_details.cached_tokens
            )
            assert (
                attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO)
                == usage.prompt_tokens_details.audio_tokens
            )

        if hasattr(usage, "completion_tokens_details"):
            assert (
                attr.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO)
                == usage.completion_tokens_details.audio_tokens
            )
            assert (
                attr.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING)
                == usage.completion_tokens_details.reasoning_tokens
            )

    @pytest.mark.vcr
    def test_anthropic(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        messages = [{"role": "user", "content": "Hello!"}]
        resp = litellm.completion(
            model="anthropic/claude-3-5-haiku-20241022",
            messages=messages,
            api_key=os.getenv("ANTHROPIC_API_KEY", "sk-"),
        )
        span = in_memory_span_exporter.get_finished_spans()[0]
        attr = dict(span.attributes or {})

        # make sure we are not leaking any sensitive information
        params_str = attr.get(LLM_INVOCATION_PARAMETERS)
        if params_str is not None:
            params = json.loads(str(params_str))
            assert isinstance(params, dict)
            assert "api_key" not in params

        usage = resp.usage
        # Check the token counts litellm always returns
        assert attr.pop(LLM_TOKEN_COUNT_PROMPT) == usage.prompt_tokens
        assert attr.pop(LLM_TOKEN_COUNT_COMPLETION) == usage.completion_tokens
        assert attr.pop(LLM_TOKEN_COUNT_TOTAL) == usage.total_tokens

        # Check additional token counts if present
        if hasattr(usage, "cache_creation_input_tokens"):
            assert (
                attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE)
                == usage.cache_creation_input_tokens
            )

        if hasattr(usage, "cache_read_input_tokens"):
            assert (
                attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == usage.cache_read_input_tokens
            )


def test_text_tokens_are_not_reported_as_cost_or_cache() -> None:
    """Regression test for a token-count vs. USD-cost mismap.

    ``completion_tokens_details.text_tokens`` and ``prompt_tokens_details.text_tokens``
    are raw token *counts*. They must never be written to a cost attribute
    (``llm.cost.*``, defined by the semantic conventions as USD amounts) nor to the
    cache-input token-count key (which represents cached tokens, not text tokens).
    """
    exporter = InMemorySpanExporter()
    provider = SDKTracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        prompt_tokens_details=SimpleNamespace(
            cached_tokens=10,
            audio_tokens=0,
            text_tokens=90,
        ),
        completion_tokens_details=SimpleNamespace(
            reasoning_tokens=5,
            audio_tokens=0,
            text_tokens=45,
        ),
    )
    result = SimpleNamespace(usage=usage)

    with tracer.start_as_current_span("test") as span:
        _set_token_counts_from_usage(span, result)

    attributes = dict(exporter.get_finished_spans()[0].attributes or {})

    # A token count must not be emitted as a USD cost.
    assert LLM_COST_COMPLETION_DETAILS_OUTPUT not in attributes
    # text_tokens is not a cached-input count; cached_tokens covers that key.
    assert LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_INPUT not in attributes

    # The correctly mapped token-detail counts still land on their proper keys.
    assert attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 10
    assert attributes[LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] == 5


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
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_INPUT = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_INPUT
)
LLM_COST_COMPLETION_DETAILS_OUTPUT = SpanAttributes.LLM_COST_COMPLETION_DETAILS_OUTPUT
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
