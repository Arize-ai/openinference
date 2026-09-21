import os
import random
import string
from importlib import import_module
from importlib.metadata import version
from typing import Dict, Tuple, cast

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import OpenInferenceLLMProviderValues, SpanAttributes

LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)

# OpenAI only caches prompt prefixes above ~1024 tokens, so the shared prefix has
# to be comfortably larger than that for the cache write / cache read to happen.
_CACHEABLE_PREFIX_CHARS = 8000


def _cacheable_prefix() -> str:
    """A prefix long enough to be cached, randomised so recording starts cold.

    Randomising matters only while recording: it guarantees the first request is a
    genuine cache miss (a cache write) rather than a hit left over from a previous
    run. On replay the value is irrelevant because VCR matches on method and URI.
    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=_CACHEABLE_PREFIX_CHARS))


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.6-terra"])
@pytest.mark.vcr
def test_chat_completions_cache_write_then_read(
    model: str,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
) -> None:
    """Real Chat Completions traffic: the same prefix is written to, then read from, cache."""
    if _openai_version() < (1, 12, 0):
        pytest.skip("Not supported")
    openai = import_module("openai")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-"))
    prefix = _cacheable_prefix()
    for question in ("Write me a haiku.", "Write me a sonnet."):
        # The cacheable prefix is a standalone system message: these models reuse the
        # cache across whole leading messages, so appending the varying question to
        # the same message as the prefix would miss the cache entirely.
        client.chat.completions.create(
            extra_headers={"Accept-Encoding": "gzip"},
            model=model,
            messages=[
                {"role": "system", "content": prefix},
                {"role": "user", "content": question},
            ],
        )

    _assert_cold_write_then_warm_read(in_memory_span_exporter, prefix)


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "gpt-5.6-terra"])
@pytest.mark.vcr
def test_responses_cache_write_then_read(
    model: str,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
) -> None:
    """Real Responses API traffic: the same prefix is written to, then read from, cache."""
    if _openai_version() < (1, 66, 0):
        pytest.skip("Responses API not supported")
    openai = import_module("openai")

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-"))
    prefix = _cacheable_prefix()
    for question in ("Write me a haiku.", "Write me a sonnet."):
        client.responses.create(
            extra_headers={"Accept-Encoding": "gzip"},
            model=model,
            instructions=prefix,
            input=question,
        )

    _assert_cold_write_then_warm_read(in_memory_span_exporter, prefix)


def _assert_cold_write_then_warm_read(
    in_memory_span_exporter: InMemorySpanExporter, prefix: str
) -> None:
    """Assert a cold cache write span followed by a warm cache read span of the same prefix."""
    first, second = _cache_token_counts(in_memory_span_exporter, prefix)

    # First call is a cache miss: nothing read, (almost) the whole prompt written.
    assert first[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 0
    assert first[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] > 1024

    # Second call reuses the prefix: it is read from cache instead of written again.
    assert second[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] > 1024
    assert (
        second[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]
        < first[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]
    )

    # Cache reads and writes are subsets of the prompt; some input may be neither.
    for usage in (first, second):
        cache_total = (
            usage[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]
            + usage[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]
        )
        assert cache_total <= usage[LLM_TOKEN_COUNT_PROMPT]


def _cache_token_counts(
    in_memory_span_exporter: InMemorySpanExporter, prefix: str
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return the token counts of the two OpenAI LLM spans that sent ``prefix``, in call order."""
    # Match on the prefix so a span left over from an earlier test cannot be counted.
    spans = tuple(
        span
        for span in get_openai_llm_spans(in_memory_span_exporter.get_finished_spans())
        if prefix in str((span.attributes or {}).get(SpanAttributes.INPUT_VALUE, ""))
    )
    assert len(spans) == 2
    usages = []
    for span in spans:
        usage = {}
        for key in (
            LLM_TOKEN_COUNT_PROMPT,
            LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
            LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
        ):
            value = (span.attributes or {}).get(key)
            assert isinstance(value, int), f"{key} missing or not an int on span {span.name}"
            usage[key] = value
        usages.append(usage)
    return usages[0], usages[1]


def _openai_version() -> Tuple[int, int, int]:
    return cast(Tuple[int, int, int], tuple(map(int, version("openai").split(".")[:3])))


def get_openai_llm_spans(spans: Tuple[ReadableSpan, ...]) -> Tuple[ReadableSpan, ...]:
    """Filter spans to only the primary OpenAI LLM response spans to avoid extra internal spans."""
    llm_spans = [
        span
        for span in spans
        if span.attributes
        and span.attributes.get(SpanAttributes.LLM_PROVIDER)
        == OpenInferenceLLMProviderValues.OPENAI.value
    ]
    if not llm_spans:
        raise ValueError("No OpenAI LLM spans found in spans.")
    return tuple(llm_spans)
