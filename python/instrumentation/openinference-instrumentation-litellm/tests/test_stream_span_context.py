"""Streaming calls run the wrapped litellm function with the litellm span current.

LiteLLM delegates to provider SDKs (the OpenAI client, for example) inside the
wrapped call. If the litellm span is not current at that point, spans created
by other instrumentors for that provider call start a new trace instead of
nesting under the litellm span. These tests stand in for such an instrumentor:
the wrapped function opens a span of its own and records what was current.
"""

import asyncio
from typing import Any, Callable, Dict, Generator, Iterator

import litellm
import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.litellm import LiteLLMInstrumentor

PROVIDER_SPAN = "provider.call"


@pytest.fixture()
def setup_litellm_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LiteLLMInstrumentor().uninstrument()


@pytest.fixture()
def provider_span_inside(
    tracer_provider: TracerProvider,
) -> Iterator[Callable[[str, Callable[..., Any]], None]]:
    """Wrap an original litellm function so it opens a provider span while it runs."""
    tracer = tracer_provider.get_tracer("downstream-instrumentor")
    saved: Dict[str, Any] = {}

    def wrap(name: str, inner: Callable[..., Any]) -> None:
        saved[name] = LiteLLMInstrumentor.original_litellm_funcs[name]
        if asyncio.iscoroutinefunction(inner):

            async def traced_async(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(PROVIDER_SPAN):
                    return await inner(*args, **kwargs)

            LiteLLMInstrumentor.original_litellm_funcs[name] = traced_async
        else:

            def traced(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(PROVIDER_SPAN):
                    return inner(*args, **kwargs)

            LiteLLMInstrumentor.original_litellm_funcs[name] = traced

    yield wrap
    LiteLLMInstrumentor.original_litellm_funcs.update(saved)


def _assert_nested(exporter: InMemorySpanExporter, litellm_span_name: str) -> None:
    spans = {span.name: span for span in exporter.get_finished_spans()}
    outer: ReadableSpan = spans[litellm_span_name]
    inner: ReadableSpan = spans[PROVIDER_SPAN]
    assert inner.parent is not None
    assert inner.parent.span_id == outer.context.span_id
    assert inner.context.trace_id == outer.context.trace_id


def test_completion_stream_runs_under_the_litellm_span(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    provider_span_inside: Callable[[str, Callable[..., Any]], None],
) -> None:
    in_memory_span_exporter.clear()
    provider_span_inside("completion", LiteLLMInstrumentor.original_litellm_funcs["completion"])

    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"content": "What's the capital of China?", "role": "user"}],
        mock_response="The capital of China is Beijing",
        stream=True,
    )
    for _ in response:
        pass

    _assert_nested(in_memory_span_exporter, "completion")


def test_acompletion_stream_runs_under_the_litellm_span(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    provider_span_inside: Callable[[str, Callable[..., Any]], None],
) -> None:
    in_memory_span_exporter.clear()
    provider_span_inside("acompletion", LiteLLMInstrumentor.original_litellm_funcs["acompletion"])

    async def run() -> None:
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
            mock_response="The capital of China is Beijing",
            stream=True,
        )
        async for _ in response:
            pass

    asyncio.run(run())

    _assert_nested(in_memory_span_exporter, "acompletion")


def test_responses_stream_runs_under_the_litellm_span(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    provider_span_inside: Callable[[str, Callable[..., Any]], None],
) -> None:
    in_memory_span_exporter.clear()
    # Not a streaming iterator, so the wrapper ends its span right away; what
    # matters here is what was current while the wrapped function ran.
    provider_span_inside("responses", lambda *args, **kwargs: object())

    litellm.responses(model="gpt-4o-mini", input="hi", stream=True)

    _assert_nested(in_memory_span_exporter, "responses")


def test_aresponses_stream_runs_under_the_litellm_span(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    provider_span_inside: Callable[[str, Callable[..., Any]], None],
) -> None:
    in_memory_span_exporter.clear()

    async def not_a_stream(*args: Any, **kwargs: Any) -> Any:
        return object()

    provider_span_inside("aresponses", not_a_stream)

    asyncio.run(litellm.aresponses(model="gpt-4o-mini", input="hi", stream=True))

    _assert_nested(in_memory_span_exporter, "aresponses")


def test_stream_span_is_not_left_current_after_the_call(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"content": "hi", "role": "user"}],
        mock_response="hello",
        stream=True,
    )
    # The span stays open until the stream is consumed, but it must not leak
    # into the caller's context.
    assert not trace_api.get_current_span().get_span_context().is_valid
    for _ in response:
        pass
