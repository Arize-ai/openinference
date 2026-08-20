"""Instrumentation must be transparent to callers of the instrumented functions.

Covers three failure modes observed when litellm's own Responses API bridge
re-enters the instrumented ``litellm.acompletion``:

1. Streaming results must keep their type (``CustomStreamWrapper``), because
   callers may ``isinstance``-check them — the bridge raises
   ``Unexpected response type: <class 'async_generator'>`` otherwise.
2. ``aresponses`` streaming results of foreign iterator types must pass
   through untouched instead of being drained into an empty stream.
3. Recording span attributes must never raise into the traced call
   (e.g. ``ValueError: Circular reference detected`` on self-referencing
   kwargs seen on litellm router retries).
"""

import asyncio
from typing import Any, Dict, Generator
from unittest.mock import MagicMock

import litellm
import pytest
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.litellm import (
    LiteLLMInstrumentor,
    _instrument_func_type_completion,
    _instrument_func_type_responses,
)


@pytest.fixture()
def setup_litellm_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LiteLLMInstrumentor().uninstrument()


def test_sync_streaming_preserves_stream_type(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"content": "What's the capital of China?", "role": "user"}],
        mock_response="The capital of China is Beijing",
        stream=True,
    )

    assert isinstance(response, CustomStreamWrapper)

    output_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            output_message += chunk.choices[0].delta.content
    assert output_message == "The capital of China is Beijing"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "completion"


def test_async_streaming_preserves_stream_type(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    async def run() -> str:
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
            mock_response="The capital of China is Beijing",
            stream=True,
        )
        assert isinstance(response, CustomStreamWrapper)
        output_message = ""
        async for chunk in response:
            if chunk.choices[0].delta.content:
                output_message += chunk.choices[0].delta.content
        return output_message

    assert asyncio.run(run()) == "The capital of China is Beijing"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "acompletion"


def test_aresponses_foreign_stream_type_passes_through(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """A stream type the finalizer doesn't understand must not be drained empty."""
    in_memory_span_exporter.clear()

    class ForeignAsyncIterator:
        def __init__(self) -> None:
            self._tokens = ["a", "b"]

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            if not self._tokens:
                raise StopAsyncIteration
            return self._tokens.pop(0)

    foreign = ForeignAsyncIterator()
    original_func = LiteLLMInstrumentor.original_litellm_funcs["aresponses"]
    try:

        async def fake_aresponses(*args: Any, **kwargs: Any) -> Any:
            return foreign

        LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = fake_aresponses

        async def run() -> Any:
            return await litellm.aresponses(
                model="gpt-4o-mini",
                input="Hi",
                stream=True,
            )

        result = asyncio.run(run())
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = original_func

    assert result is foreign

    async def drain() -> list[Any]:
        return [token async for token in result]

    assert asyncio.run(drain()) == ["a", "b"]

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "aresponses"


def test_attribute_extraction_errors_do_not_raise() -> None:
    cyclic: Dict[str, Any] = {"model": "gpt-4o"}
    cyclic["self"] = cyclic
    cyclic["messages"] = [{"role": "user", "content": "hi"}]

    span = MagicMock()
    assert _instrument_func_type_completion(span, cyclic) is None
    assert _instrument_func_type_responses(span, cyclic) is None
