"""The ``invocation`` and ``agent_run`` spans are held open by an async generator.

ADK does not always exhaust the generators it iterates: an ``LlmAgent`` stops on
``transfer_to_agent``, and callers often ``break`` out of ``Runner.run_async`` on the
final response. The generator is then closed either explicitly through ``aclose()``
or by asyncio's async-generator finalizer, which runs ``aclose()`` in a fresh task
whose ``contextvars.Context`` is a copy. Resetting the OpenTelemetry context there
fails and ``opentelemetry.context`` logs ``Failed to detach context`` at ERROR level
on every early exit. See https://github.com/Arize-ai/openinference/issues/1769.
"""

import asyncio
import logging
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Callable, cast

import pytest
from google.adk.events import Event, EventActions
from google.genai import types
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation.google_adk._wrappers import (
    _BaseAgentRunAsync,
    _RunnerRunAsync,
)
from openinference.semconv.trace import SpanAttributes

_OTEL_CONTEXT_LOGGER = "opentelemetry.context"


def _event(text: str) -> Event:
    return Event(
        author="test-agent",
        invocation_id="test-invocation",
        actions=EventActions(),
        content=types.Content(role="model", parts=[types.Part(text=text)]),
    )


class _Stream:
    """A fake ``run_async`` that records whether ADK's generator was closed."""

    def __init__(self, count: int = 3) -> None:
        self.count = count
        self.closed = False

    async def __call__(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Event, None]:
        try:
            for index in range(self.count):
                yield _event(f"event {index}")
        finally:
            self.closed = True


def _instrument_agent(tracer: trace_api.Tracer, stream: _Stream) -> Any:
    return _BaseAgentRunAsync(tracer)(stream, cast(Any, SimpleNamespace(name="test-agent")), (), {})


def _instrument_runner(tracer: trace_api.Tracer, stream: _Stream) -> Any:
    return _RunnerRunAsync(tracer)(
        stream,
        cast(Any, SimpleNamespace(app_name="test-app")),
        (),
        {
            "user_id": "test-user",
            "session_id": "test-session",
            "new_message": types.Content(role="user", parts=[types.Part(text="hi")]),
        },
    )


_INSTRUMENT: dict[str, Callable[[trace_api.Tracer, _Stream], Any]] = {
    "agent_run [test-agent]": _instrument_agent,
    "invocation [test-app]": _instrument_runner,
}


def _otel_context_errors(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == _OTEL_CONTEXT_LOGGER and record.levelno >= logging.ERROR
    ]


@pytest.mark.parametrize("span_name", list(_INSTRUMENT))
async def test_span_ends_quietly_when_generator_is_closed_in_another_task(
    span_name: str,
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _Stream()
    generator = _INSTRUMENT[span_name](tracer_provider.get_tracer(__name__), stream)
    iterator = generator.__aiter__()
    assert (await iterator.__anext__()).content.parts[0].text == "event 0"
    assert in_memory_span_exporter.get_finished_spans() == ()

    # A new task runs in a copy of the current context, exactly like the event
    # loop's async-generator finalizer when the generator is dropped unclosed.
    with caplog.at_level(logging.ERROR, logger=_OTEL_CONTEXT_LOGGER):
        await asyncio.create_task(iterator.aclose())

    assert _otel_context_errors(caplog) == []
    [span] = in_memory_span_exporter.get_finished_spans()
    assert span.name == span_name
    assert span.status.status_code is StatusCode.UNSET, "GeneratorExit is not an error"
    assert not span.events


@pytest.mark.parametrize("span_name", list(_INSTRUMENT))
async def test_aclose_ends_span_in_the_calling_task_before_closing_adk_generator(
    span_name: str,
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _Stream()
    generator = _INSTRUMENT[span_name](tracer_provider.get_tracer(__name__), stream)

    with caplog.at_level(logging.ERROR, logger=_OTEL_CONTEXT_LOGGER):
        async for event in generator:
            assert event.content is not None
            break  # early exit, like ADK on transfer_to_agent
        await generator.aclose()  # what ADK's ``Aclosing`` does
        assert stream.closed

    assert _otel_context_errors(caplog) == []
    [span] = in_memory_span_exporter.get_finished_spans()
    assert span.name == span_name
    assert span.status.status_code is StatusCode.UNSET
    assert trace_api.get_current_span() is trace_api.INVALID_SPAN, "context was restored"


class _BlockingStream(_Stream):
    """Yields one event, then waits until the consuming task is cancelled."""

    async def __call__(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Event, None]:
        try:
            yield _event("event 0")
            await asyncio.Event().wait()
        finally:
            self.closed = True


@pytest.mark.parametrize("span_name", list(_INSTRUMENT))
async def test_cancelled_run_ends_span_unset_like_use_span(
    span_name: str,
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stream = _BlockingStream()
    generator = _INSTRUMENT[span_name](tracer_provider.get_tracer(__name__), stream)
    first_event = asyncio.Event()

    async def consume() -> None:
        async for _ in generator:
            first_event.set()

    with caplog.at_level(logging.ERROR, logger=_OTEL_CONTEXT_LOGGER):
        task = asyncio.create_task(consume())
        await first_event.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert stream.closed
    assert _otel_context_errors(caplog) == []
    [span] = in_memory_span_exporter.get_finished_spans()
    assert span.name == span_name
    # CancelledError is a BaseException: like ``use_span``, it is neither recorded
    # nor turned into an ERROR status.
    assert span.status.status_code is StatusCode.UNSET
    assert not span.events


async def test_aclose_before_iteration_only_closes_adk_generator(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    stream = _Stream()
    generator = _instrument_agent(tracer_provider.get_tracer(__name__), stream)

    await generator.aclose()

    with pytest.raises(StopAsyncIteration):
        await generator.__wrapped__.__anext__()
    assert in_memory_span_exporter.get_finished_spans() == ()


async def test_exhausted_generator_records_output_and_ok_status(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    generator = _instrument_runner(tracer_provider.get_tracer(__name__), _Stream(count=2))
    seen = []
    async for event in generator:
        seen.append(event)
        assert trace_api.get_current_span().name == "invocation [test-app]"  # type: ignore[attr-defined]
        assert get_current_user_id() == "test-user"
        assert get_current_session_id() == "test-session"

    assert len(seen) == 2
    [span] = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.OK
    assert span.attributes is not None
    assert span.attributes[SpanAttributes.USER_ID] == "test-user"
    assert span.attributes[SpanAttributes.SESSION_ID] == "test-session"
    assert SpanAttributes.OUTPUT_VALUE in span.attributes


async def test_failing_generator_records_exception_and_error_status(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async def failing_run_async(*args: Any, **kwargs: Any) -> AsyncGenerator[Event, None]:
        yield _event("first")
        raise RuntimeError("boom")

    generator = _BaseAgentRunAsync(tracer_provider.get_tracer(__name__))(
        failing_run_async, cast(Any, SimpleNamespace(name="test-agent")), (), {}
    )
    with pytest.raises(RuntimeError, match="boom"):
        async for _ in generator:
            pass

    [span] = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "RuntimeError: boom"
    assert [event.name for event in span.events] == ["exception"]


def get_current_user_id() -> Any:
    from opentelemetry import context as context_api

    return context_api.get_value(SpanAttributes.USER_ID)


def get_current_session_id() -> Any:
    from opentelemetry import context as context_api

    return context_api.get_value(SpanAttributes.SESSION_ID)
