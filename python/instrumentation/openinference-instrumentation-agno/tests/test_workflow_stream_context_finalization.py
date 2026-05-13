import asyncio
import logging
import threading
from typing import Any, AsyncIterator, Iterator

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.agno._workflow_wrapper import (
    _ParallelWrapper,
    _StepWrapper,
    _WorkflowWrapper,
)
from openinference.instrumentation.agno.utils import _AGNO_PARENT_NODE_CONTEXT_KEY


class _Response:
    def __init__(self, content: str) -> None:
        self.content = content


class _StepInstance:
    name = "test step"
    agent = None
    team = None


class _WorkflowInstance:
    name = "test workflow"
    description = "workflow used for stream context finalization"
    steps = [_StepInstance()]
    id = "workflow-123"
    user_id = "user-123"


class _ParallelInstance:
    name = "test parallel"
    steps = [_StepInstance()]


def _sync_stream() -> Iterator[_Response]:
    yield _Response("one")
    yield _Response("two")


async def _async_stream() -> AsyncIterator[_Response]:
    yield _Response("one")
    await asyncio.sleep(0)
    yield _Response("two")


def _build_wrapper(wrapper_cls: type[Any]) -> tuple[Any, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(
        trace_api.get_tracer(
            "test-agno-workflow-stream-context",
            tracer_provider=tracer_provider,
        ),
        config=TraceConfig(),
    )
    return wrapper_cls(tracer=tracer), exporter


STREAM_WRAPPER_CASES = (
    pytest.param(_WorkflowWrapper, "run", _WorkflowInstance(), id="workflow"),
    pytest.param(_StepWrapper, "run", _StepInstance(), id="step"),
    pytest.param(_ParallelWrapper, "execute", _ParallelInstance(), id="parallel"),
)

ASYNC_STREAM_WRAPPER_CASES = (
    pytest.param(_WorkflowWrapper, "arun", _WorkflowInstance(), id="workflow"),
    pytest.param(_StepWrapper, "arun", _StepInstance(), id="step"),
    pytest.param(_ParallelWrapper, "aexecute", _ParallelInstance(), id="parallel"),
)


@pytest.mark.parametrize(("wrapper_cls", "method_name", "instance"), STREAM_WRAPPER_CASES)
def test_sync_stream_close_from_other_thread_does_not_log_detach_error(
    wrapper_cls: type[Any],
    method_name: str,
    instance: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    wrapper, exporter = _build_wrapper(wrapper_cls)
    errors: list[BaseException] = []
    restored_context: list[tuple[bool, bool]] = []

    def consume_and_close() -> None:
        try:
            initial_span = trace_api.get_current_span()
            initial_parent_node = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)
            stream = getattr(wrapper, method_name)(_sync_stream, instance, (), {})
            next(stream)
            restored_context.append(
                (
                    trace_api.get_current_span() is initial_span,
                    context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY) == initial_parent_node,
                )
            )

            def close_stream() -> None:
                try:
                    stream.close()
                except BaseException as exc:
                    errors.append(exc)

            close_thread = threading.Thread(target=close_stream)
            close_thread.start()
            close_thread.join()
            restored_context.append(
                (
                    trace_api.get_current_span() is initial_span,
                    context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY) == initial_parent_node,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    consume_thread = threading.Thread(target=consume_and_close)
    consume_thread.start()
    consume_thread.join()

    assert errors == []
    assert restored_context == [(True, True), (True, True)]
    assert len(exporter.get_finished_spans()) == 1
    assert "Failed to detach context" not in caplog.text


@pytest.mark.parametrize(("wrapper_cls", "method_name", "instance"), ASYNC_STREAM_WRAPPER_CASES)
def test_async_stream_aclose_from_other_task_does_not_log_detach_error(
    wrapper_cls: type[Any],
    method_name: str,
    instance: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    wrapper, exporter = _build_wrapper(wrapper_cls)
    restored_context: list[tuple[bool, bool]] = []

    async def run_test() -> None:
        async def consume_and_close() -> None:
            initial_span = trace_api.get_current_span()
            initial_parent_node = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)
            stream = getattr(wrapper, method_name)(_async_stream, instance, (), {})
            await anext(stream)
            restored_context.append(
                (
                    trace_api.get_current_span() is initial_span,
                    context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY) == initial_parent_node,
                )
            )

            async def close_stream() -> None:
                await stream.aclose()

            await asyncio.create_task(close_stream())
            restored_context.append(
                (
                    trace_api.get_current_span() is initial_span,
                    context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY) == initial_parent_node,
                )
            )

        await asyncio.create_task(consume_and_close())

    asyncio.run(run_test())

    assert restored_context == [(True, True), (True, True)]
    assert len(exporter.get_finished_spans()) == 1
    assert "Failed to detach context" not in caplog.text
