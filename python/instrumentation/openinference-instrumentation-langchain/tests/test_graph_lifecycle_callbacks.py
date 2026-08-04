"""Regression tests for LangGraph graph-lifecycle callbacks on the tracer.

LangGraph dispatches its graph-level lifecycle callbacks (``on_interrupt`` and
``on_resume``) to every registered callback handler. The LangChain instrumentor
injects ``OpenInferenceTracer`` into every callback manager, so the tracer
receives these events even though they are not part of the ``BaseTracer``
interface. Before the no-op handlers existed, ``langchain_core``'s
``handle_event`` called ``getattr(handler, "on_interrupt")``, raised
``AttributeError``, and logged a warning on every interrupt/resume.

These tests reproduce the dispatch path through ``langchain_core`` directly so
they hold regardless of the installed LangGraph version.
"""

import logging
from typing import Any

import pytest
from langchain_core.callbacks.manager import handle_event
from opentelemetry.trace import NoOpTracer

from openinference.instrumentation.langchain._tracer import OpenInferenceTracer

_MANAGER_LOGGER = "langchain_core.callbacks.manager"


def _build_tracer() -> OpenInferenceTracer:
    return OpenInferenceTracer(
        tracer=NoOpTracer(),
        separate_trace_from_runtime_context=False,
    )


def test_graph_lifecycle_callbacks_do_not_raise() -> None:
    tracer = _build_tracer()
    event: Any = object()
    # The tracer has no span work to do for these events; it only needs to
    # accept them without raising.
    tracer.on_interrupt(event)
    tracer.on_resume(event)


def test_dispatching_graph_lifecycle_events_logs_no_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tracer = _build_tracer()
    event: Any = object()
    with caplog.at_level(logging.WARNING, logger=_MANAGER_LOGGER):
        handle_event([tracer], "on_interrupt", None, event)
        handle_event([tracer], "on_resume", None, event)
    unexpected = [record.getMessage() for record in caplog.records]
    assert not unexpected, f"unexpected warnings logged: {unexpected}"
