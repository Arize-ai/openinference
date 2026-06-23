import types

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agno._workflow_wrapper import (
    _ParallelWrapper,
    _StepWrapper,
    _WorkflowWrapper,
)


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


def _wrapped(message: str) -> str:
    return "ok"


@pytest.mark.parametrize(
    "wrapper_cls, method, default_prefix",
    [
        (_WorkflowWrapper, "run", "Workflow"),
        (_StepWrapper, "run", "Step"),
        (_ParallelWrapper, "execute", "Parallel"),
    ],
)
def test_span_name_falls_back_when_name_is_none(
    wrapper_cls: type,
    method: str,
    default_prefix: str,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """agno's Workflow/Step/Parallel declare ``name: Optional[str] = None``, so an
    unnamed instance has ``name is None`` (the attribute exists). The span-name
    construction used ``getattr(instance, "name", <default>).replace(...)``, which
    only substitutes the default when the attribute is *missing* — on a ``None``
    name it raised ``AttributeError: 'NoneType' object has no attribute 'replace'``
    and aborted the user's run before it started. It must fall back to the default
    span-name prefix instead.
    """
    instance = types.SimpleNamespace(name=None, description=None, steps=[], team=None, agent=None)
    wrapper = wrapper_cls(tracer=tracer_provider.get_tracer(__name__))

    # Pre-fix this raised AttributeError before `_wrapped` ever ran.
    result = getattr(wrapper, method)(_wrapped, instance, ("hello",), {})
    assert result == "ok"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name.startswith(f"{default_prefix}.")
