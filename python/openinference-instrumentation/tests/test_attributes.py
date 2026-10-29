import json
from typing import Any, Dict, cast

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    get_input_attributes,
    get_tool_attributes,
)
from openinference.semconv.trace import SpanAttributes


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({("a", "b"): 1}, id="tuple-key"),
        pytest.param({1: "int", 2: "int"}, id="int-keys"),
        pytest.param({"outer": {("x", "y"): ["nested"]}}, id="nested-tuple-key"),
    ],
)
def test_input_attributes_serialize_dicts_with_non_string_keys(value: Dict[Any, Any]) -> None:
    """`input.value` is advertised as JSON, so it must parse back as JSON."""
    attributes = get_input_attributes(value)

    assert attributes[SpanAttributes.INPUT_MIME_TYPE] == "application/json"
    assert isinstance(json.loads(cast(str, attributes[SpanAttributes.INPUT_VALUE])), dict)


def test_tool_parameters_with_non_string_keys_is_json() -> None:
    attributes = get_tool_attributes(name="f", parameters={("a", "b"): 1})

    assert json.loads(cast(str, attributes[SpanAttributes.TOOL_PARAMETERS])) == {"('a', 'b')": 1}


def test_tool_decorator_runs_the_wrapped_function_with_non_string_keys() -> None:
    """Instrumentation must never raise into user code."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(provider.get_tracer(__name__), config=TraceConfig())

    @tracer.tool(name="f", parameters={("a", "b"): 1})
    def f() -> str:
        return "ok"

    assert f() == "ok"
    (span,) = exporter.get_finished_spans()
    assert span.attributes is not None
    assert json.loads(cast(str, span.attributes[SpanAttributes.TOOL_PARAMETERS])) == {
        "('a', 'b')": 1
    }
