"""Regression coverage for non-string mapping keys in core JSON serialization."""

import json
from enum import Enum
from typing import Any, Dict, List

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import OITracer
from openinference.instrumentation._attributes import _json_serialize
from openinference.semconv.trace import SpanAttributes


class KeyEnum(Enum):
    FOO = "foo"


def test_json_serialize_stringifies_unsupported_keys_recursively() -> None:
    parameters: Dict[Any, Any] = {
        ("a", "b"): 1,
        b"raw": 2,
        KeyEnum.FOO: 3,
        "nested": {
            ("x", "y"): [{(b"deep",): "value", "keep": "café"}],
        },
    }

    serialized = _json_serialize(parameters)

    assert json.loads(serialized) == {
        "('a', 'b')": 1,
        "b'raw'": 2,
        str(KeyEnum.FOO): 3,
        "nested": {
            "('x', 'y')": [{"(b'deep',)": "value", "keep": "café"}],
        },
    }
    assert "café" in serialized  # ensure_ascii=False is preserved


def test_json_serialize_preserves_json_native_keys() -> None:
    serialized = _json_serialize(
        {3: "int", 2.5: "float", True: "bool", None: "none", "str": "str"},
    )

    assert json.loads(serialized) == {
        "3": "int",
        "2.5": "float",
        "true": "bool",
        "null": "none",
        "str": "str",
    }


def test_tracer_tool_runs_with_non_string_parameter_keys(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer: OITracer,
) -> None:
    calls: List[int] = []

    @tracer.tool(parameters={("a", "b"): 1})
    def decorated_tool(value: int) -> int:
        calls.append(value)
        return value * 2

    assert decorated_tool(3) == 6
    assert calls == [3]

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    tool_parameters = attributes[SpanAttributes.TOOL_PARAMETERS]
    assert isinstance(tool_parameters, str)
    assert json.loads(tool_parameters) == {"('a', 'b')": 1}
