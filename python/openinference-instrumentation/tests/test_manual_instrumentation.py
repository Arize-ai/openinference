import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, TypedDict, Union

import jsonschema
import pydantic
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode, get_current_span
from pydantic import BaseModel
from typing_extensions import Annotated, TypeAlias

from openinference.instrumentation import (
    OITracer,
    suppress_tracing,
    using_session,
)
from openinference.instrumentation._tracers import _infer_tool_parameters
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


class TestStartAsCurrentSpanContextManager:
    def test_chain_with_plain_text_input_and_output(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with tracer.start_as_current_span(
            "span-name",
            openinference_span_kind="chain",
        ) as chain_span:
            chain_span.set_input("plain-text-input")
            chain_span.set_output("plain-text-output")
            chain_span.set_status(Status(StatusCode.OK))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "span-name"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "plain-text-input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "plain-text-output"
        assert not attributes

    def test_chain_with_json_input_and_output(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with tracer.start_as_current_span(
            "span-name",
            openinference_span_kind="chain",
        ) as chain_span:
            chain_span.set_input(
                {"input-key": "input-value"},
            )
            chain_span.set_output(
                json.dumps({"output-key": "output-value"}),
                mime_type="application/json",
            )
            chain_span.set_status(Status(StatusCode.OK))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "span-name"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert attributes.pop(INPUT_VALUE) == json.dumps({"input-key": "input-value"})
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE) == json.dumps({"output-key": "output-value"})
        assert not attributes

    def test_chain_with_pydantic_and_dataclass_input_and_output(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        class InputModel(BaseModel):
            string_input: str
            int_input: int

        @dataclass
        class OutputModel:
            string_output: str
            int_output: int

        with tracer.start_as_current_span(
            "span-name",
            openinference_span_kind="chain",
        ) as chain_span:
            chain_span.set_input(InputModel(string_input="input1", int_input=1))
            chain_span.set_output(OutputModel(string_output="output", int_output=2))
            chain_span.set_status(Status(StatusCode.OK))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "span-name"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"string_input": "input1", "int_input": 1}
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert json.loads(output_value) == {"string_output": "output", "int_output": 2}
        assert not attributes

    def test_agent(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with tracer.start_as_current_span(
            "agent-span-name",
            openinference_span_kind="agent",
        ) as agent_span:
            agent_span.set_input("input")
            agent_span.set_output("output")
            agent_span.set_status(Status(StatusCode.OK))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "agent-span-name"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == AGENT
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    def test_set_tool_attributes_on_non_tool_span_raises_exception(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with tracer.start_as_current_span(
            "chain-span-name",
            openinference_span_kind="chain",
        ) as chain_span:
            with pytest.raises(ValueError) as exc_info:
                chain_span.set_tool(
                    name="tool-name",
                    description="tool-description",
                    parameters={"type": "string"},
                )
            assert str(exc_info.value) == "Cannot set tool attributes on a non-tool span"

    def test_non_openinference_span(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with tracer.start_as_current_span("non-openinference-span") as non_openinference_span:
            non_openinference_span.set_attribute("custom.attribute", "value")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "non-openinference-span"
        attributes = dict(span.attributes or {})
        assert attributes.pop("custom.attribute") == "value"
        assert not attributes

    def test_cannot_set_input_on_non_openinference_span(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with tracer.start_as_current_span("non-openinference-span") as span:
            with pytest.raises(ValueError) as exc_info:
                span.set_input("input")
            assert str(exc_info.value) == "Cannot set input attributes on a non-OpenInference span"

    def test_cannot_set_output_on_non_openinference_span(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with tracer.start_as_current_span("non-openinference-span") as span:
            with pytest.raises(ValueError) as exc_info:
                span.set_output("output")
            assert str(exc_info.value) == "Cannot set output attributes on a non-OpenInference span"

    def test_tool(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with tracer.start_as_current_span(
            "tool-span-name",
            openinference_span_kind="tool",
        ) as tool_span:
            tool_span.set_input("input")
            tool_span.set_output("output")
            tool_span.set_tool(
                name="tool-name",
                description="tool-description",
                parameters={
                    "type": "string",
                },
            )
            tool_span.set_status(Status(StatusCode.OK))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "tool-span-name"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert attributes.pop(TOOL_NAME) == "tool-name"
        assert attributes.pop(TOOL_DESCRIPTION) == "tool-description"
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == {"type": "string"}
        assert not attributes

    def test_tool_with_string_parameters_and_no_description(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with tracer.start_as_current_span(
            "tool-span-name",
            openinference_span_kind="tool",
        ) as tool_span:
            tool_span.set_input("input")
            tool_span.set_output("output")
            tool_span.set_tool(
                name="tool-name",
                parameters=json.dumps({"type": "string"}),
            )
            tool_span.set_status(Status(StatusCode.OK))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "tool-span-name"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert attributes.pop(TOOL_NAME) == "tool-name"
        assert attributes.pop(TOOL_PARAMETERS) == json.dumps({"type": "string"})
        assert not attributes

    def test_unhandled_exception(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        error_message = "Something went wrong"
        with pytest.raises(ValueError, match=error_message):
            with tracer.start_as_current_span(
                "span-name",
                openinference_span_kind="chain",
            ):
                raise ValueError(error_message)

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "span-name"
        status = span.status
        assert not status.is_ok
        assert status.status_code == StatusCode.ERROR
        assert status.description == "ValueError: Something went wrong"
        events = list(span.events)
        assert len(events) == 1
        event = events[0]
        assert event.name == "exception"
        event_attributes = dict(event.attributes or {})
        assert event_attributes.pop("exception.type") == "ValueError"
        assert event_attributes.pop("exception.message") == "Something went wrong"
        assert event_attributes.pop("exception.stacktrace")
        assert event_attributes.pop("exception.escaped") == "False"
        assert not event_attributes
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert not attributes

    def test_suppress_tracing(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with pytest.raises(ValueError, match="Something went wrong"):
            with suppress_tracing():
                with tracer.start_as_current_span(
                    "span-name",
                    openinference_span_kind="chain",
                ):
                    raise ValueError("Something went wrong")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 0

    def test_no_traces_are_produced_when_otel_sdk_disabled(
        self,
        otel_sdk_disabled: pytest.MonkeyPatch,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with tracer.start_as_current_span(
            "span-name",
            openinference_span_kind="chain",
        ) as span:
            span.set_input("hello")
            span.set_output("world")
            span.set_status(Status(StatusCode.OK))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 0

    def test_using_session(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        with using_session("123"):
            with tracer.start_as_current_span(
                "chain-span-with-session",
                openinference_span_kind="chain",
            ) as chain_span:
                chain_span.set_input("input")
                chain_span.set_output("output")
                chain_span.set_status(Status(StatusCode.OK))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "chain-span-with-session"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes[SESSION_ID] == "123"


class TestTracerChainDecorator:
    def test_plain_text_input_and_output(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.chain
        def decorated_chain_with_plain_text_io(input: str) -> str:
            return "output"

        decorated_chain_with_plain_text_io("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_chain_with_plain_text_io"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    def test_json_output(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.chain
        def decorated_chain_with_json_output(input: str) -> Dict[str, Any]:
            return {"output": "output"}

        decorated_chain_with_json_output("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_chain_with_json_output"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE) == json.dumps({"output": "output"})
        assert not attributes

    async def test_async(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.chain
        async def decorated_async_chain(input: str) -> str:
            return "output"

        await decorated_async_chain("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_async_chain"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    def test_no_parameters(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.chain()  # apply decorator with no parameters
        def decorated_chain_with_empty_parens(input: str) -> Dict[str, Any]:
            return {"output": "output"}

        decorated_chain_with_empty_parens("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_chain_with_empty_parens"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE) == json.dumps({"output": "output"})
        assert not attributes

    def test_overridden_name(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.chain(name="overridden-name")
        def decorated_chain_with_overridden_name(input: str) -> Dict[str, Any]:
            return {"output": "output"}

        decorated_chain_with_overridden_name("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "overridden-name"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE) == json.dumps({"output": "output"})
        assert not attributes

    def test_pydantic_input_and_output(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        class InputModel(BaseModel):
            string_input: str
            int_input: int

        class OutputModel(BaseModel):
            string_output: str
            int_output: int

        @tracer.chain
        def decorated_chain_with_pydantic_io(input: InputModel) -> OutputModel:
            return OutputModel(string_output="output", int_output=42)

        decorated_chain_with_pydantic_io(InputModel(string_input="test", int_input=123))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_chain_with_pydantic_io"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert attributes.pop(INPUT_VALUE) == json.dumps({"string_input": "test", "int_input": 123})
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE) == json.dumps(
            {"string_output": "output", "int_output": 42}
        )
        assert not attributes

    def test_dataclass_input_and_output(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @dataclass
        class InputModel:
            string_input: str
            int_input: int

        @dataclass
        class OutputModel:
            string_output: str
            int_output: int

        @tracer.chain
        def decorated_chain_with_dataclass_io(input: InputModel) -> OutputModel:
            return OutputModel(string_output="output", int_output=42)

        decorated_chain_with_dataclass_io(InputModel(string_input="test", int_input=123))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_chain_with_dataclass_io"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert attributes.pop(INPUT_VALUE) == json.dumps({"string_input": "test", "int_input": 123})
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(OUTPUT_VALUE) == json.dumps(
            {"string_output": "output", "int_output": 42}
        )
        assert not attributes

    def test_multiple_inputs_and_nested_output(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        class NestedPydanticModel(BaseModel):
            value: int
            name: str

        class OutputPydanticModel(BaseModel):
            nested: NestedPydanticModel
            description: str

        @dataclass
        class NestedDataclass:
            count: int
            active: bool

        @dataclass
        class ComplexOutput:
            pydantic_part: OutputPydanticModel
            dataclass_part: NestedDataclass
            string_part: str

        @tracer.chain
        def decorated_chain_complex_io(
            model: BaseModel,
            text: str,
            number: int,
            time: datetime,
        ) -> ComplexOutput:
            nested = NestedPydanticModel(value=42, name="nested")
            pydantic_out = OutputPydanticModel(nested=nested, description="pydantic output")
            dataclass_out = NestedDataclass(count=123, active=True)
            return ComplexOutput(
                pydantic_part=pydantic_out, dataclass_part=dataclass_out, string_part="complete"
            )

        input_model = NestedPydanticModel(value=10, name="test")
        decorated_chain_complex_io(
            model=input_model, text="sample text", number=42, time=datetime(2024, 1, 1, 12, 0)
        )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_chain_complex_io"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        input_value_data = json.loads(input_value)
        assert input_value_data == {
            "model": {"value": 10, "name": "test"},
            "text": "sample text",
            "number": 42,
            "time": "2024-01-01T12:00:00",
        }
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        output_value_data = json.loads(output_value)
        assert output_value_data == {
            "pydantic_part": {
                "nested": {"value": 42, "name": "nested"},
                "description": "pydantic output",
            },
            "dataclass_part": {"count": 123, "active": True},
            "string_part": "complete",
        }
        assert not attributes

    def test_chain_applied_as_function(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        def chain_with_decorator_applied_as_function(input: str) -> str:
            return "output"

        decorated = tracer.chain(chain_with_decorator_applied_as_function)
        decorated("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "chain_with_decorator_applied_as_function"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    def test_chain_applied_as_function_with_parameters(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        def chain_with_overridden_name(input: str) -> str:
            return "output"

        decorated = tracer.chain(name="overridden-name")(chain_with_overridden_name)
        decorated("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "overridden-name"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    def test_unhandled_exception(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.chain
        def chain_with_error(input: str) -> str:
            raise ValueError("error message")

        with pytest.raises(ValueError, match="error message"):
            chain_with_error("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "chain_with_error"
        status = span.status
        assert not status.is_ok
        assert status.status_code == StatusCode.ERROR
        assert status.description == "ValueError: error message"
        assert len(span.events) == 1
        event = span.events[0]
        assert event.name == "exception"
        event_attributes = dict(event.attributes or {})
        assert event_attributes.pop("exception.type") == "ValueError"
        assert event_attributes.pop("exception.message") == "error message"
        assert event_attributes.pop("exception.stacktrace")
        assert event_attributes.pop("exception.escaped") == "False"
        assert not event_attributes
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert not attributes

    def test_method(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        class ChainRunner:
            @tracer.chain
            def decorated_chain_method(self, input1: str, input2: str) -> str:
                return "output"

        chain_runner = ChainRunner()
        chain_runner.decorated_chain_method("input1", "input2")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "ChainRunner.decorated_chain_method"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input1": "input1", "input2": "input2"}
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    def test_class_method(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        class ChainRunner:
            @tracer.chain
            @classmethod
            def decorated_chain_method(cls, input1: str, input2: str) -> str:
                return "output"

        ChainRunner.decorated_chain_method("input1", "input2")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "ChainRunner.decorated_chain_method"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input1": "input1", "input2": "input2"}
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    def test_static_method(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        class ChainRunner:
            @tracer.chain
            @staticmethod
            def decorated_chain_method(input1: str, input2: str) -> str:
                return "output"

        ChainRunner.decorated_chain_method("input1", "input2")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_chain_method"  # hard to reliably grab the class name
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input1": "input1", "input2": "input2"}
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    def test_manual_span_updates(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.chain
        def chain_with_manual_span_updates(input: str) -> str:
            span = get_current_span()
            span.set_input("overridden-input")  # type: ignore[attr-defined]
            span.set_output("overridden-output")  # type: ignore[attr-defined]
            return "output"

        chain_with_manual_span_updates("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "chain_with_manual_span_updates"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == CHAIN
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "overridden-input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "overridden-output"
        assert not attributes

    def test_suppress_tracing(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.chain
        def decorated_chain(input: str) -> str:
            return "output"

        with suppress_tracing():
            decorated_chain("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 0

    def test_using_session(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.chain
        def decorated_chain_with_session(input: str) -> str:
            return "output"

        with using_session("123"):
            decorated_chain_with_session("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_chain_with_session"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes[SESSION_ID] == "123"


class TestAgentDecorator:
    def test_plain_text_input_and_output(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.agent
        def decorated_agent(input: str) -> str:
            return "output"

        decorated_agent("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_agent"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == AGENT
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    async def test_async_with_overridden_name(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.agent(name="custom-name")
        async def decorated_agent(input: str) -> str:
            return "output"

        await decorated_agent("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "custom-name"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == AGENT
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes


class TestTracerToolDecorator:
    def test_tool_with_one_argument_and_docstring(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.tool
        def decorated_tool(input: str) -> None:
            """
            tool-description
            """

        decorated_tool("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_tool"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input": "input"}
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "None"
        assert attributes.pop(TOOL_NAME) == "decorated_tool"
        assert attributes.pop(TOOL_DESCRIPTION) == "tool-description"
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == {
            "type": "object",
            "title": "decorated_tool",
            "description": "tool-description",
            "properties": {
                "input": {
                    "type": "string",
                },
            },
            "required": ["input"],
        }
        assert not attributes

    def test_tool_with_two_arguments_and_no_docstring(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.tool
        def decorated_tool(input1: str, input2: int) -> None:
            pass

        decorated_tool("input1", 1)

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_tool"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input1": "input1", "input2": 1}
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "None"
        assert attributes.pop(TOOL_NAME) == "decorated_tool"
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == {
            "type": "object",
            "title": "decorated_tool",
            "properties": {
                "input1": {
                    "type": "string",
                },
                "input2": {
                    "type": "integer",
                },
            },
            "required": ["input1", "input2"],
        }
        assert not attributes

    def test_class_tool_with_call_method(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        class ClassTool:
            @tracer.tool
            def __call__(self, input: str) -> None:
                """
                tool-description
                """
                pass

        callable_instance = ClassTool()
        callable_instance("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "ClassTool.__call__"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input": "input"}
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "None"
        assert attributes.pop(TOOL_NAME) == "ClassTool.__call__"
        assert attributes.pop(TOOL_DESCRIPTION) == "tool-description"
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == {
            "type": "object",
            "title": "ClassTool.__call__",
            "description": "tool-description",
            "properties": {
                "input": {
                    "type": "string",
                },
            },
            "required": ["input"],
        }
        assert not attributes

    async def test_async_tool(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.tool
        async def decorated_async_tool(input: str) -> None:
            """
            tool-description
            """
            pass

        await decorated_async_tool("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated_async_tool"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input": "input"}
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "None"
        assert attributes.pop(TOOL_NAME) == "decorated_async_tool"
        assert attributes.pop(TOOL_DESCRIPTION) == "tool-description"
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == {
            "type": "object",
            "title": "decorated_async_tool",
            "description": "tool-description",
            "properties": {
                "input": {
                    "type": "string",
                },
            },
            "required": ["input"],
        }
        assert not attributes

    async def test_async_tool_with_overriddes(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        overridden_parameters = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "overridden-description",
                },
            },
            "required": ["input"],
        }

        @tracer.tool(
            name="overridden-name",
            description="overridden-description",
            parameters=overridden_parameters,
        )
        async def decorated_async_tool(input: str) -> None:
            pass

        await decorated_async_tool("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "overridden-name"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input": "input"}
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "None"
        assert attributes.pop(TOOL_NAME) == "overridden-name"
        assert attributes.pop(TOOL_DESCRIPTION) == "overridden-description"
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == overridden_parameters
        assert not attributes

    def test_tool_with_zero_arguments_and_overridden_name_and_description(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.tool(
            name="decorated-tool-with-overriden-name",
            description="overriden-tool-description",
        )
        def this_tool_name_should_be_overriden() -> None:
            """
            this tool description should be overriden
            """

        this_tool_name_should_be_overriden()

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "decorated-tool-with-overriden-name"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {}
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "None"
        assert attributes.pop(TOOL_NAME) == "decorated-tool-with-overriden-name"
        assert attributes.pop(TOOL_DESCRIPTION) == "overriden-tool-description"
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == {
            "type": "object",
            "title": "decorated-tool-with-overriden-name",
            "description": "overriden-tool-description",
            "properties": {},
        }
        assert not attributes

    def test_manual_span_updates(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        parameters = {
            "type": "object",
            "properties": {
                "input1": {
                    "type": "string",
                    "description": "First input parameter as a string",
                },
                "input2": {
                    "type": "integer",
                    "description": "Second input parameter as an integer",
                },
            },
            "required": ["input1", "input2"],
        }

        @tracer.tool
        def tool_with_manual_span_updates(input1: str, input2: int) -> str:
            span = get_current_span()
            span.set_input("inside-input")  # type: ignore[attr-defined]
            span.set_output("inside-output")  # type: ignore[attr-defined]
            span.set_tool(  # type: ignore[attr-defined]
                name="inside-tool-name",
                description="inside-tool-description",
                parameters=parameters,
            )
            return "output"

        tool_with_manual_span_updates("input1", 1)

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "tool_with_manual_span_updates"
        assert span.status.is_ok
        assert not span.events
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert attributes.pop(INPUT_MIME_TYPE) == TEXT
        assert attributes.pop(INPUT_VALUE) == "inside-input"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(OUTPUT_VALUE) == "inside-output"
        assert attributes.pop(TOOL_NAME) == "inside-tool-name"
        assert attributes.pop(TOOL_DESCRIPTION) == "inside-tool-description"
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == parameters
        assert not attributes

    def test_unhandled_exception(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.tool
        def tool_with_error(input_str: str = "default") -> str:
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            tool_with_error("input")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "tool_with_error"
        assert not span.status.is_ok
        assert span.status.status_code == StatusCode.ERROR
        assert span.status.description == "ValueError: test error"
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input_str": "input"}
        assert attributes.pop(TOOL_NAME) == "tool_with_error"
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == {
            "type": "object",
            "title": "tool_with_error",
            "properties": {
                "input_str": {
                    "type": "string",
                    "default": "default",
                },
            },
        }
        assert not attributes

    def test_suppress_tracing(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.tool
        def tool_function(input_str: str) -> str:
            return f"processed {input_str}"

        with suppress_tracing():
            result = tool_function("test input")
            assert result == "processed test input"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 0

    def test_using_session(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        session_id = "test-session-id"
        with using_session(session_id):

            @tracer.tool
            def tool_function(input_str: str) -> str:
                return f"processed {input_str}"

            result = tool_function("test input")
            assert result == "processed test input"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "tool_function"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes[SESSION_ID] == session_id


def test_infer_tool_parameters() -> None:
    class PydanticModel(pydantic.BaseModel):
        string_param: str
        int_param: int
        float_param: float
        bool_param: bool
        any_param: Any

    class TypedDictModel(TypedDict):
        """
        typed-dict-description
        """

        string_param: str
        int_param: int
        float_param: float
        bool_param: bool
        any_param: Any

    AnnotatedWithTypeAlias: TypeAlias = Annotated[str, "This is a description"]

    def example_function(  # type: ignore[no-untyped-def]
        untyped_param,
        none_param: None,
        string_param: str,
        int_param: int,
        float_param: float,
        bool_param: bool,
        datetime_param: datetime,
        any_param: Any,
        optional_int_param: Optional[int],
        union_string_int_param: Union[str, int],
        literal_string_param: Literal["hello", "world"],
        literal_string_int_param: Literal[1, "hello"],
        list_string_param: List[str],
        list_of_union_string_int_param: List[Union[str, int]],
        list_without_item_type_param: List,  # type: ignore[type-arg]
        sequence_string_param: Sequence[str],
        sequence_without_item_type_param: Sequence,  # type: ignore[type-arg]
        tuple_string_int_param: Tuple[str, int],
        tuple_of_strings_param: Tuple[str, ...],
        tuple_of_union_string_int_param: Tuple[Union[str, int], ...],
        tuple_without_item_type_param: Tuple,  # type: ignore[type-arg]
        dict_string_param: Dict[str, str],
        dict_union_string_int_param: Dict[str, Union[str, int]],
        dict_without_type_param: Dict,  # type: ignore[type-arg]
        mapping_string_param: Mapping[str, str],
        mapping_union_string_int_param: Mapping[str, Union[str, int]],
        mapping_without_type_param: Mapping,  # type: ignore[type-arg]
        annotated_string_param: Annotated[str, "This is a description"],
        annotated_param_with_type_alias: AnnotatedWithTypeAlias,
        pydantic_model_param: PydanticModel,
        typed_dict_param: TypedDictModel,
        string_param_with_default: str = "default",
        untyped_param_with_default="default",
    ) -> None:
        pass

    schema = _infer_tool_parameters(
        callable=example_function,
        tool_name="example_function",
        tool_description="example-function-description",
    )
    expected_schema = {
        "type": "object",
        "title": "example_function",
        "description": "example-function-description",
        "properties": {
            "untyped_param": {},
            "none_param": {
                "type": "null",
            },
            "string_param": {
                "type": "string",
            },
            "int_param": {
                "type": "integer",
            },
            "float_param": {
                "type": "number",
            },
            "bool_param": {
                "type": "boolean",
            },
            "datetime_param": {
                "type": "string",
                "format": "date-time",
            },
            "any_param": {},
            "optional_int_param": {
                "anyOf": [
                    {
                        "type": "integer",
                    },
                    {
                        "type": "null",
                    },
                ],
            },
            "union_string_int_param": {
                "anyOf": [
                    {
                        "type": "string",
                    },
                    {
                        "type": "integer",
                    },
                ]
            },
            "literal_string_param": {
                "type": "string",
                "enum": [
                    "hello",
                    "world",
                ],
            },
            "literal_string_int_param": {
                "anyOf": [
                    {
                        "type": "integer",
                    },
                    {
                        "type": "string",
                    },
                ],
                "enum": [
                    1,
                    "hello",
                ],
            },
            "list_string_param": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "list_of_union_string_int_param": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {
                            "type": "string",
                        },
                        {
                            "type": "integer",
                        },
                    ]
                },
            },
            "list_without_item_type_param": {
                "type": "array",
            },
            "sequence_string_param": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "sequence_without_item_type_param": {
                "type": "array",
            },
            "tuple_string_int_param": {
                "type": "array",
                "items": [
                    {
                        "type": "string",
                    },
                    {
                        "type": "integer",
                    },
                ],
                "minItems": 2,
                "maxItems": 2,
            },
            "tuple_of_strings_param": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "tuple_of_union_string_int_param": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {
                            "type": "string",
                        },
                        {
                            "type": "integer",
                        },
                    ]
                },
            },
            "tuple_without_item_type_param": {
                "type": "array",
            },
            "dict_string_param": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                },
            },
            "dict_union_string_int_param": {
                "type": "object",
                "additionalProperties": {
                    "anyOf": [
                        {
                            "type": "string",
                        },
                        {
                            "type": "integer",
                        },
                    ]
                },
            },
            "dict_without_type_param": {
                "type": "object",
            },
            "mapping_string_param": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                },
            },
            "mapping_union_string_int_param": {
                "type": "object",
                "additionalProperties": {
                    "anyOf": [
                        {
                            "type": "string",
                        },
                        {
                            "type": "integer",
                        },
                    ]
                },
            },
            "mapping_without_type_param": {
                "type": "object",
            },
            "annotated_string_param": {
                "type": "string",
                "description": "This is a description",
            },
            "annotated_param_with_type_alias": {
                "type": "string",
                "description": "This is a description",
            },
            "pydantic_model_param": {
                "properties": {
                    "string_param": {
                        "title": "String Param",
                        "type": "string",
                    },
                    "int_param": {
                        "title": "Int Param",
                        "type": "integer",
                    },
                    "float_param": {
                        "title": "Float Param",
                        "type": "number",
                    },
                    "bool_param": {
                        "title": "Bool Param",
                        "type": "boolean",
                    },
                    "any_param": {
                        "title": "Any Param",
                    },
                },
                "required": [
                    "string_param",
                    "int_param",
                    "float_param",
                    "bool_param",
                    "any_param",
                ],
                "title": "PydanticModel",
                "type": "object",
            },
            "typed_dict_param": {
                "type": "object",
                "properties": {
                    "string_param": {
                        "type": "string",
                    },
                    "int_param": {
                        "type": "integer",
                    },
                    "float_param": {
                        "type": "number",
                    },
                    "bool_param": {
                        "type": "boolean",
                    },
                    "any_param": {},
                },
            },
            "string_param_with_default": {
                "default": "default",
                "type": "string",
            },
            "untyped_param_with_default": {
                "default": "default",
            },
        },
        "required": [
            "untyped_param",
            "none_param",
            "string_param",
            "int_param",
            "float_param",
            "bool_param",
            "datetime_param",
            "any_param",
            "optional_int_param",
            "union_string_int_param",
            "literal_string_param",
            "literal_string_int_param",
            "list_string_param",
            "list_of_union_string_int_param",
            "list_without_item_type_param",
            "sequence_string_param",
            "sequence_without_item_type_param",
            "tuple_string_int_param",
            "tuple_of_strings_param",
            "tuple_of_union_string_int_param",
            "tuple_without_item_type_param",
            "dict_string_param",
            "dict_union_string_int_param",
            "dict_without_type_param",
            "mapping_string_param",
            "mapping_union_string_int_param",
            "mapping_without_type_param",
            "annotated_string_param",
            "annotated_param_with_type_alias",
            "pydantic_model_param",
            "typed_dict_param",
        ],
    }
    jsonschema.Draft7Validator.check_schema(
        expected_schema
    )  # check that the expected schema is valid
    assert schema == expected_schema


# mime types
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

# span kinds
AGENT = OpenInferenceSpanKindValues.AGENT.value
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
TOOL = OpenInferenceSpanKindValues.TOOL.value

# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
SESSION_ID = SpanAttributes.SESSION_ID
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS
