import json
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import jsonschema
import pydantic
import pytest
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode, get_current_span
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel
from typing_extensions import Annotated, TypeAlias

from openinference.instrumentation import (
    Image,
    ImageMessageContent,
    Message,
    OITracer,
    PromptDetails,
    TextMessageContent,
    TokenCount,
    Tool,
    ToolCall,
    ToolCallFunction,
    get_llm_attributes,
    suppress_tracing,
    using_session,
)
from openinference.instrumentation._tracers import _infer_tool_parameters
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


def remove_all_vcr_request_headers(request: Any) -> Any:
    """
    Removes all request headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_request_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    request.headers.clear()
    return request


def remove_all_vcr_response_headers(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes all response headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_response_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    response["headers"] = {}
    return response


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


class TestTracerLLMDecorator:
    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_sync_function_with_unapplied_decorator(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
        sync_openai_client: OpenAI,
    ) -> None:
        @tracer.llm
        def sync_llm_function(input_messages: List[ChatCompletionMessageParam]) -> ChatCompletion:
            return sync_openai_client.chat.completions.create(
                model="gpt-4o",
                messages=input_messages,
                temperature=0.5,
            )

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Who won the World Cup in 2022? Answer in one word with no punctuation.",
            )
        ]
        response = sync_llm_function(input_messages)
        output_message = response.choices[0].message
        assert output_message.content == "Argentina"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "sync_llm_function"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input_messages": input_messages}
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert json.loads(output_value) == response.model_dump()
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert not attributes

    def test_unhandled_exception_in_sync_function_with_unapplied_decorator(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.llm
        def sync_llm_function(input_messages: List[ChatCompletionMessageParam]) -> ChatCompletion:
            raise ValueError("Something went wrong")

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content="Test message")
        ]

        with pytest.raises(ValueError):
            sync_llm_function(input_messages)

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "sync_llm_function"
        assert not span.status.is_ok
        assert span.status.status_code == StatusCode.ERROR
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input_messages": input_messages}
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert not attributes

        events = span.events
        assert len(events) == 1
        event = events[0]
        assert event.name == "exception"
        attributes = dict(event.attributes or {})
        assert attributes["exception.type"] == "ValueError"
        assert attributes["exception.message"] == "Something went wrong"

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    async def test_async_function_with_unapplied_decorator(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
        async_openai_client: AsyncOpenAI,
    ) -> None:
        @tracer.llm
        async def async_llm_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> ChatCompletion:
            return await async_openai_client.chat.completions.create(
                model="gpt-4o",
                messages=input_messages,
                temperature=0.5,
            )

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Who won the World Cup in 2022? Answer in one word with no punctuation.",
            )
        ]
        response = await async_llm_function(input_messages)
        output_message = response.choices[0].message
        assert output_message.content == "Argentina"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "async_llm_function"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input_messages": input_messages}
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert json.loads(output_value) == response.model_dump()
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert not attributes

    async def test_unhandled_exception_in_async_function_with_unapplied_decorator(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.llm
        async def async_llm_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> ChatCompletion:
            raise ValueError("Something went wrong")

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content="Test message")
        ]

        with pytest.raises(ValueError):
            await async_llm_function(input_messages)

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "async_llm_function"
        assert not span.status.is_ok
        assert span.status.status_code == StatusCode.ERROR
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input_messages": input_messages}
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert not attributes

        events = span.events
        assert len(events) == 1
        event = events[0]
        assert event.name == "exception"
        attributes = dict(event.attributes or {})
        assert attributes["exception.type"] == "ValueError"
        assert attributes["exception.message"] == "Something went wrong"

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_sync_generator_with_unapplied_decorator(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
        sync_openai_client: OpenAI,
    ) -> None:
        @tracer.llm
        def sync_llm_generator_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> Generator[ChatCompletionChunk, None, None]:
            stream = sync_openai_client.chat.completions.create(
                model="gpt-4o",
                messages=input_messages,
                temperature=0.5,
                stream=True,
            )
            for chunk in stream:
                yield chunk

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Who won the World Cup in 2022? Answer with a sentence of the form: "
                "'The winner of the World Cup in 2022 was <country>.'.",
            )
        ]
        chunks = list(sync_llm_generator_function(input_messages))
        content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert content == "The winner of the World Cup in 2022 was Argentina."

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "sync_llm_generator_function"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input_messages": input_messages}
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert json.loads(output_value) == [chunk.model_dump() for chunk in chunks]
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert not attributes

    def test_unhandled_exception_in_sync_generator_function_with_unapplied_decorator(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.llm
        def sync_llm_generator_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> Generator[str, None, None]:
            yield "Argentina "
            yield "won"
            yield "the "
            yield "World "
            raise ValueError("Something went wrong")

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content="Who won the World Cup in 2022?")
        ]

        with pytest.raises(ValueError):
            for chunk in sync_llm_generator_function(input_messages):
                pass

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "sync_llm_generator_function"
        assert not span.status.is_ok
        assert span.status.status_code == StatusCode.ERROR
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input_messages": input_messages}
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert json.loads(output_value) == ["Argentina ", "won", "the ", "World "]
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert not attributes

        events = span.events
        assert len(events) == 1
        event = events[0]
        assert event.name == "exception"
        attributes = dict(event.attributes or {})
        assert attributes["exception.type"] == "ValueError"
        assert attributes["exception.message"] == "Something went wrong"

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    async def test_async_generator_with_unapplied_decorator(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
        async_openai_client: AsyncOpenAI,
    ) -> None:
        @tracer.llm
        async def async_llm_generator_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> AsyncGenerator[ChatCompletionChunk, None]:
            stream = await async_openai_client.chat.completions.create(
                model="gpt-4o",
                messages=input_messages,
                temperature=0.5,
                stream=True,
            )
            async for chunk in stream:
                yield chunk

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Who won the World Cup in 2022? Answer with a sentence of the form: "
                "'The winner of the World Cup in 2022 was <country>.'.",
            )
        ]
        chunks = []
        async for chunk in async_llm_generator_function(input_messages):
            chunks.append(chunk)
        content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert content == "The winner of the World Cup in 2022 was Argentina."

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "async_llm_generator_function"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input_messages": input_messages}
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert json.loads(output_value) == [chunk.model_dump() for chunk in chunks]
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert not attributes

    async def test_unhandled_exception_in_async_generator_function_with_unapplied_decorator(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        @tracer.llm
        async def async_llm_generator_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> AsyncGenerator[str, None]:
            yield "Argentina "
            yield "won"
            yield "the "
            yield "World "
            raise ValueError("Something went wrong")

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content="Who won the World Cup in 2022?")
        ]

        with pytest.raises(ValueError):
            async for chunk in async_llm_generator_function(input_messages):
                pass

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "async_llm_generator_function"
        assert not span.status.is_ok
        assert span.status.status_code == StatusCode.ERROR
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {"input_messages": input_messages}
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert json.loads(output_value) == ["Argentina ", "won", "the ", "World "]
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert not attributes

        events = span.events
        assert len(events) == 1
        event = events[0]
        assert event.name == "exception"
        attributes = dict(event.attributes or {})
        assert attributes["exception.type"] == "ValueError"
        assert attributes["exception.message"] == "Something went wrong"

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_sync_function_with_applied_decorator(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
        sync_openai_client: OpenAI,
    ) -> None:
        def get_input_attributes(
            input_messages: List[ChatCompletionMessageParam],
        ) -> "Mapping[str, AttributeValue]":
            return {INPUT_VALUE: "input-messages"}

        def get_output_attributes(output_message: ChatCompletion) -> "Mapping[str, AttributeValue]":
            return {OUTPUT_VALUE: "output"}

        @tracer.llm(
            name="custom-llm-name",
            process_input=get_input_attributes,
            process_output=get_output_attributes,
        )
        def sync_llm_function(input_messages: List[ChatCompletionMessageParam]) -> ChatCompletion:
            return sync_openai_client.chat.completions.create(
                model="gpt-4o",
                messages=input_messages,
                temperature=0.5,
            )

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Who won the World Cup in 2022? Answer in one word with no punctuation.",
            )
        ]
        response = sync_llm_function(input_messages)
        output_message = response.choices[0].message
        assert output_message.content == "Argentina"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "custom-llm-name"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_VALUE) == "input-messages"
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    def test_unhandled_exception_in_sync_function_with_applied_decorator(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        def get_input_attributes(
            input_messages: List[ChatCompletionMessageParam],
        ) -> "Mapping[str, AttributeValue]":
            return {INPUT_VALUE: "input-messages"}

        def get_output_attributes(output_message: ChatCompletion) -> "Mapping[str, AttributeValue]":
            return {OUTPUT_VALUE: "output"}

        @tracer.llm(
            name="custom-llm-name",
            process_input=get_input_attributes,
            process_output=get_output_attributes,
        )
        def sync_llm_function(input_messages: List[ChatCompletionMessageParam]) -> ChatCompletion:
            raise ValueError("Something went wrong")

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content="Test message")
        ]

        with pytest.raises(ValueError):
            sync_llm_function(input_messages)

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "custom-llm-name"
        assert not span.status.is_ok
        assert span.status.status_code == StatusCode.ERROR
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_VALUE) == "input-messages"
        assert not attributes

        events = span.events
        assert len(events) == 1
        event = events[0]
        assert event.name == "exception"
        attributes = dict(event.attributes or {})
        assert attributes["exception.type"] == "ValueError"
        assert attributes["exception.message"] == "Something went wrong"

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    async def test_async_function_with_applied_decorator(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
        async_openai_client: AsyncOpenAI,
    ) -> None:
        def get_input_attributes(
            input_messages: List[ChatCompletionMessageParam],
        ) -> "Mapping[str, AttributeValue]":
            return {INPUT_VALUE: "input-messages"}

        def get_output_attributes(output_message: ChatCompletion) -> "Mapping[str, AttributeValue]":
            return {OUTPUT_VALUE: "output"}

        @tracer.llm(
            name="custom-llm-name",
            process_input=get_input_attributes,
            process_output=get_output_attributes,
        )
        async def async_llm_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> ChatCompletion:
            return await async_openai_client.chat.completions.create(
                model="gpt-4o",
                messages=input_messages,
                temperature=0.5,
            )

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Who won the World Cup in 2022? Answer in one word with no punctuation.",
            )
        ]
        response = await async_llm_function(input_messages)
        output_message = response.choices[0].message
        assert output_message.content == "Argentina"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "custom-llm-name"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_VALUE) == "input-messages"
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    async def test_unhandled_exception_in_async_function_with_applied_decorator(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        def get_input_attributes(
            input_messages: List[ChatCompletionMessageParam],
        ) -> "Mapping[str, AttributeValue]":
            return {INPUT_VALUE: "input-messages"}

        def get_output_attributes(output_message: ChatCompletion) -> "Mapping[str, AttributeValue]":
            return {OUTPUT_VALUE: "output"}

        @tracer.llm(
            name="custom-llm-name",
            process_input=get_input_attributes,
            process_output=get_output_attributes,
        )
        async def async_llm_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> ChatCompletion:
            raise ValueError("Something went wrong")

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content="Test message")
        ]

        with pytest.raises(ValueError):
            await async_llm_function(input_messages)

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "custom-llm-name"
        assert not span.status.is_ok
        assert span.status.status_code == StatusCode.ERROR
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_VALUE) == "input-messages"
        assert not attributes

        events = span.events
        assert len(events) == 1
        event = events[0]
        assert event.name == "exception"
        attributes = dict(event.attributes or {})
        assert attributes["exception.type"] == "ValueError"
        assert attributes["exception.message"] == "Something went wrong"

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_sync_generator_with_applied_decorator(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
        sync_openai_client: OpenAI,
    ) -> None:
        def get_input_attributes(
            input_messages: List[ChatCompletionMessageParam],
        ) -> "Mapping[str, AttributeValue]":
            return {INPUT_VALUE: "input-messages"}

        def get_output_attributes(
            outputs: Sequence[ChatCompletionChunk],
        ) -> "Mapping[str, AttributeValue]":
            return {OUTPUT_VALUE: "output"}

        @tracer.llm(
            name="custom-llm-name",
            process_input=get_input_attributes,
            process_output=get_output_attributes,
        )
        def sync_llm_generator_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> Generator[ChatCompletionChunk, None, None]:
            stream = sync_openai_client.chat.completions.create(
                model="gpt-4o",
                messages=input_messages,
                temperature=0.5,
                stream=True,
            )
            for chunk in stream:
                yield chunk

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Who won the World Cup in 2022? Answer with a sentence of the form: "
                "'The winner of the World Cup in 2022 was <country>.'.",
            )
        ]
        chunks = list(sync_llm_generator_function(input_messages))
        content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert content == "The winner of the World Cup in 2022 was Argentina."

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "custom-llm-name"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_VALUE) == "input-messages"
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    def test_unhandled_exception_in_sync_generator_function_with_applied_decorator(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        def get_input_attributes(
            input_messages: List[ChatCompletionMessageParam],
        ) -> "Mapping[str, AttributeValue]":
            return {INPUT_VALUE: "input-messages"}

        def get_output_attributes(
            outputs: Sequence[str],
        ) -> "Mapping[str, AttributeValue]":
            return {OUTPUT_VALUE: "".join(outputs)}

        @tracer.llm(
            name="custom-llm-name",
            process_input=get_input_attributes,
            process_output=get_output_attributes,
        )
        def sync_llm_generator_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> Generator[str, None, None]:
            yield "Argentina "
            yield "won "
            yield "the "
            yield "World "
            raise ValueError("Something went wrong")

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content="Who won the World Cup in 2022?")
        ]

        with pytest.raises(ValueError):
            for chunk in sync_llm_generator_function(input_messages):
                pass

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "custom-llm-name"
        assert not span.status.is_ok
        assert span.status.status_code == StatusCode.ERROR
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_VALUE) == "input-messages"
        assert attributes.pop(OUTPUT_VALUE) == "Argentina won the World "
        assert not attributes

        events = span.events
        assert len(events) == 1
        event = events[0]
        assert event.name == "exception"
        attributes = dict(event.attributes or {})
        assert attributes["exception.type"] == "ValueError"
        assert attributes["exception.message"] == "Something went wrong"

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    async def test_async_generator_with_custom_attributes(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
        async_openai_client: AsyncOpenAI,
    ) -> None:
        def get_input_attributes(
            input_messages: List[ChatCompletionMessageParam],
        ) -> "Mapping[str, AttributeValue]":
            return {INPUT_VALUE: "input-messages"}

        def get_output_attributes(
            outputs: Sequence[ChatCompletionChunk],
        ) -> "Mapping[str, AttributeValue]":
            return {OUTPUT_VALUE: "output"}

        @tracer.llm(
            name="custom-llm-name",
            process_input=get_input_attributes,
            process_output=get_output_attributes,
        )
        async def async_llm_generator_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> AsyncGenerator[ChatCompletionChunk, None]:
            stream = await async_openai_client.chat.completions.create(
                model="gpt-4o",
                messages=input_messages,
                temperature=0.5,
                stream=True,
            )
            async for chunk in stream:
                yield chunk

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(
                role="user",
                content="Who won the World Cup in 2022? Answer with a sentence of the form: "
                "'The winner of the World Cup in 2022 was <country>.'.",
            )
        ]
        chunks = [chunk async for chunk in async_llm_generator_function(input_messages)]
        content = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert content == "The winner of the World Cup in 2022 was Argentina."

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "custom-llm-name"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_VALUE) == "input-messages"
        assert attributes.pop(OUTPUT_VALUE) == "output"
        assert not attributes

    async def test_unhandled_exception_in_async_generator_function_with_applied_decorator(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer: OITracer,
    ) -> None:
        def get_input_attributes(
            input_messages: List[ChatCompletionMessageParam],
        ) -> "Mapping[str, AttributeValue]":
            return {INPUT_VALUE: "input-messages"}

        def get_output_attributes(
            outputs: Sequence[str],
        ) -> "Mapping[str, AttributeValue]":
            return {OUTPUT_VALUE: "".join(outputs)}

        @tracer.llm(
            name="custom-llm-name",
            process_input=get_input_attributes,
            process_output=get_output_attributes,
        )
        async def async_llm_generator_function(
            input_messages: List[ChatCompletionMessageParam],
        ) -> AsyncGenerator[str, None]:
            yield "Argentina "
            yield "won "
            yield "the "
            yield "World "
            raise ValueError("Something went wrong")

        input_messages: List[ChatCompletionMessageParam] = [
            ChatCompletionUserMessageParam(role="user", content="Who won the World Cup in 2022?")
        ]

        with pytest.raises(ValueError):
            async for chunk in async_llm_generator_function(input_messages):
                pass

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        assert span.name == "custom-llm-name"
        assert not span.status.is_ok
        assert span.status.status_code == StatusCode.ERROR
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_VALUE) == "input-messages"
        assert attributes.pop(OUTPUT_VALUE) == "Argentina won the World "
        assert not attributes

        events = span.events
        assert len(events) == 1
        event = events[0]
        assert event.name == "exception"
        attributes = dict(event.attributes or {})
        assert attributes["exception.type"] == "ValueError"
        assert attributes["exception.message"] == "Something went wrong"


def test_get_llm_attributes_returns_expected_attributes() -> None:
    input_messages: Sequence[Message] = [
        Message(
            role="user",
            content="Hello",
            contents=[
                TextMessageContent(type="text", text="Hello"),
                ImageMessageContent(type="image", image=Image(url="https://example.com/image.jpg")),
            ],
            tool_call_id="call-123",
            tool_calls=[
                ToolCall(
                    id="call-456",
                    function=ToolCallFunction(
                        name="search",
                        arguments='{"query": "test"}',
                    ),
                ),
                ToolCall(
                    id="call-789",
                    function=ToolCallFunction(
                        name="calculate",
                        arguments={"operation": "add", "numbers": [1, 2, 3]},
                    ),
                ),
            ],
        )
    ]
    output_messages: Sequence[Message] = [
        Message(
            role="assistant",
            content="Hi there!",
            contents=[TextMessageContent(type="text", text="Hi there!")],
        )
    ]
    token_count: TokenCount = TokenCount(
        prompt=10,
        completion=5,
        total=15,
        prompt_details=PromptDetails(
            audio=3,
            cache_read=2,
            cache_write=1,
        ),
    )
    tools: Sequence[Tool] = [
        Tool(
            json_schema=json.dumps({"type": "object", "properties": {"query": {"type": "string"}}})
        ),
        Tool(json_schema={"type": "object", "properties": {"operation": {"type": "string"}}}),
    ]
    attributes = get_llm_attributes(
        provider="openai",
        system="openai",
        model_name="gpt-4",
        invocation_parameters={"temperature": 0.7, "max_tokens": 100},
        input_messages=input_messages,
        output_messages=output_messages,
        token_count=token_count,
        tools=tools,
    )
    assert attributes.pop(LLM_PROVIDER) == "openai"
    assert attributes.pop(LLM_SYSTEM) == "openai"
    assert attributes.pop(LLM_MODEL_NAME) == "gpt-4"
    invocation_params = attributes.pop(LLM_INVOCATION_PARAMETERS)
    assert isinstance(invocation_params, str)
    params_dict = json.loads(invocation_params)
    assert params_dict == {"temperature": 0.7, "max_tokens": 100}
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "Hello"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
        == "Hello"
    )
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}")
        == "image"
    )
    assert (
        attributes.pop(
            f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}"
        )
        == "https://example.com/image.jpg"
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_TOOL_CALL_ID}") == "call-123"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_ID}")
        == "call-456"
    )
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "search"
    )
    assert (
        attributes.pop(
            f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        )
        == '{"query": "test"}'
    )
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_ID}")
        == "call-789"
    )
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_NAME}")
        == "calculate"
    )
    function_args = attributes.pop(
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )
    assert isinstance(function_args, str)
    assert json.loads(function_args) == {"operation": "add", "numbers": [1, 2, 3]}
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "Hi there!"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
        == "Hi there!"
    )
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO) == 3
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 2
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE) == 1
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 5
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 15
    assert (
        attributes.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}")
        == '{"type": "object", "properties": {"query": {"type": "string"}}}'
    )
    tool_schema = attributes.pop(f"{LLM_TOOLS}.1.{TOOL_JSON_SCHEMA}")
    assert isinstance(tool_schema, str)
    assert json.loads(tool_schema) == {
        "type": "object",
        "properties": {"operation": {"type": "string"}},
    }


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


# Image attributes
IMAGE_URL = ImageAttributes.IMAGE_URL

# Message attributes
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS

# Message content attributes
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE

# Span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOOLS = SpanAttributes.LLM_TOOLS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS

# Tool attributes
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA

# Tool call attributes
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID

# Mime types
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

# Span kinds
AGENT = OpenInferenceSpanKindValues.AGENT.value
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
LLM = OpenInferenceSpanKindValues.LLM.value
TOOL = OpenInferenceSpanKindValues.TOOL.value

# Session ID
SESSION_ID = SpanAttributes.SESSION_ID


class TestSamplerAttributeAccess:
    """Test that samplers receive attributes during span creation for sampling decisions."""

    def test_sampler_receives_all_attributes(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        captured_attributes: Dict[str, Any] = {}

        from typing import Optional, Sequence

        # Import the actual types used in the signature
        import opentelemetry.trace
        from opentelemetry.context import Context
        from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
        from opentelemetry.trace import Link
        from opentelemetry.util.types import Attributes

        class AttributeCapturingSampler(Sampler):
            def should_sample(
                self,
                parent_context: Optional[Context],
                trace_id: int,
                name: str,
                kind: Optional[opentelemetry.trace.SpanKind] = None,
                attributes: Optional[Attributes] = None,
                links: Optional[Sequence[Link]] = None,
                trace_state: Optional[opentelemetry.trace.TraceState] = None,
            ) -> SamplingResult:
                print(f"SAMPLER CALLED: name={name}, attributes={attributes}")
                if attributes:
                    captured_attributes.update(attributes)
                    print(f"CAPTURED: {captured_attributes}")
                return SamplingResult(Decision.RECORD_AND_SAMPLE)

            def get_description(self) -> str:
                return "AttributeCapturingSampler"

        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from openinference.instrumentation import TraceConfig, TracerProvider, using_attributes

        # Create TracerProvider with custom sampler
        tracer_provider = TracerProvider(config=TraceConfig(), sampler=AttributeCapturingSampler())
        tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
        tracer = tracer_provider.get_tracer(__name__)

        user_attributes = {"user.custom": "user_value", "user.id": "123"}

        with using_attributes(session_id="session_123", metadata={"key": "value"}):
            with tracer.start_as_current_span(
                "test-span",
                openinference_span_kind="chain",
                attributes=user_attributes,
            ) as span:
                span.set_input("test input")
                span.set_output("test output")
                print(f"SPAN ATTRIBUTES: {getattr(span, 'attributes', 'no attributes attr')}")

        print(f"FINAL CAPTURED ATTRIBUTES: {captured_attributes}")
        spans = in_memory_span_exporter.get_finished_spans()
        print(f"EXPORTED SPANS: {len(spans)}")
        if spans:
            print(f"EXPORTED SPAN ATTRIBUTES: {dict(spans[0].attributes or {})}")

        assert "user.custom" in captured_attributes
        assert captured_attributes["user.custom"] == "user_value"
        assert "user.id" in captured_attributes
        assert captured_attributes["user.id"] == "123"
        assert "openinference.span.kind" in captured_attributes
        assert captured_attributes["openinference.span.kind"] == "CHAIN"
        assert "session.id" in captured_attributes
        assert captured_attributes["session.id"] == "session_123"
        assert "metadata" in captured_attributes  # metadata is JSON-encoded

    def test_sampler_receives_masked_attributes(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        captured_attributes: Dict[str, Any] = {}

        from typing import Optional, Sequence, Union

        import opentelemetry.trace
        from opentelemetry.context import Context
        from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
        from opentelemetry.trace import Link
        from opentelemetry.util.types import Attributes

        class AttributeCapturingSampler(Sampler):
            def should_sample(
                self,
                parent_context: Optional[Context],
                trace_id: int,
                name: str,
                kind: Optional[opentelemetry.trace.SpanKind] = None,
                attributes: Optional[Attributes] = None,
                links: Optional[Sequence[Link]] = None,
                trace_state: Optional[opentelemetry.trace.TraceState] = None,
            ) -> SamplingResult:
                print(f"SAMPLER CALLED: name={name}, attributes={attributes}")
                if attributes:
                    captured_attributes.update(attributes)
                    print(f"CAPTURED: {captured_attributes}")
                return SamplingResult(Decision.RECORD_AND_SAMPLE)

            def get_description(self) -> str:
                return "AttributeCapturingSampler"

        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.util.types import AttributeValue

        from openinference.instrumentation import TraceConfig, TracerProvider

        class CustomTraceConfig(TraceConfig):
            def mask(
                self,
                key: str,
                value: Union[AttributeValue, Callable[[], AttributeValue]],
            ) -> Optional[AttributeValue]:
                if "sensitive" in key.lower() or "password" in key.lower():
                    return "[MASKED]"
                return super().mask(key, value)

        trace_config = CustomTraceConfig()

        # Create TracerProvider with custom sampler and config
        tracer_provider = TracerProvider(config=trace_config, sampler=AttributeCapturingSampler())
        tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
        tracer = tracer_provider.get_tracer(__name__)

        sensitive_attributes = {
            "user.password": "secret123",
            "sensitive_data": "private_info",
            "normal_attribute": "normal_value",
        }

        with tracer.start_as_current_span(
            "test-span",
            openinference_span_kind="llm",
            attributes=sensitive_attributes,
        ):
            pass

        assert "user.password" in captured_attributes
        assert captured_attributes["user.password"] == "[MASKED]"
        assert "sensitive_data" in captured_attributes
        assert captured_attributes["sensitive_data"] == "[MASKED]"
        assert "normal_attribute" in captured_attributes
        assert captured_attributes["normal_attribute"] == "normal_value"
        assert "openinference.span.kind" in captured_attributes
        assert captured_attributes["openinference.span.kind"] == "LLM"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
