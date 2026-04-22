import json
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points
from smolagents import LiteLLMModel, OpenAIServerModel, Tool, tool
from smolagents.agents import (  # type: ignore[import-untyped]
    CodeAgent,
    ToolCallingAgent,
)
from smolagents.models import (  # type: ignore[import-untyped]
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
)

from openinference.instrumentation import OITracer
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from openinference.instrumentation.smolagents._wrappers import infer_llm_provider_from_class_name
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(
            group="opentelemetry_instrumentor", name="smolagents"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, SmolagentsInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self) -> None:
        assert isinstance(SmolagentsInstrumentor()._tracer, OITracer)


class TestModels:
    @pytest.mark.vcr
    def test_openai_server_model_has_expected_attributes(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        model = OpenAIServerModel(
            model_id="gpt-4o",
            api_key=openai_api_key,
            api_base="https://api.openai.com/v1",
        )
        input_message_content = (
            "Who won the World Cup in 2018? Answer in one word with no punctuation."
        )
        output_message = model(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": input_message_content}],
                }
            ]
        )
        output_message_content = output_message.content
        assert output_message_content == "France"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "OpenAIModel.generate"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        input_data = json.loads(input_value)
        assert "messages" in input_data
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert isinstance(json.loads(output_value), dict)
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4o"
        assert attributes.pop(LLM_PROVIDER, None) == OpenInferenceLLMProviderValues.OPENAI.value
        assert attributes.pop(LLM_SYSTEM, None) == OpenInferenceLLMSystemValues.OPENAI.value
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {}
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message_content
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert (
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
            == output_message_content
        )
        assert (
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
            == "text"
        )
        assert not attributes

    @pytest.mark.vcr
    def test_openai_server_model_with_tool_has_expected_attributes(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer_provider: trace_api.TracerProvider,
    ) -> None:
        model = OpenAIServerModel(
            model_id="gpt-4o",
            api_key=openai_api_key,
            api_base="https://api.openai.com/v1",
        )
        input_message_content = "What is the weather in Paris?"

        class GetWeatherTool(Tool):  # type: ignore[misc]
            name = "get_weather"
            description = "Get the weather for a given city"
            inputs = {
                "location": {"type": "string", "description": "The city to get the weather for"}
            }
            output_type = "string"

            def forward(self, location: str) -> str:
                return "sunny"

        output_message = model(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": input_message_content}],
                }
            ],
            tools_to_call_from=[GetWeatherTool()],
        )
        output_message_content = output_message.content
        assert output_message_content is None
        tool_calls = output_message.tool_calls
        assert len(tool_calls) == 1
        tool_call = tool_calls[0]
        tool_function = getattr(tool_call, "function", None)
        assert tool_function is not None
        assert getattr(tool_function, "name", None) == "get_weather"
        assert getattr(tool_function, "arguments", None) == '{"location":"Paris"}'

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "OpenAIModel.generate"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        input_data = json.loads(input_value)
        assert "messages" in input_data
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert isinstance(json.loads(output_value), dict)
        assert attributes.pop(LLM_MODEL_NAME) == "gpt-4o"
        assert attributes.pop(LLM_PROVIDER, None) == OpenInferenceLLMProviderValues.OPENAI.value
        assert attributes.pop(LLM_SYSTEM, None) == OpenInferenceLLMSystemValues.OPENAI.value
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {}
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message_content
        assert isinstance(
            tool_json_schema := attributes.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}"), str
        )
        assert json.loads(tool_json_schema) == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city to get the weather for",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert (
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_ID}")
            == tool_call.id
        )
        assert (
            attributes.pop(
                f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}"
            )
            == "get_weather"
        )
        assert isinstance(
            tool_call_arguments_json := attributes.pop(
                f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
            ),
            str,
        )
        assert json.loads(tool_call_arguments_json) == {"location": "Paris"}
        assert not attributes

    @pytest.mark.vcr
    def test_litellm_reasoning_model_has_expected_attributes(
        self,
        anthropic_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
        patch_tiktoken_encoding: None,
    ) -> None:
        model_params = {"thinking": {"type": "enabled", "budget_tokens": 4000}}

        model = LiteLLMModel(
            model_id="anthropic/claude-3-7-sonnet-20250219",
            api_key=anthropic_api_key,
            **model_params,
        )

        input_message_content = (
            "Who won the World Cup in 2018? Answer in one word with no punctuation."
        )
        output_message = model(
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": input_message_content}],
                }
            ]
        )
        output_message_content = output_message.content
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "LiteLLMModel.generate"
        assert span.status.is_ok
        attributes = dict(span.attributes or {})
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == LLM
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        input_data = json.loads(input_value)
        assert "messages" in input_data
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert isinstance(json.loads(output_value), dict)
        assert attributes.pop(LLM_MODEL_NAME) == "anthropic/claude-3-7-sonnet-20250219"
        assert attributes.pop(LLM_PROVIDER, None) == OpenInferenceLLMProviderValues.ANTHROPIC.value
        assert attributes.pop(LLM_SYSTEM, None) == OpenInferenceLLMSystemValues.ANTHROPIC.value
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == model_params
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message_content
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert (
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
            == output_message_content
        )
        assert (
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
            == "text"
        )
        assert isinstance(
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}"),
            str,
        )
        assert (
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}")
            == "text"
        )
        assert not attributes


class TestRuns:
    def test_streaming_and_non_streaming_code_agent_runs(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        class FakeModel:
            def __init__(self, streaming: bool = False):
                self.streaming = streaming
                self.call_count = 0

            def __call__(
                self,
                messages: list[dict[str, Any]],
                stop_sequences: Optional[list[str]] = None,
                grammar: Optional[Any] = None,
                tools_to_call_from: Optional[list[Tool]] = None,
            ) -> Any:
                self.call_count += 1
                return ChatMessage(
                    role="assistant",
                    content="""
Thought: Let's return a simple answer.
Code:
```py
final_answer("Test result from CodeAgent")
```<end_code>
""",
                )

            def generate(
                self,
                messages: list[dict[str, Any]],
                stop_sequences: Optional[list[str]] = None,
                grammar: Optional[Any] = None,
                tools_to_call_from: Optional[list[Tool]] = None,
            ) -> Any:
                return self(messages, stop_sequences, grammar, tools_to_call_from)

        # Handle non-streaming (normal) run
        non_streaming_model = FakeModel(streaming=False)
        code_agent_non_stream = CodeAgent(
            tools=[],
            model=non_streaming_model,
            max_steps=5,
            additional_authorized_imports=["json", "os"],
        )
        result_non_stream = code_agent_non_stream.run("Test question for non-streaming")

        assert result_non_stream == "Test result from CodeAgent"

        non_stream_spans = in_memory_span_exporter.get_finished_spans()
        assert len(non_stream_spans) > 0

        agent_spans_non_stream = [
            span
            for span in non_stream_spans
            if span.attributes is not None
            and span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.AGENT.value
        ]
        assert len(agent_spans_non_stream) >= 1

        # Check attributes in main span
        main_span_non_stream = agent_spans_non_stream[0]
        main_span_non_stream_attributes = dict(main_span_non_stream.attributes or {})
        assert main_span_non_stream.name == "CodeAgent.run"
        assert main_span_non_stream_attributes.get(SpanAttributes.INPUT_VALUE) is not None
        assert (
            main_span_non_stream_attributes.get(SpanAttributes.OUTPUT_VALUE)
            == "Test result from CodeAgent"
        )
        assert main_span_non_stream_attributes.get("smolagents.max_steps") == 5

        step_spans_non_stream = [
            span
            for span in non_stream_spans
            if span.attributes is not None
            and span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.CHAIN.value
        ]
        assert len(step_spans_non_stream) >= 1

        # Check attributes in step span
        step_span_non_stream = step_spans_non_stream[0]
        step_span_non_stream_attributes = dict(step_span_non_stream.attributes or {})
        assert step_span_non_stream.name == "Step 1"
        assert step_span_non_stream_attributes.get(SpanAttributes.INPUT_VALUE) is not None
        assert step_span_non_stream_attributes.get(SpanAttributes.OUTPUT_VALUE) is not None

        # Check token usage metadata
        assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT in main_span_non_stream_attributes
        assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION in main_span_non_stream_attributes
        assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL in main_span_non_stream_attributes

        in_memory_span_exporter.clear()

        # Handle streaming (generator) run
        streaming_model = FakeModel(streaming=True)
        code_agent_stream = CodeAgent(
            tools=[],
            model=streaming_model,
            max_steps=3,
            additional_authorized_imports=["json"],
        )
        result_stream = code_agent_stream.run("Test question for streaming", stream=True)

        # Check that CodeAgent result is a generator
        assert hasattr(result_stream, "__iter__")
        assert hasattr(result_stream, "__next__")

        # Collect chunks for final output
        output_chunks = []
        for chunk in result_stream:
            output_chunks.append(str(chunk))
        final_output = "".join(output_chunks)
        assert "Test result from CodeAgent" in final_output

        stream_spans = in_memory_span_exporter.get_finished_spans()
        assert len(stream_spans) > 0

        agent_spans_stream = [
            span
            for span in stream_spans
            if span.attributes is not None
            and span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.AGENT.value
        ]
        assert len(agent_spans_stream) >= 1

        # Check attributes in main span
        main_span_stream = agent_spans_stream[0]
        main_span_stream_attributes = dict(main_span_stream.attributes or {})
        assert main_span_stream.name == "CodeAgent.run"
        assert main_span_stream_attributes.get(SpanAttributes.INPUT_VALUE) is not None
        assert main_span_stream_attributes.get("smolagents.max_steps") == 3

        output_value = main_span_stream_attributes.get(SpanAttributes.OUTPUT_VALUE)
        assert output_value is not None

        step_spans_stream = [
            span
            for span in stream_spans
            if span.attributes is not None
            and span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.CHAIN.value
        ]
        assert len(step_spans_stream) >= 1

        # Check attributes in step span
        step_span_stream = step_spans_stream[0]
        step_span_stream_attributes = dict(step_span_stream.attributes or {})
        assert step_span_stream.name == "Step 1"
        assert step_span_stream_attributes.get(SpanAttributes.INPUT_VALUE) is not None
        assert step_span_stream_attributes.get(SpanAttributes.OUTPUT_VALUE) is not None

        # Check token usage metadata
        assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT in main_span_stream_attributes
        assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION in main_span_stream_attributes
        assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL in main_span_stream_attributes

        # Check common attributes & status codes
        common_attributes = [
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            SpanAttributes.INPUT_VALUE,
            "smolagents.max_steps",
            "smolagents.tools_names",
        ]
        for attr in common_attributes:
            assert attr in main_span_non_stream_attributes
            assert attr in main_span_stream_attributes

        assert main_span_non_stream.status.status_code == trace_api.StatusCode.OK
        assert main_span_stream.status.status_code == trace_api.StatusCode.OK

    def test_multiagents(self) -> None:
        class FakeModelMultiagentsManagerAgent:
            def __call__(
                self,
                messages: list[dict[str, Any]],
                stop_sequences: Optional[list[str]] = None,
                grammar: Optional[Any] = None,
                tools_to_call_from: Optional[list[Tool]] = None,
            ) -> Any:
                if tools_to_call_from is not None:
                    if len(messages) < 3:
                        return ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatMessageToolCallFunction(
                                        name="search_agent",
                                        arguments="Who is the current US president?",
                                    ),
                                )
                            ],
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatMessageToolCallFunction(
                                        name="final_answer", arguments="Final report."
                                    ),
                                )
                            ],
                        )
                else:
                    if len(messages) < 3:
                        return ChatMessage(
                            role="assistant",
                            content="""
Thought: Let's call our search agent.
Code:
```py
result = search_agent("Who is the current US president?")
```<end_code>
""",
                        )
                    else:
                        assert "Report on the current US president" in str(messages)
                        return ChatMessage(
                            role="assistant",
                            content="""
Thought: Let's return the report.
Code:
```py
final_answer("Final report.")
```<end_code>
""",
                        )

            def generate(
                self,
                messages: list[dict[str, Any]],
                stop_sequences: Optional[list[str]] = None,
                grammar: Optional[Any] = None,
                tools_to_call_from: Optional[list[Tool]] = None,
            ) -> Any:
                return self(messages, stop_sequences, grammar, tools_to_call_from)

            def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
                if not message.tool_calls:
                    message.tool_calls = [
                        ChatMessageToolCall(
                            id="call_0",
                            type="function",
                            function=ChatMessageToolCallFunction(
                                name="final_answer", arguments="Final report."
                            ),
                        )
                    ]
                return message

        manager_model = FakeModelMultiagentsManagerAgent()

        class FakeModelMultiagentsManagedAgent:
            def __call__(
                self,
                messages: list[dict[str, Any]],
                stop_sequences: Optional[list[str]] = None,
                grammar: Optional[Any] = None,
                tools_to_call_from: Optional[list[Tool]] = None,
            ) -> Any:
                return ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ChatMessageToolCall(
                            id="call_0",
                            type="function",
                            function=ChatMessageToolCallFunction(
                                name="final_answer",
                                arguments="Report on the current US president",
                            ),
                        )
                    ],
                )

            def generate(
                self,
                messages: list[dict[str, Any]],
                stop_sequences: Optional[list[str]] = None,
                grammar: Optional[Any] = None,
                tools_to_call_from: Optional[list[Tool]] = None,
            ) -> Any:
                return self(messages, stop_sequences, grammar, tools_to_call_from)

        managed_model = FakeModelMultiagentsManagedAgent()

        web_agent = ToolCallingAgent(
            tools=[],
            model=managed_model,
            max_steps=10,
            name="search_agent",
            description=(
                "Runs web searches for you. Give it your request as an argument. "
                "Make the request as detailed as needed, you can ask for thorough reports"
            ),
        )

        manager_code_agent = CodeAgent(
            tools=[],
            model=manager_model,
            managed_agents=[web_agent],
        )

        report = manager_code_agent.run("Fake question.")
        assert report == "Final report."

        manager_toolcalling_agent = ToolCallingAgent(
            tools=[],
            model=manager_model,
            managed_agents=[web_agent],
        )

        report = manager_toolcalling_agent.run("Fake question.")
        assert report == "Final report."


class TestTools:
    def test_tool_invocation_returning_string_has_expected_attributes(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        class GetWeatherTool(Tool):  # type: ignore[misc]
            name = "get_weather"
            description = "Get the weather for a given city"
            inputs = {
                "location": {"type": "string", "description": "The city to get the weather for"}
            }
            output_type = "string"

            def forward(self, location: str) -> str:
                return "sunny"

        weather_tool = GetWeatherTool()

        result = weather_tool("Paris")
        assert result == "sunny"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        attributes = dict(span.attributes or {})

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {
            "args": ["Paris"],
            "sanitize_inputs_outputs": False,
            "kwargs": {},
        }
        assert attributes.pop(OUTPUT_VALUE) == "sunny"
        assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
        assert attributes.pop(TOOL_NAME) == "get_weather"
        assert attributes.pop(TOOL_DESCRIPTION) == "Get the weather for a given city"
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == {
            "location": {
                "type": "string",
                "description": "The city to get the weather for",
            },
        }
        assert not attributes

    def test_tool_invocation_returning_dict_has_expected_attributes(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        class GetWeatherTool(Tool):  # type: ignore[misc]
            name = "get_weather"
            description = "Get detailed weather information for a given city"
            inputs = {
                "location": {"type": "string", "description": "The city to get the weather for"}
            }
            output_type = "object"

            def forward(self, location: str) -> dict[str, Any]:
                return {"condition": "sunny", "temperature": 25, "humidity": 60}

        weather_tool = GetWeatherTool()

        result = weather_tool("Paris")
        assert result == {"condition": "sunny", "temperature": 25, "humidity": 60}

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        attributes = dict(span.attributes or {})

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {
            "args": ["Paris"],
            "sanitize_inputs_outputs": False,
            "kwargs": {},
        }
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert json.loads(output_value) == {"condition": "sunny", "temperature": 25, "humidity": 60}
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(TOOL_NAME) == "get_weather"
        assert (
            attributes.pop(TOOL_DESCRIPTION) == "Get detailed weather information for a given city"
        )
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == {
            "location": {
                "type": "string",
                "description": "The city to get the weather for",
            },
        }
        assert not attributes

    def test_tool_invocation_returning_tuple_has_expected_attributes(
        self, in_memory_span_exporter: InMemorySpanExporter
    ) -> None:
        @tool  # type: ignore[misc]
        def get_population(location: str) -> tuple[str, str]:
            """
            Get Population of the location and location type for the given location.

            Args:
                location: the location
            """
            return f"Population In {location} is 10 million", "City"

        result = get_population("Paris")
        assert result == ("Population In Paris is 10 million", "City")

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        attributes = dict(span.attributes or {})

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == TOOL
        assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
        assert json.loads(input_value) == {
            "args": ["Paris"],
            "sanitize_inputs_outputs": False,
            "kwargs": {},
        }
        assert isinstance(output_value := attributes.pop(OUTPUT_VALUE), str)
        assert json.loads(output_value) == ["Population In Paris is 10 million", "City"]
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert attributes.pop(TOOL_NAME) == "get_population"
        assert (
            attributes.pop(TOOL_DESCRIPTION) == "Get Population of the location and location "
            "type for the given location."
        )
        assert isinstance(tool_parameters := attributes.pop(TOOL_PARAMETERS), str)
        assert json.loads(tool_parameters) == {
            "location": {
                "type": "string",
                "description": "the location",
            },
        }
        assert not attributes


class TestInferLLMProviderFromClassName:
    def test_returns_none_when_instance_is_none(self) -> None:
        result = infer_llm_provider_from_class_name(None)
        assert result is None

    @pytest.mark.parametrize(
        "class_name, model_id, expected",
        [
            ("LiteLLMModel", "anthropic/claude-3-opus", OpenInferenceLLMProviderValues.ANTHROPIC),
            ("LiteLLMModel", "openai/gpt-4", OpenInferenceLLMProviderValues.OPENAI),
            ("LiteLLMModel", "azure/gpt-4", OpenInferenceLLMProviderValues.AZURE),
            ("LiteLLMModel", "cohere/command-r", OpenInferenceLLMProviderValues.COHERE),
            ("LiteLLMRouterModel", "anthropic/claude-3", OpenInferenceLLMProviderValues.ANTHROPIC),
            ("LiteLLMRouterModel", "openai/gpt-3.5", OpenInferenceLLMProviderValues.OPENAI),
        ],
    )
    def test_litellm_models_with_valid_provider_prefix(
        self, class_name: str, model_id: str, expected: OpenInferenceLLMProviderValues
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = class_name
        mock_instance.model_id = model_id

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result == expected

    @pytest.mark.parametrize(
        "class_name, model_id",
        [
            ("LiteLLMModel", "invalid_provider/some-model"),
            ("LiteLLMModel", "gpt-4"),
            ("LiteLLMRouterModel", "unknown/model"),
        ],
    )
    def test_litellm_models_with_invalid_model_id_returns_none(
        self, class_name: str, model_id: str
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = class_name
        mock_instance.model_id = model_id

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result is None

    @pytest.mark.parametrize(
        "model_id",
        [None, 12345, [], {}],
    )
    def test_litellm_model_with_invalid_model_id_type(self, model_id: Any) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = "LiteLLMModel"
        mock_instance.model_id = model_id

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result is None

    def test_litellm_model_with_missing_model_id_attribute(self) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = "LiteLLMModel"
        del mock_instance.model_id

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result is None

    @pytest.mark.parametrize(
        "class_name, expected",
        [
            ("OpenAIModel", OpenInferenceLLMProviderValues.OPENAI),
            ("AzureOpenAIModel", OpenInferenceLLMProviderValues.AZURE),
            ("AmazonBedrockModel", OpenInferenceLLMProviderValues.AWS),
        ],
    )
    def test_known_server_models_return_expected_provider(
        self, class_name: str, expected: OpenInferenceLLMProviderValues
    ) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = class_name

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result == expected

    @pytest.mark.parametrize(
        "class_name",
        ["InferenceClientModel", "UnknownModelClass", "CustomModel"],
    )
    def test_unknown_or_special_class_names_return_none(self, class_name: str) -> None:
        mock_instance = MagicMock()
        mock_instance.__class__.__name__ = class_name

        result = infer_llm_provider_from_class_name(mock_instance)
        assert result is None


# message attributes
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS

# mime types
JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

# span kinds
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
LLM = OpenInferenceSpanKindValues.LLM.value
TOOL = OpenInferenceSpanKindValues.TOOL.value

# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOOLS = SpanAttributes.LLM_TOOLS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS

# tool attributes
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA

# tool call attributes
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
