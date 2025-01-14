import json
import os
from typing import Any, Generator

import pytest
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from smolagents import OpenAIServerModel
from smolagents.tools import Tool
from smolagents.agents import CodeAgent, ManagedAgent, ToolCallingAgent
from smoalgents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallDefinition

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
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


def remove_all_vcr_response_headers(response: dict[str, Any]) -> dict[str, Any]:
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


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    yield
    SmolagentsInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.fixture
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    api_key = "sk-0123456789"
    monkeypatch.setenv("OPENAI_API_KEY", api_key)
    return api_key


class TestModels:
    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_openai_server_model_has_expected_attributes(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        model = OpenAIServerModel(
            model_id="gpt-4o",
            api_key=os.environ["OPENAI_API_KEY"],
            api_base="https://api.openai.com/v1",
        )
        input_message_content = (
            "Who won the World Cup in 2018? Answer in one word with no punctuation."
        )
        output_message = model(
            messages=[
                {
                    "role": "user",
                    "content": input_message_content,
                }
            ]
        )
        output_message_content = output_message.content
        assert output_message_content == "France"

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "OpenAIServerModel.__call__"
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
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {}
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message_content
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert (
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == output_message_content
        )
        assert not attributes

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_openai_server_model_with_tool_has_expected_attributes(
        self,
        openai_api_key: str,
        in_memory_span_exporter: InMemorySpanExporter,
        tracer_provider: trace_api.TracerProvider,
    ) -> None:
        model = OpenAIServerModel(
            model_id="gpt-4o",
            api_key=os.environ["OPENAI_API_KEY"],
            api_base="https://api.openai.com/v1",
        )
        input_message_content = "What is the weather in Paris?"

        class GetWeatherTool(Tool):
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
                    "content": input_message_content,
                }
            ],
            tools_to_call_from=[GetWeatherTool()],
        )
        output_message_content = output_message.content
        assert output_message_content is None
        tool_calls = output_message.tool_calls
        assert len(tool_calls) == 1
        assert isinstance(tool_call := tool_calls[0], ChatCompletionMessageToolCall)
        assert tool_call.function.name == "get_weather"
        assert isinstance(tool_call_arguments := tool_call.function.arguments, str)
        assert json.loads(tool_call_arguments) == {"location": "Paris"}

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "OpenAIServerModel.__call__"
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

class TestRun:
    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_multiagents(self):
        class FakeModelMultiagentsManagerAgent:
            def __call__(
                self,
                messages,
                stop_sequences=None,
                grammar=None,
                tools_to_call_from=None,
            ):
                if tools_to_call_from is not None:
                    if len(messages) < 3:
                        return ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=[
                                ChatMessageToolCall(
                                    id="call_0",
                                    type="function",
                                    function=ChatMessageToolCallDefinition(
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
                                    function=ChatMessageToolCallDefinition(
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

        manager_model = FakeModelMultiagentsManagerAgent()

        class FakeModelMultiagentsManagedAgent:
            def __call__(
                self,
                messages,
                tools_to_call_from=None,
                stop_sequences=None,
                grammar=None,
            ):
                return ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ChatMessageToolCall(
                            id="call_0",
                            type="function",
                            function=ChatMessageToolCallDefinition(
                                name="final_answer",
                                arguments="Report on the current US president",
                            ),
                        )
                    ],
                )

        managed_model = FakeModelMultiagentsManagedAgent()

        web_agent = ToolCallingAgent(
            tools=[],
            model=managed_model,
            max_steps=10,
        )

        managed_web_agent = ManagedAgent(
            agent=web_agent,
            name="search_agent",
            description="Runs web searches for you. Give it your request as an argument. Make the request as detailed as needed, you can ask for thorough reports",
        )

        manager_code_agent = CodeAgent(
            tools=[],
            model=manager_model,
            managed_agents=[managed_web_agent],
            additional_authorized_imports=["time", "numpy", "pandas"],
        )

        report = manager_code_agent.run("Fake question.")
        assert report == "Final report."

        manager_toolcalling_agent = ToolCallingAgent(
            tools=[],
            model=manager_model,
            managed_agents=[managed_web_agent],
        )

        report = manager_toolcalling_agent.run("Fake question.")
        assert report == "Final report."

# message attributes
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
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
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOOLS = SpanAttributes.LLM_TOOLS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE

# tool attributes
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA

# tool call attributes
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
