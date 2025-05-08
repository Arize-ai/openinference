from typing import Any

import boto3
import pytest
from botocore.eventstream import EventStream
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import MessageAttributes, SpanAttributes, ToolCallAttributes


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


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_tool_calls_with_input_params(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="ap-south-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    attributes = dict(
        inputText="Find the sum of 10 and 20",
        agentId="G0OUMYARBX",
        agentAliasId="TSTALIASID",
        sessionId="default_session_id",
        enableTrace=True,
    )
    response = client.invoke_agent(**attributes)
    assert isinstance(response["completion"], EventStream)
    events = [event for event in response["completion"]]
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(events) == 10
    assert len(spans) == 5
    span_names = [span.name for span in spans]
    assert span_names == [
        "LLM",
        "action_group",
        "LLM",
        "orchestrationTrace",
        "bedrock_agent.invoke_agent",
    ]
    llm_span = [span for span in spans if span.name == "LLM"][0]
    llm_span_attributes = dict(llm_span.attributes or {})
    tool_prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    tool_function_key = f"{tool_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    tool_inputs_key = f"{tool_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    assert (
        llm_span_attributes[tool_function_key] == "action_group_quick_start_7jzsu__add_two_numbers"
    )
    assert llm_span_attributes[tool_inputs_key] == '{"number_1": 10, "number_2": 20}'
    tool_result_span = [span for span in spans if span.name == "action_group"][0]
    tool_span_attributes = dict(tool_result_span.attributes or {})
    output_value = "The result of adding 10 and 20 is 30"
    assert tool_span_attributes[SpanAttributes.OUTPUT_VALUE] == output_value
    assert tool_span_attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "TOOL"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_tool_calls_without_input_params(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="ap-south-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    attributes = dict(
        inputText="What is the time?",
        agentId="G0OUMYARBX",
        agentAliasId="TSTALIASID",
        sessionId="default_session_id",
        enableTrace=True,
    )
    response = client.invoke_agent(**attributes)
    assert isinstance(response["completion"], EventStream)
    events = [event for event in response["completion"]]
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(events) == 10
    assert len(spans) == 5
    span_names = [span.name for span in spans]
    assert span_names == [
        "LLM",
        "action_group",
        "LLM",
        "orchestrationTrace",
        "bedrock_agent.invoke_agent",
    ]
    llm_span = [span for span in spans if span.name == "LLM"][0]
    llm_span_attributes = dict(llm_span.attributes or {})
    tool_prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    tool_function_key = f"{tool_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    tool_inputs_key = f"{tool_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    assert llm_span_attributes[tool_function_key] == "action_group_quick_start_7jzsu__get_time"
    assert llm_span_attributes[tool_inputs_key] == "{}"
    tool_result_span = [span for span in spans if span.name == "action_group"][0]
    tool_span_attributes = dict(tool_result_span.attributes or {})

    assert tool_span_attributes[SpanAttributes.OUTPUT_VALUE] == "The current time is 18:41:58"
    assert tool_span_attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "TOOL"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_knowledge_base_results(
    tracer_provider: trace_sdk.TracerProvider, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="ap-south-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    attributes = dict(
        inputText="What is task decomposition?",
        agentId="G0OUMYARBX",
        agentAliasId="TSTALIASID",
        sessionId="default_session_id",
        enableTrace=True,
    )
    response = client.invoke_agent(**attributes)
    assert isinstance(response["completion"], EventStream)
    events = [event for event in response["completion"]]
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(events) == 10
    span_names = [span.name for span in spans]
    assert span_names == [
        "LLM",
        "knowledge_base",
        "LLM",
        "orchestrationTrace",
        "bedrock_agent.invoke_agent",
    ]
    assert len(spans) == 5
    llm_span = [span for span in spans if span.name == "LLM"][0]
    llm_span_attributes = dict(llm_span.attributes or {})
    tool_prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    tool_function_key = f"{tool_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    tool_inputs_key = f"{tool_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    assert llm_span_attributes[tool_function_key] == "GET__x_amz_knowledgebase_HFUFBERTZV__Search"
    assert llm_span_attributes[tool_inputs_key] == (
        '{"searchQuery": "What is task decomposition? Provide a definition and explanation."}'
    )


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_preprocessing_trace(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="ap-south-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    attributes = dict(
        inputText="What is best time to visit the Taj Mahal?",
        agentId="1CF333B9DE",
        agentAliasId="9QFT0YAI71",
        sessionId="default_session_id2",
        enableTrace=True,
    )
    response = client.invoke_agent(**attributes)
    assert isinstance(response["completion"], EventStream)
    events = [event for event in response["completion"]]
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(events) == 7
    assert len(spans) == 5
    span_names = [span.name for span in spans]
    assert span_names == [
        "LLM",
        "LLM",
        "preProcessingTrace",
        "orchestrationTrace",
        "bedrock_agent.invoke_agent",
    ]


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_post_processing_trace(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="ap-south-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    attributes = dict(
        inputText="What is best time to visit the Taj Mahal?",
        agentId="1CF333B9DE",
        agentAliasId="PG0PYIUJ2F",
        sessionId="default_session_id2",
        enableTrace=True,
    )
    response = client.invoke_agent(**attributes)
    assert isinstance(response["completion"], EventStream)
    events = [event for event in response["completion"]]
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(events) == 7
    assert len(spans) == 5
    span_names = [span.name for span in spans]
    assert span_names == [
        "LLM",
        "LLM",
        "orchestrationTrace",
        "postProcessingTrace",
        "bedrock_agent.invoke_agent",
    ]


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_agent_call_without_traces(
    tracer_provider: trace_sdk.TracerProvider, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="ap-south-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    attributes = dict(
        inputText="What is task decomposition?",
        agentId="G0OUMYARBX",
        agentAliasId="TSTALIASID",
        sessionId="default_session_id",
        enableTrace=False,
    )
    response = client.invoke_agent(**attributes)
    assert isinstance(response["completion"], EventStream)
    events = [event for event in response["completion"]]
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(events) == 1
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "AGENT"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "aws"
    assert isinstance(attributes.pop(SpanAttributes.INPUT_VALUE), str)
    assert isinstance(attributes.pop(SpanAttributes.OUTPUT_VALUE), str)
    assert spans[0].name == "bedrock_agent.invoke_agent"
