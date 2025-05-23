from collections import Counter
from typing import Any

import boto3
import pytest
from botocore.eventstream import EventStream
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def starts_with(left_value: Any, right_value: str) -> bool:
    return str(left_value).startswith(right_value)


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
        agentId="FQBGXINMYT",
        agentAliasId="PPY9UTNXJL",
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
    llm_span = [span for span in spans if span.name == "LLM"][-1]
    llm_span_attributes = dict(llm_span.attributes or {})
    assert llm_span_attributes.pop("input.mime_type") == "text/plain"
    assert starts_with(llm_span_attributes.pop("input.value"), '{"system":" You are agent to ans')

    assert starts_with(
        llm_span_attributes.pop("llm.input_messages.0.message.content"),
        " You are agent to answer customer",
    )
    assert llm_span_attributes.pop("llm.input_messages.0.message.role") == "system"

    assert (
        llm_span_attributes.pop("llm.input_messages.1.message.content")
        == "Find the sum of 10 and 20"
    )
    assert llm_span_attributes.pop("llm.input_messages.1.message.role") == "user"

    assert starts_with(
        llm_span_attributes.pop("llm.input_messages.2.message.content"),
        "[{text=<thinking>To find the sum of 10",
    )
    assert llm_span_attributes.pop("llm.input_messages.2.message.role") == "assistant"

    assert starts_with(
        llm_span_attributes.pop("llm.input_messages.3.message.content"),
        "[{tool_use_id=toolu_bdrk_01W8JC7eXo51cwZDKQgajC2F, type=tool_result",
    )
    assert llm_span_attributes.pop("llm.input_messages.3.message.role") == "user"

    assert starts_with(
        llm_span_attributes.pop("llm.invocation_parameters"), '{"maximumLength": 2048'
    )
    assert llm_span_attributes.pop("llm.model_name") == "claude-3-5-sonnet-20240620"
    assert llm_span_attributes.pop("llm.provider") == "aws"
    assert starts_with(
        llm_span_attributes.pop("llm.output_messages.0.message.content"),
        "<thinking>The function has returned the result of adding 10 and 20, which is 30",
    )
    assert llm_span_attributes.pop("llm.output_messages.0.message.role") == "assistant"
    assert llm_span_attributes.pop("llm.token_count.completion") == 56
    assert llm_span_attributes.pop("llm.token_count.prompt") == 915
    assert llm_span_attributes.pop("llm.token_count.total") == 971

    assert starts_with(llm_span_attributes.pop("metadata"), '{"client_request_id": "edfe2c05-72')
    assert llm_span_attributes.pop("openinference.span.kind") == "LLM"
    assert llm_span_attributes.pop("output.mime_type") == "text/plain"
    assert starts_with(
        llm_span_attributes.pop("output.value"), "The function has returned the result"
    )
    assert not llm_span_attributes
    action_group_span = [span for span in spans if span.name == "action_group"][-1]
    action_group_span_attributes = dict(action_group_span.attributes or {})

    assert action_group_span_attributes.pop("llm.input_messages.0.message.role") == "tool"
    assert (
        action_group_span_attributes.pop("llm.input_messages.0.message.tool_call_id") == "default"
    )
    assert (
        action_group_span_attributes.pop(
            "llm.input_messages.0.message.tool_calls.0.tool_call.function.name"
        )
        == "add_two_numbers"
    )
    assert (
        action_group_span_attributes.pop("llm.input_messages.0.message.tool_calls.0.tool_call.id")
        == "default"
    )

    assert starts_with(
        action_group_span_attributes.pop("metadata"), '{"client_request_id": "73759b26'
    )
    assert action_group_span_attributes.pop("openinference.span.kind") == "TOOL"
    assert action_group_span_attributes.pop("output.mime_type") == "text/plain"
    assert (
        action_group_span_attributes.pop("output.value") == "The result of adding 10 and 20 is 30"
    )
    assert action_group_span_attributes.pop("tool.description") == ""
    assert action_group_span_attributes.pop("tool.name") == "add_two_numbers"
    assert starts_with(
        action_group_span_attributes.pop("tool.parameters"), '[{"name": "n1", "type": "numb'
    )
    assert not action_group_span_attributes


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
        agentId="FQBGXINMYT",
        agentAliasId="PPY9UTNXJL",
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
    assert llm_span_attributes.pop("input.mime_type") == "text/plain"
    assert starts_with(llm_span_attributes.pop("input.value"), '{"system":" You are agent to a')
    assert starts_with(
        llm_span_attributes.pop("llm.input_messages.0.message.content"),
        " You are agent to answer cus",
    )
    assert llm_span_attributes.pop("llm.input_messages.0.message.role") == "system"
    assert (
        llm_span_attributes.pop("llm.input_messages.1.message.content")
        == "Find the sum of 10 and 20"
    )
    assert llm_span_attributes.pop("llm.input_messages.1.message.role") == "user"
    assert starts_with(
        llm_span_attributes.pop("llm.input_messages.2.message.content"),
        "[{text=<thinking>To find the ",
    )
    assert llm_span_attributes.pop("llm.input_messages.2.message.role") == "assistant"
    assert starts_with(
        llm_span_attributes.pop("llm.input_messages.3.message.content"),
        "[{tool_use_id=toolu_bdrk_01W",
    )
    assert llm_span_attributes.pop("llm.input_messages.3.message.role") == "user"
    assert starts_with(
        llm_span_attributes.pop("llm.input_messages.4.message.content"),
        "[{text=<thinking>The functio",
    )
    assert llm_span_attributes.pop("llm.input_messages.4.message.role") == "assistant"
    assert llm_span_attributes.pop("llm.input_messages.5.message.content") == "What is the time?"
    assert llm_span_attributes.pop("llm.input_messages.5.message.role") == "user"
    assert starts_with(
        llm_span_attributes.pop("llm.invocation_parameters"), '{"maximumLength": 2048, "stopSe'
    )
    assert llm_span_attributes.pop("llm.model_name") == "claude-3-5-sonnet-20240620"
    assert llm_span_attributes.pop("llm.provider") == "aws"
    assert starts_with(
        llm_span_attributes.pop("llm.output_messages.0.message.content"),
        "<thinking>To answer the que",
    )
    assert llm_span_attributes.pop("llm.output_messages.0.message.role") == "assistant"
    assert llm_span_attributes.pop("llm.output_messages.1.message.role") == "tool"
    assert (
        llm_span_attributes.pop("llm.output_messages.1.message.tool_call_id")
        == "toolu_bdrk_015fnHoQg33WC8XhG4TTXcZ2"
    )
    tool_prefix = "llm.output_messages.1.message.tool_calls.0.tool_call"
    assert llm_span_attributes.pop(f"{tool_prefix}.function.arguments") == "{}"
    assert (
        llm_span_attributes.pop(f"{tool_prefix}.function.name")
        == "action_group_quick_start_6gq19__get_time"
    )
    assert llm_span_attributes.pop(f"{tool_prefix}.id") == "toolu_bdrk_015fnHoQg33WC8XhG4TTXcZ2"
    assert llm_span_attributes.pop("llm.token_count.completion") == 103
    assert llm_span_attributes.pop("llm.token_count.prompt") == 980
    assert llm_span_attributes.pop("llm.token_count.total") == 1083
    assert starts_with(llm_span_attributes.pop("metadata"), '{"client_request_id": "603b')
    assert llm_span_attributes.pop("openinference.span.kind") == "LLM"
    assert llm_span_attributes.pop("output.mime_type") == "text/plain"
    assert starts_with(llm_span_attributes.pop("output.value"), "To answer the question about")
    assert not llm_span_attributes
    action_group_span = [span for span in spans if span.name == "action_group"][0]
    action_group_span_attributes = dict(action_group_span.attributes or {})
    assert action_group_span_attributes.pop("llm.input_messages.0.message.role") == "tool"
    assert (
        action_group_span_attributes.pop("llm.input_messages.0.message.tool_call_id") == "default"
    )
    tool_prefix = "llm.input_messages.0.message.tool_calls"
    assert action_group_span_attributes.pop(f"{tool_prefix}.0.tool_call.function.arguments") == "{}"
    assert (
        action_group_span_attributes.pop(f"{tool_prefix}.0.tool_call.function.name") == "get_time"
    )
    assert action_group_span_attributes.pop(f"{tool_prefix}.0.tool_call.id") == "default"
    assert starts_with(
        action_group_span_attributes.pop("metadata"), '{"client_request_id": "fd79a6'
    )
    assert action_group_span_attributes.pop("openinference.span.kind") == "TOOL"
    assert action_group_span_attributes.pop("output.mime_type") == "text/plain"
    assert starts_with(
        action_group_span_attributes.pop("output.value"), "The current time is 09:52"
    )
    assert action_group_span_attributes.pop("tool.description") == ""
    assert action_group_span_attributes.pop("tool.name") == "get_time"
    assert action_group_span_attributes.pop("tool.parameters") == "[]"
    assert not action_group_span_attributes


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
        agentAliasId="YYTTVM7BYE",
        sessionId="default_session_id",
        enableTrace=True,
    )
    response = client.invoke_agent(**attributes)
    assert isinstance(response["completion"], EventStream)
    events = [event for event in response["completion"]]
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(events) == 6
    span_names = [span.name for span in spans]
    assert span_names == [
        "knowledge_base",
        "LLM",
        "orchestrationTrace",
        "bedrock_agent.invoke_agent",
    ]
    assert len(spans) == 4
    llm_span = [span for span in spans if span.name == "LLM"][-1]
    llm_span_attributes = dict(llm_span.attributes or {})
    assert llm_span_attributes.pop("llm.invocation_parameters") == (
        '{"maximumLength": 2048, "stopSequences": '
        '["\\n\\nHuman:"], "temperature": 0.0, "topK": 250,'
        ' "topP": 1.0}'
    )
    assert llm_span_attributes.pop("llm.input_messages.0.message.role") == "assistant"
    content = "You are a question answering agent."
    assert starts_with(llm_span_attributes.pop("llm.input_messages.0.message.content"), content)
    assert starts_with(llm_span_attributes.pop("input.value"), content)
    assert llm_span_attributes.pop("input.mime_type") == "text/plain"
    assert llm_span_attributes.pop("llm.model_name") == "claude-3-5-sonnet-20240620"
    assert llm_span_attributes.pop("llm.provider") == "aws"
    assert llm_span_attributes.pop("llm.output_messages.0.message.role") == "assistant"
    assert starts_with(
        llm_span_attributes.pop("llm.output_messages.0.message.content"), "<answer>\n<answer_part>"
    )
    assert llm_span_attributes.pop("llm.token_count.prompt") == 2068
    assert llm_span_attributes.pop("llm.token_count.completion") == 385
    assert llm_span_attributes.pop("llm.token_count.total") == 2453
    assert starts_with(llm_span_attributes.pop("output.value"), '{"stop_sequence":null,"usage"')
    assert llm_span_attributes.pop("output.mime_type") == "text/plain"
    assert llm_span_attributes.pop("metadata") is not None
    assert llm_span_attributes.pop("openinference.span.kind") == "LLM"
    assert not llm_span_attributes
    knowledge_base_span = [span for span in spans if span.name == "knowledge_base"][-1]
    data = dict(knowledge_base_span.attributes or {})
    assert starts_with(data.pop("input.mime_type"), "text/plain")
    assert starts_with(data.pop("input.value"), "What is task decompositi")
    assert starts_with(data.pop("metadata"), '{"client_request_id": "7fc9')
    assert starts_with(data.pop("openinference.span.kind"), "RETRIEVER")
    assert starts_with(
        data.pop("retrieval.documents.0.document.content"), "Fig. 1. Overview of a LLM-pow"
    )
    assert starts_with(
        data.pop("retrieval.documents.0.document.metadata"), '{"location": {"customDocume'
    )
    assert starts_with(
        data.pop("retrieval.documents.1.document.content"), "They use few-shot examples to"
    )
    assert starts_with(
        data.pop("retrieval.documents.1.document.metadata"), '{"location": {"customDocume'
    )
    assert starts_with(
        data.pop("retrieval.documents.2.document.content"), "The training dataset in their"
    )
    assert starts_with(
        data.pop("retrieval.documents.2.document.metadata"), '{"location": {"customDocume'
    )
    assert starts_with(
        data.pop("retrieval.documents.3.document.content"), "The code should be fully func"
    )
    assert starts_with(
        data.pop("retrieval.documents.3.document.metadata"), '{"location": {"customDocume'
    )
    assert starts_with(
        data.pop("retrieval.documents.4.document.content"), "LLM is presented with a list "
    )
    assert starts_with(
        data.pop("retrieval.documents.4.document.metadata"), '{"location": {"customDocume'
    )
    assert not data


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
        agentId="DJJ1HRFGOM",
        agentAliasId="EXL0UP4ON9",
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
        "preProcessingTrace",
        "LLM",
        "orchestrationTrace",
        "bedrock_agent.invoke_agent",
    ]
    preprocessing_span = [span for span in spans if span.name == "preProcessingTrace"][-1]
    preprocessing_span_attributes = dict(preprocessing_span.attributes or {})

    assert preprocessing_span_attributes.pop("input.mime_type") == "text/plain"
    assert starts_with(preprocessing_span_attributes.pop("input.value"), "[{text=What is best time")
    assert preprocessing_span_attributes.pop("openinference.span.kind") == "CHAIN"
    assert preprocessing_span_attributes.pop("output.mime_type") == "text/plain"
    assert "This input is a straightforward" in str(
        preprocessing_span_attributes.pop("output.value")
    )
    assert not preprocessing_span_attributes

    initial_span = [span for span in spans if span.name == "bedrock_agent.invoke_agent"][-1]
    initial_span_attributes = dict(initial_span.attributes or {})
    assert initial_span_attributes.pop("agentAliasId") == "EXL0UP4ON9"
    assert initial_span_attributes.pop("agentId") == "DJJ1HRFGOM"
    assert initial_span_attributes.pop("enableTrace") is True
    assert initial_span_attributes.pop("input.value") == "What is best time to visit the Taj Mahal?"
    assert initial_span_attributes.pop("inputText") == "What is best time to visit the Taj Mahal?"
    assert initial_span_attributes.pop("llm.provider") == "aws"
    assert initial_span_attributes.pop("openinference.span.kind") == "AGENT"
    assert initial_span_attributes.pop("output.mime_type") == "text/plain"
    assert starts_with(
        initial_span_attributes.pop("output.value"), "The best time to visit the Taj Ma"
    )
    assert initial_span_attributes.pop("sessionId") == "default_session_id2"
    assert not initial_span_attributes


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
        agentId="3EL4X42BSO",
        agentAliasId="LSIZXQMZDN",
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
        "orchestrationTrace",
        "LLM",
        "postProcessingTrace",
        "bedrock_agent.invoke_agent",
    ]
    postprocessing_span = [span for span in spans if span.name == "postProcessingTrace"][-1]
    postprocessing_span_attributes = dict(postprocessing_span.attributes or {})
    assert postprocessing_span_attributes.pop("openinference.span.kind") == "CHAIN"
    assert postprocessing_span_attributes.pop("output.mime_type") == "text/plain"
    assert starts_with(
        postprocessing_span_attributes.pop("output.value"),
        "Based on the information I've gathered about the Taj Mahal",
    )
    assert not postprocessing_span_attributes


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
        region_name="us-east-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    attributes = dict(
        inputText="What is task decomposition?",
        agentId="XNW1LGJJZT",
        agentAliasId="K0P4LV9GPO",
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
    assert attributes.pop("agentAliasId") == "K0P4LV9GPO"
    assert attributes.pop("agentId") == "XNW1LGJJZT"
    assert attributes.pop("enableTrace") is False
    assert attributes.pop("input.value") == "What is task decomposition?"
    assert attributes.pop("inputText") == "What is task decomposition?"
    assert attributes.pop("llm.provider") == "aws"
    assert attributes.pop("openinference.span.kind") == "AGENT"
    assert attributes.pop("output.mime_type") == "text/plain"
    assert (
        attributes.pop("output.value") == "Sorry, I don't have enough information to answer that."
    )
    assert attributes.pop("sessionId") == "default_session_id"
    assert spans[0].name == "bedrock_agent.invoke_agent"
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_multi_agent_collaborator(
    tracer_provider: trace_sdk.TracerProvider, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    agent_id = "2X9SRVPLWB"
    agent_alias_id = "KUXISKYLTT"
    session_id = "12345680"
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        aws_access_key_id="123",
        aws_secret_access_key="321",
    )
    attributes = dict(
        inputText="Find the sum of 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10.",
        agentId=agent_id,
        agentAliasId=agent_alias_id,
        sessionId=session_id,
        enableTrace=True,
    )
    response = client.invoke_agent(**attributes)
    assert isinstance(response["completion"], EventStream)
    events = [event for event in response["completion"]]
    assert len(events) == 34
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 15
    span_name_counter = Counter([span.name for span in spans])
    assert span_name_counter["orchestrationTrace"] == 3
    assert span_name_counter["LLM"] == 9
    # assert span_name_counter["agent_collaborator[MathSolverAgent]"] == 1
    # assert span_name_counter["agent_collaborator[SimpleSupervisor]"] == 1
    assert span_name_counter["bedrock_agent.invoke_agent"] == 1
    supervisor_agent_span = [
        span for span in spans if span.name == "agent_collaborator[SimpleSupervisor]"
    ][-1]
    simple_supervisor_attributes = dict(supervisor_agent_span.attributes or {})
    assert simple_supervisor_attributes.pop("input.mime_type") == "text/plain"
    assert (
        simple_supervisor_attributes.pop("input.value")
        == "Please calculate the sum of the numbers 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10."
    )
    assert (
        simple_supervisor_attributes.pop("llm.input_messages.0.message.content")
        == "Please calculate the sum of the numbers 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10."
    )
    assert simple_supervisor_attributes.pop("llm.input_messages.0.message.role") == "assistant"
    assert (
        simple_supervisor_attributes.pop("llm.output_messages.0.message.content")
        == "The sum of the numbers 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10 is 55."
    )
    assert simple_supervisor_attributes.pop("llm.output_messages.0.message.role") == "assistant"
    assert starts_with(
        simple_supervisor_attributes.pop("metadata"), '{"client_request_id": "0a6ddb'
    )
    assert simple_supervisor_attributes.pop("openinference.span.kind") == "AGENT"
    assert simple_supervisor_attributes.pop("output.mime_type") == "text/plain"
    assert (
        simple_supervisor_attributes.pop("output.value")
        == "The sum of the numbers 1, 2, 3, 4, 5, 6, 7, 8, 9, and 10 is 55."
    )
    assert not simple_supervisor_attributes

    math_agent_span = [
        span for span in spans if span.name == "agent_collaborator[MathSolverAgent]"
    ][-1]
    math_agent_span_attributes = dict(math_agent_span.attributes or {})
    assert math_agent_span_attributes.pop("input.mime_type") == "text/plain"
    assert (
        math_agent_span_attributes.pop("input.value")
        == "Calculate the sum of 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10"
    )
    assert (
        math_agent_span_attributes.pop("llm.input_messages.0.message.content")
        == "Calculate the sum of 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10"
    )
    assert math_agent_span_attributes.pop("llm.input_messages.0.message.role") == "assistant"
    assert (
        math_agent_span_attributes.pop("llm.output_messages.0.message.content")
        == "The sum of 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 is 55."
    )
    assert math_agent_span_attributes.pop("llm.output_messages.0.message.role") == "assistant"
    assert starts_with(
        math_agent_span_attributes.pop("metadata"),
        '{"client_request_id": "5e3443ad-23b1-4b06-a073-b805ed323336", "end_time": 1.74782',
    )
    assert math_agent_span_attributes.pop("openinference.span.kind") == "AGENT"
    assert math_agent_span_attributes.pop("output.mime_type") == "text/plain"
    assert (
        math_agent_span_attributes.pop("output.value")
        == "The sum of 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 is 55."
    )
    assert not math_agent_span_attributes
