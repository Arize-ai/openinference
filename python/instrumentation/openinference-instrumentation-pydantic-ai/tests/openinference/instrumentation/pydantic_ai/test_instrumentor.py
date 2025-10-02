import json
import os
from datetime import datetime
from typing import List, Mapping, Sequence, Union, cast

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel

# Import necessary Pydantic AI components
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_openai_agent_and_llm_spans_v1(
    in_memory_span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
) -> None:
    # Version 1 is deprecated and will be removed in a future release.
    _test_openai_agent_and_llm_spans(
        in_memory_span_exporter,
        tracer_provider,
        InstrumentationSettings(version=1, event_mode="attributes"),
    )
    in_memory_span_exporter.clear()
    _test_openai_agent_and_llm_spans_message_history(
        in_memory_span_exporter,
        tracer_provider,
        InstrumentationSettings(version=1, event_mode="attributes"),
    )


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_openai_agent_and_llm_spans_v2(
    in_memory_span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
) -> None:
    # Version 2 stores messages as gen_ai.input.messages and gen_ai.output.messages
    # which our instrumentation converts to OpenInference attributes
    _test_openai_agent_and_llm_spans(
        in_memory_span_exporter, tracer_provider, InstrumentationSettings(version=2)
    )
    in_memory_span_exporter.clear()
    _test_openai_agent_and_llm_spans_message_history(
        in_memory_span_exporter, tracer_provider, InstrumentationSettings(version=2)
    )


def _test_openai_agent_and_llm_spans(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    instrumentation: InstrumentationSettings,
) -> None:
    """Test that Pydantic AI agent.run_sync() creates proper LLM and AGENT spans."""

    trace.set_tracer_provider(tracer_provider)

    class LocationModel(BaseModel):
        city: str
        country: str

    # API key from environment - only used when re-recording the cassette
    # When using the cassette, the key is not needed
    api_key = os.getenv("OPENAI_API_KEY", "sk-test")

    # Create the model and agent
    model = OpenAIModel("gpt-4o", provider=OpenAIProvider(api_key=api_key))
    agent = Agent(model, output_type=LocationModel, instrument=instrumentation)

    # Run the agent
    result = agent.run_sync("The windy city in the US of A.")

    # Verify we got a result
    assert result is not None

    # Get the spans - pydantic_ai creates 2 spans: LLM and AGENT
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2, f"Expected 2 spans (LLM and AGENT), got {len(spans)}"

    llm_span = get_span_by_kind(spans, OpenInferenceSpanKindValues.LLM.value)
    agent_span = get_span_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)

    _verify_llm_span(llm_span)

    _verify_agent_span(agent_span)


def _verify_llm_span(span: ReadableSpan) -> None:
    """Verify the LLM span has correct attributes."""
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    assert (
        attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.get(SpanAttributes.LLM_SYSTEM) == "openai"
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o"

    assert (
        attributes.get(f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}")
        == "user"
    )
    assert (
        attributes.get(f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}")
        == "The windy city in the US of A."
    )

    assert (
        attributes.get(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}")
        == "assistant"
    )

    tool_call_name = attributes.get(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{MessageAttributes.MESSAGE_FUNCTION_CALL_NAME}"
    )
    assert tool_call_name == "final_result"

    tool_call_arguments = attributes.get(
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON}"
    )
    assert isinstance(tool_call_arguments, str)
    arguments_dict = json.loads(tool_call_arguments)
    assert arguments_dict == {
        "city": "Chicago",
        "country": "United States of America",
    }

    assert (
        attributes.get(f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_NAME}") == "final_result"
    )
    assert (
        attributes.get(f"{SpanAttributes.LLM_TOOLS}.0.{SpanAttributes.TOOL_DESCRIPTION}")
        == "The final response which ends this conversation"
    )

    prompt_tokens = attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT)
    completion_tokens = attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION)
    total_tokens = attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL)
    assert isinstance(prompt_tokens, int)
    assert isinstance(completion_tokens, int)
    assert isinstance(total_tokens, int)
    assert total_tokens == prompt_tokens + completion_tokens


def _verify_agent_span(span: ReadableSpan) -> None:
    """Verify the AGENT span has correct attributes."""
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    assert (
        attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.AGENT.value
    )

    input_value = attributes.get(SpanAttributes.INPUT_VALUE)
    assert input_value == "The windy city in the US of A."

    output_value = attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert json.loads(output_value) == {
        "city": "Chicago",
        "country": "United States of America",
    }


def get_span_by_kind(spans: Sequence[ReadableSpan], kind: str) -> ReadableSpan:
    matching_spans = [
        span
        for span in spans
        if span.attributes and span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == kind
    ]
    if len(matching_spans) != 1:
        pytest.fail(f"Expected exactly one span of kind '{kind}', but found {len(matching_spans)}")
    return matching_spans[0]


def _test_openai_agent_and_llm_spans_message_history(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    instrumentation: InstrumentationSettings,
) -> None:
    """Test that INPUT_VALUE uses the last message from history, not the first."""

    trace.set_tracer_provider(tracer_provider)

    class LocationModel(BaseModel):
        city: str
        country: str

    # Create the model and agent
    model = OpenAIModel("gpt-4o", provider=OpenAIProvider(api_key="sk-test"))
    agent = Agent(model, output_type=LocationModel, instrument=instrumentation)

    # Create message history with multiple messages
    message_history: List[Union[ModelRequest, ModelResponse]] = [
        ModelRequest(
            parts=[UserPromptPart(content="first message", timestamp=datetime.now())],
            kind="request",
        ),
        ModelResponse(
            parts=[TextPart(content="first response")], timestamp=datetime.now(), kind="response"
        ),
        ModelRequest(
            parts=[UserPromptPart(content="second message", timestamp=datetime.now())],
            kind="request",
        ),
        ModelResponse(
            parts=[TextPart(content="second response")], timestamp=datetime.now(), kind="response"
        ),
    ]

    # Run the agent with message history
    result = agent.run_sync("third message", message_history=message_history)

    # Verify we got a result
    assert result is not None

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2, f"Expected 2 spans (LLM and AGENT), got {len(spans)}"

    agent_span = get_span_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)

    # Verify the INPUT_VALUE is the last message ("third message"), not the first ("first message")
    attributes = dict(cast(Mapping[str, AttributeValue], agent_span.attributes))
    input_value = attributes.get(SpanAttributes.INPUT_VALUE)

    assert input_value == "third message", (
        f"Expected INPUT_VALUE to be 'third message', but got '{input_value}'"
    )
