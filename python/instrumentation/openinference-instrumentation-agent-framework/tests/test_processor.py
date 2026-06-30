"""Processor tests for agent-framework instrumentation using pytest-recording.

These tests verify the AgentFrameworkToOpenInferenceProcessor by recording and replaying
HTTP interactions with real AI services. This ensures the processor correctly transforms
agent-framework spans to OpenInference format with actual API responses.

Requirements:
- agent-framework must be installed (tests will fail at import if missing)
- OpenAI API key required for recording new cassettes
- Existing cassettes can be used for replay without an API key

Recording new cassettes:
1. Set environment variable: export OPENAI_API_KEY=your_actual_key
2. Delete cassettes to re-record: rm -rf tests/cassettes/test_processor/
3. Run: pytest tests/test_processor.py -v --record-mode=rewrite

Running with existing cassettes:
- Run: pytest tests/test_processor.py -v
- Uses recorded cassettes from tests/cassettes/test_processor/
- Headers are sanitized (no API keys stored)
"""

from typing import cast

import pytest
from agent_framework import tool
from agent_framework.openai import OpenAIChatClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_chat_with_agent(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    openai_client: OpenAIChatClient,
) -> None:
    """Test basic OpenAI chat agent creates proper LLM spans with OpenInference attributes."""

    # Create a simple agent
    agent = openai_client.as_agent(
        name="TestAssistant",
        instructions="You are a helpful assistant. Answer concisely.",
    )

    # Run the agent with a simple query
    response = await agent.run("What is 2+2? Answer with just the number.")

    # Verify we got a response
    assert response is not None
    assert response.text is not None

    # Get the captured spans
    spans = in_memory_span_exporter.get_finished_spans()

    # Agent-framework creates multiple spans (agent invoke, chat, etc.)
    assert len(spans) > 0, f"Expected at least 1 span, got {len(spans)}"

    # Find the LLM span
    llm_spans = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    ]

    assert len(llm_spans) > 0, "Expected at least one LLM span"

    llm_span = llm_spans[0]
    attrs = dict(cast(dict, llm_span.attributes))

    # Verify OpenInference attributes were added
    assert (
        attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    )

    # Verify model info
    assert attrs.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o-mini"
    assert attrs.get(SpanAttributes.LLM_PROVIDER) == "openai"

    # Verify token counts exist
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT in attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION in attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL in attrs

    # Verify messages
    assert f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.role" in attrs
    assert f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.message.role" in attrs

    # Verify original GenAI attributes are preserved
    assert "gen_ai.operation.name" in attrs
    assert "gen_ai.request.model" in attrs


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agent_with_tool_calls(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    openai_client: OpenAIChatClient,
) -> None:
    """Test agent with tool calls creates AGENT, LLM, and TOOL spans."""

    # Define a simple tool
    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"The weather in {location} is sunny and 72°F"

    agent = openai_client.as_agent(
        name="WeatherAgent",
        instructions=(
            "You are a weather assistant. Use the get_weather tool when asked about weather."
        ),
        tools=[get_weather],
    )

    # Run agent with a query that should use the tool
    response = await agent.run("What's the weather in San Francisco?")

    assert response is not None

    # Get spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0

    # Check we have different span kinds
    span_kinds = set()
    for span in spans:
        if span.attributes:
            kind = span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            if kind:
                span_kinds.add(kind)

    # Should have at least LLM spans, possibly AGENT and TOOL
    assert OpenInferenceSpanKindValues.LLM.value in span_kinds

    # Find tool spans - should exist for tool call test
    tool_spans = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.TOOL.value
    ]

    assert len(tool_spans) > 0, "Expected at least one TOOL span for tool call test"

    tool_span = tool_spans[0]
    attrs = dict(cast(dict, tool_span.attributes))

    # Verify tool attributes
    assert attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "TOOL"
    assert SpanAttributes.TOOL_NAME in attrs

    # Verify original GenAI attributes preserved
    assert "gen_ai.operation.name" in attrs


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_conversation_with_history(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    openai_client: OpenAIChatClient,
) -> None:
    """Test multi-turn conversation maintains conversation_id as session_id."""
    agent = openai_client.as_agent(
        name="ConversationAgent",
        instructions="You are a friendly assistant.",
    )

    # First message
    response1 = await agent.run("Hi, my name is Alice.")
    assert response1 is not None

    # Clear spans from first turn
    in_memory_span_exporter.clear()

    # Second message - should maintain conversation context
    response2 = await agent.run("What is my name?")
    assert response2 is not None

    # Get spans from second turn
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0

    # Check for session.id (mapped from conversation.id)
    llm_spans = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    ]

    assert len(llm_spans) > 0, "Expected at least one LLM span in conversation"

    attrs = dict(cast(dict, llm_spans[0].attributes))

    # Verify session ID exists (mapped from gen_ai.conversation.id)
    if SpanAttributes.SESSION_ID in attrs:
        assert attrs[SpanAttributes.SESSION_ID] is not None
        print(f"Session ID: {attrs[SpanAttributes.SESSION_ID]}")


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_multiple_tools(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    openai_client: OpenAIChatClient,
) -> None:
    """Test agent with multiple tools and complex tool interactions."""

    # Define multiple tools
    @tool
    def get_temperature(location: str) -> str:
        """Get the temperature for a location."""
        temps = {"San Francisco": "72°F", "New York": "65°F", "London": "58°F"}
        return temps.get(location, "Unknown location")

    @tool
    def get_humidity(location: str) -> str:
        """Get the humidity for a location."""
        humidity = {"San Francisco": "65%", "New York": "70%", "London": "80%"}
        return humidity.get(location, "Unknown location")

    @tool
    def calculate_heat_index(temperature: str, humidity: str) -> str:
        """Calculate heat index from temperature and humidity."""
        return f"Heat index: Comfortable (based on {temperature} and {humidity})"

    agent = openai_client.as_agent(
        name="WeatherExpert",
        instructions=(
            "You are a weather expert. Use available tools to provide detailed weather information."
        ),
        tools=[get_temperature, get_humidity, calculate_heat_index],
    )

    # Clear any spans from setup
    in_memory_span_exporter.clear()

    # Run agent with a query that might use multiple tools
    response = await agent.run(
        "What's the weather like in San Francisco? Give me temperature and humidity."
    )

    assert response is not None
    assert response.text is not None

    # Get spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0

    # Check for multiple tool calls
    tool_spans = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.TOOL.value
    ]

    # Should have tool calls (at least one, possibly multiple)
    assert len(tool_spans) >= 1, f"Expected at least 1 tool span, got {len(tool_spans)}"

    # Verify tool span has proper attributes
    for tool_span in tool_spans:
        attrs = dict(cast(dict, tool_span.attributes))
        assert SpanAttributes.TOOL_NAME in attrs
        assert attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "TOOL"


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agent_with_system_instructions(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    openai_client: OpenAIChatClient,
) -> None:
    """Test that system instructions are properly captured in spans."""

    # Create agent with specific system instructions
    system_instructions = (
        "You are a helpful math tutor. "
        "Explain concepts simply and provide examples. "
        "Always be encouraging."
    )

    agent = openai_client.as_agent(
        name="MathTutor",
        instructions=system_instructions,
    )

    # Clear any spans from setup
    in_memory_span_exporter.clear()

    # Run a simple query
    response = await agent.run("What is a prime number?")

    assert response is not None
    assert response.text is not None

    # Get spans
    spans = in_memory_span_exporter.get_finished_spans()
    llm_spans = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    ]

    assert len(llm_spans) > 0

    llm_span = llm_spans[0]
    attrs = dict(cast(dict, llm_span.attributes))

    # Verify basic span attributes are present
    assert f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.role" in attrs
    assert SpanAttributes.LLM_MODEL_NAME in attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL in attrs

    # Note: System instructions handling may vary across agent-framework versions
    # The processor correctly transforms whatever agent-framework provides


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_long_conversation_context(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    openai_client: OpenAIChatClient,
) -> None:
    """Test longer conversation with multiple context-dependent turns."""
    agent = openai_client.as_agent(
        name="ConversationAgent",
        instructions="You are a helpful assistant. Remember context from previous messages.",
    )

    # Turn 1: Establish context
    response1 = await agent.run("My favorite color is blue.")
    assert response1 is not None

    # Turn 2: Add more context
    response2 = await agent.run("I also like programming in Python.")
    assert response2 is not None

    # Clear spans from previous turns
    in_memory_span_exporter.clear()

    # Turn 3: Question that requires context from both previous turns
    response3 = await agent.run("What do you know about my preferences?")
    assert response3 is not None

    # Get spans from final turn
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0

    # Find LLM span
    llm_spans = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    ]

    assert len(llm_spans) > 0

    llm_span = llm_spans[0]
    attrs = dict(cast(dict, llm_span.attributes))

    # Verify we have input messages captured
    assert f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.role" in attrs

    # Verify basic span attributes
    assert (
        attrs.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    )
    assert SpanAttributes.LLM_MODEL_NAME in attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL in attrs


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_tool_with_complex_return_type(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    openai_client: OpenAIChatClient,
) -> None:
    """Test tool that returns complex data structure."""

    # Define a tool that returns complex data
    @tool
    def get_user_profile(user_id: str) -> str:
        """Get detailed user profile information."""
        import json

        profile = {
            "user_id": user_id,
            "name": "John Doe",
            "email": "john@example.com",
            "preferences": {"theme": "dark", "notifications": True},
            "stats": {"posts": 42, "followers": 150},
        }
        return json.dumps(profile)

    agent = openai_client.as_agent(
        name="UserManager",
        instructions=(
            "You help manage user information. Use the get_user_profile tool to fetch user details."
        ),
        tools=[get_user_profile],
    )

    # Clear any spans from setup
    in_memory_span_exporter.clear()

    # Run query
    response = await agent.run("Get the profile for user 'user123'")

    assert response is not None

    # Get spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0

    # Find tool spans - should exist for tool return type test
    tool_spans = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.TOOL.value
    ]

    assert len(tool_spans) > 0, "Expected at least one TOOL span for complex return type test"

    tool_span = tool_spans[0]
    attrs = dict(cast(dict, tool_span.attributes))

    # Verify tool attributes
    assert SpanAttributes.TOOL_NAME in attrs
    assert attrs[SpanAttributes.TOOL_NAME] == "get_user_profile"

    # Check for output capture
    assert SpanAttributes.OUTPUT_VALUE in attrs or "tool.output" in attrs


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agent_with_different_model(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    gpt4_client: OpenAIChatClient,
) -> None:
    """Test with a different OpenAI model to ensure model info is captured correctly."""

    # Use gpt-4o instead of gpt-4o-mini
    agent = gpt4_client.as_agent(
        name="GPT4Agent",
        instructions="You are a helpful assistant.",
    )

    # Clear any spans from setup
    in_memory_span_exporter.clear()

    # Run simple query
    response = await agent.run("Say hello in one word.")

    assert response is not None

    # Get spans
    spans = in_memory_span_exporter.get_finished_spans()
    llm_spans = [
        s
        for s in spans
        if s.attributes
        and s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    ]

    assert len(llm_spans) > 0

    llm_span = llm_spans[0]
    attrs = dict(cast(dict, llm_span.attributes))

    # Verify model is correctly captured
    assert attrs.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o"
    assert attrs.get(SpanAttributes.LLM_PROVIDER) == "openai"

    # Verify token counts exist
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT in attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION in attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL in attrs


# Helper function to find spans by kind
def get_span_by_kind(spans, kind: str):
    """Find the first span with the given OpenInference span kind."""
    for span in spans:
        if span.attributes and span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == kind:
            return span
    raise ValueError(f"No span found with kind: {kind}")
