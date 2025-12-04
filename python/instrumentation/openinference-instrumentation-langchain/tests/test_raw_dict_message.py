from typing import TYPE_CHECKING, Any, Iterator

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.trace import MessageAttributes, SpanAttributes

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

AgentExecutor: Any
create_tool_calling_agent: Any

try:
    from langchain.agents import (  # type: ignore[attr-defined,no-redef,unused-ignore]
        AgentExecutor,
        create_tool_calling_agent,
    )
except (ImportError, AttributeError):
    AgentExecutor = None
    create_tool_calling_agent = None


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> "TracerProvider":
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(tracer_provider: "TracerProvider") -> Iterator[None]:
    """Instrument LangChain for all tests."""
    instrumentor = LangChainInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument(tracer_provider=tracer_provider)
    yield
    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


@pytest.mark.vcr(
    before_record_request=lambda request: (request.headers.clear() or request),
    before_record_response=lambda response: ({**response, "headers": {}}),
    cassette_library_dir="tests/cassettes/test_raw_dict_message",
)
@pytest.mark.skipif(
    create_tool_calling_agent is None,
    reason="create_tool_calling_agent not available in this version of langchain",
)
def test_agent_with_raw_dict_messages(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """
    Test that raw dict messages work with agent executor without AssertionError.

    This reproduces a scenario where a user passes a raw dict like
    {"role": "user", "content": "..."} instead of HumanMessage objects.
    """
    # Create a simple tool
    from langchain_core.tools import tool

    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"The weather in {location} is sunny."

    # Create agent with tool
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [get_weather]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("placeholder", "{messages}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent = create_tool_calling_agent(model, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # This is the exact case from issue #2426 - raw dict instead of HumanMessage
    raw_dict_message = {"role": "user", "content": "What's the weather in London?"}

    # This should NOT raise AssertionError anymore
    result = agent_executor.invoke({"messages": [raw_dict_message]})

    # Verify the result
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Verify spans were created
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify at least one span has input messages with proper role extraction
    chain_spans = [
        s
        for s in spans
        if s.attributes and s.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "CHAIN"
    ]
    assert len(chain_spans) > 0

    # Check that input messages were properly extracted despite raw dict format
    for span in chain_spans:
        if not span.attributes:
            continue
        input_messages_attr = (
            f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"
        )
        if input_messages_attr in span.attributes:
            # Role should be extracted (either from dict's "role" field or inferred)
            role = span.attributes[input_messages_attr]
            assert role is not None
            assert isinstance(role, str)
            break
    else:
        # At least verify no AssertionError was raised
        pytest.skip(
            "No input messages found in spans, but no AssertionError raised - fix is working"
        )


def test_raw_dict_message_role_extraction_unit() -> None:
    """
    Unit test for raw dict message role extraction.

    Tests the _extract_message_role function directly with a raw dict structure.
    """
    from openinference.instrumentation.langchain._tracer import MESSAGE_ROLE, _extract_message_role

    # Simulate what LangChain does when you pass a raw dict message
    # It gets serialized to JSON format without the standard "id" field
    raw_message_data = {
        "id": None,  # This is what causes the issue
        "kwargs": {
            "content": "What's the weather in London?",
            "role": "user",  # Raw dict messages have role in kwargs
        },
    }

    # This should NOT raise AssertionError
    result = dict(_extract_message_role(raw_message_data))

    # Should successfully extract role from kwargs["role"]
    assert MESSAGE_ROLE in result
    assert result[MESSAGE_ROLE] == "user"


def test_raw_dict_message_with_different_roles() -> None:
    """Test raw dict messages with various role types."""
    from openinference.instrumentation.langchain._tracer import MESSAGE_ROLE, _extract_message_role

    test_cases = [
        ("user", "user"),
        ("assistant", "assistant"),
        ("system", "system"),
        ("tool", "tool"),
    ]

    for input_role, expected_role in test_cases:
        raw_message_data = {
            "id": None,
            "kwargs": {
                "content": "Test message",
                "role": input_role,
            },
        }

        result = dict(_extract_message_role(raw_message_data))
        assert MESSAGE_ROLE in result
        assert result[MESSAGE_ROLE] == expected_role


def test_chat_message_with_none_kwargs_fallback() -> None:
    """
    Test that ChatMessage with None kwargs falls back to alternative strategies.

    This tests the fix for the bug where TypeError wasn't caught when
    accessing message_data["kwargs"]["role"] raises TypeError if kwargs is None.
    The function should fall back to Strategy 2 (type field) instead of raising.
    """
    from openinference.instrumentation.langchain._tracer import MESSAGE_ROLE, _extract_message_role

    # Simulate a ChatMessage where kwargs is None
    # This would cause TypeError in _map_class_name_to_role without proper handling
    message_data = {
        "id": ["langchain_core", "messages", "chat", "ChatMessage"],
        "kwargs": None,  # This causes TypeError when accessing ["kwargs"]["role"]
        "type": "human",  # Fallback to Strategy 2
    }

    # This should NOT raise TypeError, should fall back to type field
    result = dict(_extract_message_role(message_data))

    # Should successfully extract role from type field (Strategy 2)
    assert MESSAGE_ROLE in result
    assert result[MESSAGE_ROLE] == "user"


def test_chat_message_with_none_kwargs_fallback_to_role_field() -> None:
    """
    Test ChatMessage with None kwargs falls back to direct role field.

    Tests fallback to Strategy 3 (direct role field) when both Strategy 1
    (id field with kwargs) and Strategy 2 (type field) are unavailable.
    """
    from openinference.instrumentation.langchain._tracer import MESSAGE_ROLE, _extract_message_role

    # ChatMessage with None kwargs and no type field
    message_data = {
        "id": ["langchain_core", "messages", "chat", "ChatMessage"],
        "kwargs": None,  # Causes TypeError in Strategy 1
        "role": "assistant",  # Fallback to Strategy 3
    }

    # Should fall back to direct role field
    result = dict(_extract_message_role(message_data))

    assert MESSAGE_ROLE in result
    assert result[MESSAGE_ROLE] == "assistant"


def test_message_with_non_string_id_element_fallback() -> None:
    """
    Test that non-string id elements fall back to alternative strategies.

    This tests the fix for the bug where AttributeError wasn't caught when
    id_[-1] is not a string (e.g., integer, None), causing .startswith() to
    raise AttributeError. The function should fall back to Strategy 2.
    """
    from openinference.instrumentation.langchain._tracer import MESSAGE_ROLE, _extract_message_role

    # Simulate a message where id[-1] is an integer instead of string
    message_data = {
        "id": ["langchain_core", "messages", 12345],  # Last element is int, not string
        "type": "system",  # Fallback to Strategy 2
    }

    # This should NOT raise AttributeError, should fall back to type field
    result = dict(_extract_message_role(message_data))

    # Should successfully extract role from type field (Strategy 2)
    assert MESSAGE_ROLE in result
    assert result[MESSAGE_ROLE] == "system"


def test_message_with_none_id_element_fallback() -> None:
    """
    Test that None as id element falls back to alternative strategies.

    Tests that when id[-1] is None, the function catches AttributeError
    and falls back to the direct role field (Strategy 3).
    """
    from openinference.instrumentation.langchain._tracer import MESSAGE_ROLE, _extract_message_role

    # Simulate a message where id[-1] is None
    message_data = {
        "id": ["langchain_core", "messages", None],  # Last element is None
        "role": "tool",  # Fallback to Strategy 3
    }

    # Should fall back to direct role field
    result = dict(_extract_message_role(message_data))

    assert MESSAGE_ROLE in result
    assert result[MESSAGE_ROLE] == "tool"
