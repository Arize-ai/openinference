"""
Unit tests for AG2 (formerly AutoGen) OpenInference instrumentation.
Uses AG2's ConversableAgent with a mock reply function to avoid API calls.
"""

import json

import pytest
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import TraceConfig, using_attributes
from openinference.instrumentation.autogen import AutogenInstrumentor
from openinference.semconv.trace import SpanAttributes


@pytest.fixture
def tracer_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    AutogenInstrumentor().uninstrument()


def _mock_reply(recipient, messages, sender, config):
    return True, "Mock response. TERMINATE"


def test_entrypoint_for_opentelemetry_instrument() -> None:
    (instrumentor_entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="autogen")
    instrumentor = instrumentor_entrypoint.load()()
    assert isinstance(instrumentor, AutogenInstrumentor)


def test_initiate_chat_creates_chain_span(tracer_provider):
    from autogen import AssistantAgent, ConversableAgent, UserProxyAgent

    provider, exporter = tracer_provider
    AutogenInstrumentor().instrument(tracer_provider=provider)

    assistant = AssistantAgent(name="assistant", llm_config=False)
    assistant.register_reply(trigger=ConversableAgent, reply_func=_mock_reply)
    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
    )
    user.initiate_chat(assistant, message="Hello", max_turns=1)

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    span_names = [s.name for s in spans]
    assert any("user" in n and "assistant" in n for n in span_names)


def test_initiate_chat_span_attributes(tracer_provider):
    from autogen import AssistantAgent, ConversableAgent, UserProxyAgent

    provider, exporter = tracer_provider
    AutogenInstrumentor().instrument(tracer_provider=provider)

    assistant = AssistantAgent(name="test_assistant", llm_config=False)
    assistant.register_reply(trigger=ConversableAgent, reply_func=_mock_reply)
    user = UserProxyAgent(
        name="test_user",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
    )
    user.initiate_chat(assistant, message="Test input", max_turns=1)

    spans = exporter.get_finished_spans()
    all_attrs = {k: v for s in spans for k, v in s.attributes.items()}
    assert all_attrs.get("ag2.initiator.name") == "test_user"
    assert all_attrs.get("ag2.recipient.name") == "test_assistant"


def test_execute_function_creates_tool_span(tracer_provider):
    from autogen import ConversableAgent

    provider, exporter = tracer_provider
    AutogenInstrumentor().instrument(tracer_provider=provider)

    def mock_tool(query: str) -> str:
        return f"result for {query}"

    agent = ConversableAgent(name="executor", llm_config=False, code_execution_config=False)
    agent._function_map["mock_tool"] = mock_tool

    # Call execute_function directly — this is what the wrapper intercepts
    agent.execute_function({"name": "mock_tool", "arguments": '{"query": "test"}'})

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if any("ag2.tool" in k for k in s.attributes)]
    assert len(tool_spans) == 1
    assert tool_spans[0].attributes["ag2.tool.name"] == "mock_tool"


def test_group_chat_span_attributes(tracer_provider):
    from autogen import AssistantAgent, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent

    provider, exporter = tracer_provider
    AutogenInstrumentor().instrument(tracer_provider=provider)

    agent_a = AssistantAgent(name="agent_a", llm_config=False)
    agent_a.register_reply(trigger=ConversableAgent, reply_func=_mock_reply)
    agent_b = AssistantAgent(name="agent_b", llm_config=False)
    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
    )
    gc = GroupChat(
        agents=[user, agent_a, agent_b],
        messages=[],
        max_round=3,
        speaker_selection_method="round_robin",  # no LLM needed for speaker selection
    )
    manager = GroupChatManager(groupchat=gc, llm_config=False)
    user.initiate_chat(manager, message="Start")

    spans = exporter.get_finished_spans()
    all_attrs = {k: v for s in spans for k, v in s.attributes.items()}
    assert "ag2.groupchat.agents" in all_attrs
    assert "agent_a" in json.loads(all_attrs["ag2.groupchat.agents"])


def test_instrumentor_uninstruments_cleanly(tracer_provider):
    from autogen import AssistantAgent, ConversableAgent, UserProxyAgent

    provider, exporter = tracer_provider
    instrumentor = AutogenInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    instrumentor.uninstrument()
    exporter.clear()

    assistant = AssistantAgent(name="assistant", llm_config=False)
    assistant.register_reply(trigger=ConversableAgent, reply_func=_mock_reply)
    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
    )
    user.initiate_chat(assistant, message="test", max_turns=1)
    assert len(exporter.get_finished_spans()) == 0


def test_suppress_tracing(tracer_provider):
    from autogen import AssistantAgent, ConversableAgent, UserProxyAgent

    provider, exporter = tracer_provider
    AutogenInstrumentor().instrument(tracer_provider=provider)

    token = context_api.attach(context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        assistant = AssistantAgent(name="assistant", llm_config=False)
        assistant.register_reply(trigger=ConversableAgent, reply_func=_mock_reply)
        user = UserProxyAgent(
            name="user",
            human_input_mode="NEVER",
            code_execution_config=False,
            is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
        )
        user.initiate_chat(assistant, message="Hello", max_turns=1)
    finally:
        context_api.detach(token)

    assert len(exporter.get_finished_spans()) == 0


def test_context_attribute_propagation(tracer_provider):
    from autogen import AssistantAgent, ConversableAgent, UserProxyAgent

    provider, exporter = tracer_provider
    AutogenInstrumentor().instrument(tracer_provider=provider)

    assistant = AssistantAgent(name="assistant", llm_config=False)
    assistant.register_reply(trigger=ConversableAgent, reply_func=_mock_reply)
    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
    )

    with using_attributes(session_id="test-session-123", user_id="test-user-456"):
        user.initiate_chat(assistant, message="Hello", max_turns=1)

    spans = exporter.get_finished_spans()
    assert spans, "expected at least one span"
    for span in spans:
        assert span.attributes.get(SpanAttributes.SESSION_ID) == "test-session-123"
        assert span.attributes.get(SpanAttributes.USER_ID) == "test-user-456"


def test_trace_config_hides_inputs_outputs(tracer_provider):
    from autogen import AssistantAgent, ConversableAgent, UserProxyAgent

    provider, exporter = tracer_provider
    AutogenInstrumentor().instrument(
        tracer_provider=provider,
        config=TraceConfig(hide_inputs=True, hide_outputs=True),
    )

    assistant = AssistantAgent(name="assistant", llm_config=False)
    assistant.register_reply(trigger=ConversableAgent, reply_func=_mock_reply)
    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
    )
    user.initiate_chat(assistant, message="sensitive data", max_turns=1)

    spans = exporter.get_finished_spans()
    all_attrs = {k: v for s in spans for k, v in s.attributes.items()}
    assert "sensitive data" not in str(all_attrs.get(SpanAttributes.INPUT_VALUE, ""))
    assert "Mock response" not in str(all_attrs.get(SpanAttributes.OUTPUT_VALUE, ""))


def test_initiate_chats_creates_chain_span(tracer_provider):
    from autogen import AssistantAgent, ConversableAgent, UserProxyAgent

    provider, exporter = tracer_provider
    AutogenInstrumentor().instrument(tracer_provider=provider)

    assistant = AssistantAgent(name="assistant", llm_config=False)
    assistant.register_reply(trigger=ConversableAgent, reply_func=_mock_reply)
    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
    )

    chat_queue = [{"recipient": assistant, "message": "Hello", "max_turns": 1}]
    user.initiate_chats(chat_queue)

    spans = exporter.get_finished_spans()
    assert any("initiate_chats" in s.name for s in spans)
    all_attrs = {k: v for s in spans for k, v in s.attributes.items()}
    assert all_attrs.get("ag2.nested.chat_count") == 1
