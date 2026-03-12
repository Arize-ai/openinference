"""
Unit tests for AG2 (formerly AutoGen) OpenInference instrumentation.
Uses AG2's ConversableAgent with a mock LLM to avoid API calls.
"""
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.ag2 import AG2Instrumentor

@pytest.fixture
def tracer_provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    AG2Instrumentor().uninstrument()

def test_initiate_chat_creates_chain_span(tracer_provider):
    from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

    provider, exporter = tracer_provider
    AG2Instrumentor().instrument(tracer_provider=provider)

    # Use mock LLM reply to avoid API call
    def mock_reply(recipient, messages, sender, config):
        return True, "Mock response. TERMINATE"

    assistant = AssistantAgent(
        name="assistant",
        llm_config=False,  # no LLM — use reply_func
    )
    assistant.register_reply(trigger=ConversableAgent, reply_func=mock_reply)

    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER", code_execution_config=False,
        is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""),
    )
    user.initiate_chat(assistant, message="Hello", max_turns=1)

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    chain_span = next((s for s in spans if "initiate_chat" in s.name or "CHAIN" in str(s.kind)), None)
    assert chain_span is not None

def test_initiate_chat_span_attributes(tracer_provider):
    from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

    provider, exporter = tracer_provider
    AG2Instrumentor().instrument(tracer_provider=provider)

    def mock_reply(recipient, messages, sender, config):
        return True, "Done. TERMINATE"

    assistant = AssistantAgent(name="test_assistant", llm_config=False)
    assistant.register_reply(trigger=ConversableAgent, reply_func=mock_reply)
    user = UserProxyAgent(name="test_user", human_input_mode="NEVER", code_execution_config=False,
                          is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""))
    user.initiate_chat(assistant, message="Test input", max_turns=1)

    spans = exporter.get_finished_spans()
    attrs = {k: v for s in spans for k, v in s.attributes.items()}
    assert attrs.get("ag2.initiator.name") == "test_user"
    assert attrs.get("ag2.recipient.name") == "test_assistant"

def test_execute_function_creates_tool_span(tracer_provider):
    from autogen import AssistantAgent, UserProxyAgent, ConversableAgent

    provider, exporter = tracer_provider
    AG2Instrumentor().instrument(tracer_provider=provider)

    call_count = {"n": 0}
    def mock_tool(query: str) -> str:
        call_count["n"] += 1
        return f"result for {query}"

    def mock_llm_reply(recipient, messages, sender, config):
        if call_count["n"] == 0:
            # First: trigger tool call
            return True, {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "mock_tool", "arguments": '{"query": "test"}'}
            }
        return True, "Done. TERMINATE"

    assistant = AssistantAgent(name="assistant", llm_config=False)
    assistant.register_reply(trigger=ConversableAgent, reply_func=mock_llm_reply)
    user = UserProxyAgent(name="user", human_input_mode="NEVER", code_execution_config=False,
                          is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""))
    # Register directly in _function_map (caller=None avoids llm_config requirement)
    user._function_map["mock_tool"] = mock_tool
    user.initiate_chat(assistant, message="Run tool", max_turns=3)

    spans = exporter.get_finished_spans()
    tool_spans = [s for s in spans if "tool" in s.name.lower() or
                  any("tool.name" in k for k in s.attributes)]
    assert len(tool_spans) >= 1

def test_group_chat_span_attributes(tracer_provider):
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent

    provider, exporter = tracer_provider
    AG2Instrumentor().instrument(tracer_provider=provider)

    def mock_reply(recipient, messages, sender, config):
        return True, "Done. TERMINATE"

    agent_a = AssistantAgent(name="agent_a", llm_config=False)
    agent_a.register_reply(trigger=ConversableAgent, reply_func=mock_reply)
    agent_b = AssistantAgent(name="agent_b", llm_config=False)
    user = UserProxyAgent(name="user", human_input_mode="NEVER", code_execution_config=False,
                          is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""))
    gc = GroupChat(agents=[user, agent_a, agent_b], messages=[], max_round=3,
                   speaker_selection_method="round_robin")
    manager = GroupChatManager(groupchat=gc, llm_config=False)
    user.initiate_chat(manager, message="Start research")

    spans = exporter.get_finished_spans()
    attrs = {k: v for s in spans for k, v in s.attributes.items()}
    assert "ag2.groupchat.agents" in attrs
    import json
    agents_in_span = json.loads(attrs["ag2.groupchat.agents"])
    assert "agent_a" in agents_in_span

def test_instrumentor_uninstruments_cleanly(tracer_provider):
    provider, exporter = tracer_provider
    instrumentor = AG2Instrumentor()
    instrumentor.instrument(tracer_provider=provider)
    instrumentor.uninstrument()
    # After uninstrument, no new spans should be emitted for agent calls
    from autogen import AssistantAgent, UserProxyAgent, ConversableAgent
    exporter.clear()

    def mock_reply(recipient, messages, sender, config):
        return True, "Uninstrumented. TERMINATE"

    assistant = AssistantAgent(name="assistant", llm_config=False)
    assistant.register_reply(trigger=ConversableAgent, reply_func=mock_reply)
    user = UserProxyAgent(name="user", human_input_mode="NEVER", code_execution_config=False,
                          is_termination_msg=lambda m: "TERMINATE" in (m.get("content") or ""))
    user.initiate_chat(assistant, message="test", max_turns=1)
    assert len(exporter.get_finished_spans()) == 0
