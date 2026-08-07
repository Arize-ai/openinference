import sys
from unittest.mock import patch

from autogen_agentchat.agents import AssistantAgent
from opentelemetry import trace as trace_api

from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor


def test_instrument_without_openai(
    tracer_provider: trace_api.TracerProvider,
) -> None:
    instrumentor = AutogenAgentChatInstrumentor()
    instrumentor.uninstrument()

    with patch.dict(sys.modules, {"autogen_ext.models.openai": None}):
        instrumentor.instrument(tracer_provider=tracer_provider)

    try:
        assert hasattr(AssistantAgent.on_messages_stream, "__wrapped__")
    finally:
        instrumentor.uninstrument()
