import logging
import sys
from unittest.mock import patch

import pytest
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.teams import BaseGroupChat
from opentelemetry import trace as trace_api

from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor

# The four provider-agnostic (agent/team/tool) methods that must be wrapped regardless of
# whether the OpenAI client is importable. Referenced as (class, attribute) so the wrapped
# attribute can be re-fetched after instrumentation rather than captured at import time.
_PROVIDER_AGNOSTIC_METHODS = (
    (AssistantAgent, "on_messages_stream"),
    (BaseChatAgent, "on_messages_stream"),
    (BaseGroupChat, "run_stream"),
    (AssistantAgent, "_execute_tool_call"),
)


def test_instrument_without_openai(
    tracer_provider: trace_api.TracerProvider,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`instrument()` must succeed when `autogen_ext.models.openai` is not importable.

    Reproduces the scenario from #3328: an autogen app using a non-OpenAI model client
    (e.g. `autogen-ext[anthropic]`) with `openai` not installed. Only the four
    provider-agnostic wrappers should be applied, and a debug log should note that the
    OpenAI-client instrumentation was skipped.
    """
    instrumentor = AutogenAgentChatInstrumentor()
    # Undo the autouse `instrument` fixture so we can re-instrument with openai unavailable.
    instrumentor.uninstrument()

    # Setting the module to None in sys.modules makes `import autogen_ext.models.openai`
    # raise ImportError, simulating an environment where `openai` is not installed.
    with patch.dict(sys.modules, {"autogen_ext.models.openai": None}):
        with caplog.at_level(
            logging.DEBUG, logger="openinference.instrumentation.autogen_agentchat"
        ):
            # Must not raise (previously a ModuleNotFoundError).
            instrumentor.instrument(tracer_provider=tracer_provider)

    try:
        # Only the four provider-agnostic wrappers are applied; the two OpenAI-client
        # wrappers are skipped.
        assert len(instrumentor._originals) == 4
        for cls, attribute in _PROVIDER_AGNOSTIC_METHODS:
            assert hasattr(getattr(cls, attribute), "__wrapped__")

        # The debug log documenting the skipped OpenAI-client instrumentation is emitted.
        assert any(
            "skipping OpenAI chat-completion client instrumentation" in record.getMessage()
            for record in caplog.records
        )
    finally:
        instrumentor.uninstrument()
