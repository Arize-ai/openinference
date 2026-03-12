"""
OpenInference instrumentation for AG2 (formerly AutoGen).

Instruments the following AG2 entry points:
  ConversableAgent.initiate_chat          → CHAIN span (conversation root)
  ConversableAgent.generate_reply         → AGENT span (per turn)
  ConversableAgent.execute_function       → TOOL span
  GroupChatManager.run_chat               → CHAIN span + graph topology
  run_swarm                               → CHAIN span + handoff events
  ReasoningAgent.generate_reply           → AGENT span + tree-of-thought attrs
  ConversableAgent.initiate_chats         → CHAIN span (nested chat pipeline)
"""
from typing import Any, Collection, List, Tuple

import wrapt
from wrapt import resolve_path, wrap_function_wrapper
from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from openinference.instrumentation import OITracer, TraceConfig

from ._wrappers import (
    _InitiateChatWrapper,
    _GenerateReplyWrapper,
    _ExecuteFunctionWrapper,
    _GroupChatWrapper,
    _SwarmChatWrapper,
    _ReasoningAgentWrapper,
    _InitiateChatsWrapper,
)

# Public import path:
#   from openinference.instrumentation.ag2 import AG2Instrumentor

# Why ("autogen", "ConversableAgent.initiate_chat") and not
# ("autogen.agentchat.conversable_agent", "ConversableAgent.initiate_chat"):
#
# wrapt.wrap_function_wrapper(module, name, wrapper) internally calls
# resolve_path(module, name), which does __import__(module) and then traverses
# name.split('.') via getattr(). With module="autogen" and name="ConversableAgent.initiate_chat":
#   1. __import__("autogen") succeeds
#   2. getattr(autogen_module, "ConversableAgent") → the class (re-exported in autogen/__init__.py)
#   3. getattr(ConversableAgent, "initiate_chat") → the unbound method
#
# If we used module="autogen.agentchat.conversable_agent" instead, wrapt splits the
# dotted name and tries __import__("autogen.agentchat.conversable_agent.ConversableAgent"),
# which fails with ModuleNotFoundError — ConversableAgent is a class, not a submodule.
# Patching via the public autogen namespace is also more stable: it survives any future
# internal refactoring of ag2's submodule layout.
_METHODS = [
    ("autogen", "ConversableAgent.initiate_chat",      _InitiateChatWrapper),
    ("autogen", "ConversableAgent.generate_reply",     _GenerateReplyWrapper),
    ("autogen", "ConversableAgent.execute_function",   _ExecuteFunctionWrapper),
    ("autogen", "GroupChatManager.run_chat",           _GroupChatWrapper),
    ("autogen", "run_swarm",                           _SwarmChatWrapper),
    ("autogen", "ConversableAgent.initiate_chats",     _InitiateChatsWrapper),
]


class AG2Instrumentor(BaseInstrumentor):
    """
    OpenInference instrumentor for AG2 (formerly AutoGen) v0.11+.

    Usage:
        from openinference.instrumentation.ag2 import AG2Instrumentor
        AG2Instrumentor().instrument(tracer_provider=provider)
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("ag2 >= 0.11.0",)

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = OITracer(
            trace.get_tracer(__name__, tracer_provider=tracer_provider),
            config=kwargs.get("config", TraceConfig()),
        )
        self._originals: List[Tuple[Any, str, Any]] = []
        for module, method, wrapper_cls in _METHODS:
            parent, attribute, original = resolve_path(module, method)
            self._originals.append((parent, attribute, original))
            wrap_function_wrapper(module, method, wrapper_cls(tracer))

        # ReasoningAgent is in autogen.agents.experimental in AG2 0.11+
        reasoning_mod = "autogen.agents.experimental"
        reasoning_attr = "ReasoningAgent.generate_reply"
        parent, attribute, original = resolve_path(reasoning_mod, reasoning_attr)
        self._originals.append((parent, attribute, original))
        wrap_function_wrapper(reasoning_mod, reasoning_attr, _ReasoningAgentWrapper(tracer))

    def _uninstrument(self, **kwargs: Any) -> None:
        for parent, attribute, original in getattr(self, "_originals", []):
            setattr(parent, attribute, original)
        self._originals = []
