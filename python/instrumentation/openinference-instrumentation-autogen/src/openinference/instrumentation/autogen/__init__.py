"""
OpenInference instrumentation for AG2 (formerly AutoGen) v0.11+.

Instruments the following AG2 entry points:
  ConversableAgent.initiate_chat      → CHAIN span (conversation root)
  ConversableAgent.generate_reply     → AGENT span (per turn)
  ConversableAgent.execute_function   → TOOL span
  GroupChatManager.run_chat           → CHAIN span + graph topology
  run_swarm                           → CHAIN span
  ReasoningAgent.generate_reply       → AGENT span + tree-of-thought attrs
  ConversableAgent.initiate_chats     → CHAIN span (nested chat pipeline)
"""

from typing import Any, Collection, List, Tuple

from opentelemetry import trace
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import resolve_path, wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig

from ._wrappers import (
    _ExecuteFunctionWrapper,
    _GenerateReplyWrapper,
    _GroupChatWrapper,
    _InitiateChatWrapper,
    _InitiateChatsWrapper,
    _ReasoningAgentWrapper,
    _SwarmChatWrapper,
)
from .version import __version__

_METHODS = [
    ("autogen", "ConversableAgent.initiate_chat", _InitiateChatWrapper),
    ("autogen", "ConversableAgent.generate_reply", _GenerateReplyWrapper),
    ("autogen", "ConversableAgent.execute_function", _ExecuteFunctionWrapper),
    ("autogen", "GroupChatManager.run_chat", _GroupChatWrapper),
    ("autogen", "run_swarm", _SwarmChatWrapper),
    ("autogen", "ConversableAgent.initiate_chats", _InitiateChatsWrapper),
]

_REASONING_MODULE = "autogen.agents.experimental"
_REASONING_METHOD = "ReasoningAgent.generate_reply"


class AutogenInstrumentor(BaseInstrumentor):
    """
    OpenInference instrumentor for AG2 (formerly AutoGen) v0.11+.

    Usage:
        from openinference.instrumentation.autogen import AutogenInstrumentor
        AutogenInstrumentor().instrument(tracer_provider=provider)
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return ("ag2 >= 0.11.0",)

    def _instrument(self, **kwargs: Any) -> None:
        tracer_provider = kwargs.get("tracer_provider")
        tracer = OITracer(
            trace.get_tracer(__name__, __version__, tracer_provider=tracer_provider),
            config=kwargs.get("config", TraceConfig()),
        )
        # Save (parent, attribute, original) tuples before wrapping so
        # _uninstrument can restore them. wrapt has no unwrap() — the correct
        # pattern is resolve_path before wrapping, then setattr to restore.
        self._originals: List[Tuple[Any, str, Any]] = []
        for module, method, wrapper_cls in _METHODS:
            parent, attribute, original = resolve_path(module, method)
            self._originals.append((parent, attribute, original))
            wrap_function_wrapper(module, method, wrapper_cls(tracer))

        # ReasoningAgent lives in autogen.agents.experimental in AG2 0.11+
        parent, attribute, original = resolve_path(_REASONING_MODULE, _REASONING_METHOD)
        self._originals.append((parent, attribute, original))
        wrap_function_wrapper(_REASONING_MODULE, _REASONING_METHOD, _ReasoningAgentWrapper(tracer))

    def _uninstrument(self, **kwargs: Any) -> None:
        for parent, attribute, original in getattr(self, "_originals", []):
            setattr(parent, attribute, original)
        self._originals = []


__all__ = ["AutogenInstrumentor"]
