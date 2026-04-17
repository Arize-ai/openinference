"""OpenInference instrumentation for the Claude Agent SDK (Python)."""

import importlib
import logging
from typing import Any, Collection, Dict, List, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt import resolve_path, wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.claude_agent_sdk._wrappers import (
    _ClientConnectWrapper,
    _ClientQueryWrapper,
    _ClientReceiveResponseWrapper,
    _QueryWrapper,
)
from openinference.instrumentation.claude_agent_sdk.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_instruments = ("claude-agent-sdk>=0.1.45",)


class ClaudeAgentSDKInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Claude Agent SDK (query() and ClaudeSDKClient)."""

    __slots__ = (
        "_originals",
        "_tracer",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        # Load the submodule (package re-exports the function as claude_agent_sdk.query)
        query_module = importlib.import_module("claude_agent_sdk.query")

        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        self._tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        method_wrappers: Dict[str, Any] = {
            "claude_agent_sdk.query:query": _QueryWrapper(tracer=self._tracer),
        }

        # ClaudeSDKClient: trace each response turn (connect/query + receive_response())
        client_module = importlib.import_module("claude_agent_sdk.client")
        ClaudeSDKClient = getattr(client_module, "ClaudeSDKClient", None)
        if ClaudeSDKClient is not None:
            method_wrappers.update(
                {
                    "claude_agent_sdk.client:ClaudeSDKClient.connect": _ClientConnectWrapper(
                        tracer=self._tracer
                    ),
                    "claude_agent_sdk.client:ClaudeSDKClient.query": _ClientQueryWrapper(
                        tracer=self._tracer
                    ),
                    "claude_agent_sdk.client:ClaudeSDKClient.receive_response": (
                        _ClientReceiveResponseWrapper(tracer=self._tracer)
                    ),
                }
            )

        self._originals: List[Tuple[Any, Any, Any]] = []
        for qualified, wrapper in method_wrappers.items():
            module, name = qualified.split(":", 1)
            self._originals.append(resolve_path(module, name))
            wrap_function_wrapper(module, name, wrapper)

        # Sync package export so "from claude_agent_sdk import query" resolves to the wrapper
        import claude_agent_sdk

        setattr(claude_agent_sdk, "query", query_module.query)

    def _uninstrument(self, **kwargs: Any) -> None:
        import claude_agent_sdk

        for parent, attribute, original in getattr(self, "_originals", ()):
            setattr(parent, attribute, original)

        query_module = importlib.import_module("claude_agent_sdk.query")
        setattr(claude_agent_sdk, "query", query_module.query)
