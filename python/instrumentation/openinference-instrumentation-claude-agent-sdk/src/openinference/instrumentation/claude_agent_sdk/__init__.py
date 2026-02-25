"""OpenInference instrumentation for the Claude Agent SDK (Python)."""

import importlib
import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.claude_agent_sdk._wrappers import (
    _ClientConnectWrapper,
    _ClientQueryWrapper,
    _ClientReceiveResponseWrapper,
    _QueryWrapper,
)
from openinference.instrumentation.claude_agent_sdk.version import __version__

logger = logging.getLogger(__name__)

_instruments = ("claude-agent-sdk>=0.1.0",)


class ClaudeAgentSDKInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for the Claude Agent SDK (query() and ClaudeSDKClient)."""

    __slots__ = (
        "_original_query",
        "_original_client_connect",
        "_original_client_query",
        "_original_client_receive_response",
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

        self._original_query = query_module.query
        wrap_function_wrapper(
            module="claude_agent_sdk.query",
            name="query",
            wrapper=_QueryWrapper(tracer=self._tracer),
        )
        # Sync package export so "from claude_agent_sdk import query" resolves to the wrapper
        import claude_agent_sdk

        setattr(claude_agent_sdk, "query", query_module.query)

        # ClaudeSDKClient: trace each response turn (connect/query + receive_response())
        client_module = importlib.import_module("claude_agent_sdk.client")
        ClaudeSDKClient = getattr(client_module, "ClaudeSDKClient", None)
        if ClaudeSDKClient is not None:
            self._original_client_connect = ClaudeSDKClient.connect
            self._original_client_query = ClaudeSDKClient.query
            self._original_client_receive_response = ClaudeSDKClient.receive_response
            wrap_function_wrapper(
                module="claude_agent_sdk.client",
                name="ClaudeSDKClient.connect",
                wrapper=_ClientConnectWrapper(tracer=self._tracer),
            )
            wrap_function_wrapper(
                module="claude_agent_sdk.client",
                name="ClaudeSDKClient.query",
                wrapper=_ClientQueryWrapper(tracer=self._tracer),
            )
            wrap_function_wrapper(
                module="claude_agent_sdk.client",
                name="ClaudeSDKClient.receive_response",
                wrapper=_ClientReceiveResponseWrapper(tracer=self._tracer),
            )
        else:
            self._original_client_connect = None
            self._original_client_query = None
            self._original_client_receive_response = None

    def _uninstrument(self, **kwargs: Any) -> None:
        import claude_agent_sdk

        query_module = importlib.import_module("claude_agent_sdk.query")
        if self._original_query is not None:
            query_module.query = self._original_query  # type: ignore[attr-defined]
            setattr(claude_agent_sdk, "query", self._original_query)
            self._original_query = None

        client_module = importlib.import_module("claude_agent_sdk.client")
        ClaudeSDKClient = getattr(client_module, "ClaudeSDKClient", None)
        if ClaudeSDKClient is not None and self._original_client_query is not None:
            ClaudeSDKClient.connect = self._original_client_connect
            ClaudeSDKClient.query = self._original_client_query
            ClaudeSDKClient.receive_response = self._original_client_receive_response
            self._original_client_connect = None
            self._original_client_query = None
            self._original_client_receive_response = None
