"""OpenInference Claude Code Instrumentation."""

import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.claude_code._wrappers import (
    _ClientQueryWrapper,
    _QueryWrapper,
    _ReceiveResponseWrapper,
)
from openinference.instrumentation.claude_code.package import _instruments
from openinference.instrumentation.claude_code.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ClaudeCodeInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """An instrumentor for Claude Code SDK."""

    __slots__ = ("_tracer", "_is_instrumented")

    def __init__(self) -> None:
        super().__init__()
        self._tracer: OITracer = None  # type: ignore
        self._is_instrumented = False

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if self._is_instrumented:
            return

        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)

        tracer = trace_api.get_tracer(__name__, __version__, tracer_provider)
        self._tracer = OITracer(tracer, config=config)

        # Wrap query function
        wrap_function_wrapper(
            module="claude_agent_sdk",
            name="query",
            wrapper=_QueryWrapper(tracer=tracer),
        )

        # Wrap ClaudeSDKClient.query method to capture input
        wrap_function_wrapper(
            module="claude_agent_sdk",
            name="ClaudeSDKClient.query",
            wrapper=_ClientQueryWrapper(tracer=tracer),
        )

        # Wrap ClaudeSDKClient.receive_response method
        wrap_function_wrapper(
            module="claude_agent_sdk",
            name="ClaudeSDKClient.receive_response",
            wrapper=_ReceiveResponseWrapper(tracer=tracer),
        )

        self._is_instrumented = True
        logger.info("Claude Code instrumentation enabled")

    def _uninstrument(self, **kwargs: Any) -> None:
        if not self._is_instrumented:
            return

        # Unwrap query function
        import claude_agent_sdk

        if hasattr(claude_agent_sdk.query, "__wrapped__"):
            claude_agent_sdk.query = claude_agent_sdk.query.__wrapped__

        # Unwrap ClaudeSDKClient.query
        if hasattr(claude_agent_sdk.ClaudeSDKClient.query, "__wrapped__"):
            claude_agent_sdk.ClaudeSDKClient.query = (
                claude_agent_sdk.ClaudeSDKClient.query.__wrapped__
            )

        # Unwrap ClaudeSDKClient.receive_response
        if hasattr(claude_agent_sdk.ClaudeSDKClient.receive_response, "__wrapped__"):
            claude_agent_sdk.ClaudeSDKClient.receive_response = (
                claude_agent_sdk.ClaudeSDKClient.receive_response.__wrapped__
            )

        self._is_instrumented = False
        self._tracer = None  # type: ignore
        logger.info("Claude Code instrumentation disabled")


__all__ = ["ClaudeCodeInstrumentor", "__version__"]
