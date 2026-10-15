"""OpenInference instrumentation for livekit-agents voice sessions."""

from __future__ import annotations

import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from opentelemetry.instrumentation.utils import unwrap
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.livekit_agents.package import _instruments
from openinference.instrumentation.livekit_agents.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["LiveKitAgentsInstrumentor"]

_SESSION_MODULE = "livekit.agents.voice.agent_session"
_LLM_MODULE = "livekit.agents.llm.llm"


class LiveKitAgentsInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Emit one OpenInference audio turn for each reply an AgentSession speaks.

    It wraps AgentSession.start to follow the session's speech and user events and to tap
    its audio, and LLMStream to record each model call. A turn becomes a conversation.turn
    span of kind AUDIO with user, assistant, and tool child spans, rendered when the speech
    ends. LiveKit's own telemetry does not need to be configured.

    instrument(exclusive=True), the default, also points livekit-agents' own tracer at a
    no-op provider, so LiveKit's spans do not repeat the turn's model calls and token counts.
    Pass exclusive=False to keep them. uninstrument restores LiveKit's provider.
    """

    _livekit_tracer_provider: trace_api.TracerProvider | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        from openinference.instrumentation.livekit_agents._wrappers import (
            StartWrapper,
            llm_stream_aclose,
            llm_stream_anext,
            llm_stream_init,
        )

        if kwargs.get("exclusive", True):
            from livekit.agents.telemetry import tracer as livekit_tracer

            self._livekit_tracer_provider = livekit_tracer._tracer_provider
            livekit_tracer.set_provider(trace_api.NoOpTracerProvider())

        wrap_function_wrapper(_SESSION_MODULE, "AgentSession.start", StartWrapper(tracer, config))
        wrap_function_wrapper(_LLM_MODULE, "LLMStream.__init__", llm_stream_init)
        wrap_function_wrapper(_LLM_MODULE, "LLMStream.__anext__", llm_stream_anext)
        wrap_function_wrapper(_LLM_MODULE, "LLMStream.aclose", llm_stream_aclose)

    def _uninstrument(self, **kwargs: Any) -> None:
        from livekit.agents.llm.llm import LLMStream
        from livekit.agents.voice.agent_session import AgentSession

        unwrap(AgentSession, "start")
        if (provider := self._livekit_tracer_provider) is not None:
            from livekit.agents.telemetry import tracer as livekit_tracer

            livekit_tracer.set_provider(provider)
            self._livekit_tracer_provider = None
        for method in ("__init__", "__anext__", "aclose"):
            unwrap(LLMStream, method)
