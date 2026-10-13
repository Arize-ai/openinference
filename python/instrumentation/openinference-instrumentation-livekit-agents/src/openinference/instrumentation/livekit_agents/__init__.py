"""Public instrumentor for livekit-agents voice turns.

Call LiveKitAgentsInstrumentor().instrument(tracer_provider=provider) on the
provider you pass to livekit.agents.telemetry.set_tracer_provider. The
instrumentor registers one span processor on that provider.
"""

from __future__ import annotations

import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)
from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.livekit_agents._processor import LiveKitTurnProcessor
from openinference.instrumentation.livekit_agents.package import _instruments
from openinference.instrumentation.livekit_agents.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["LiveKitAgentsInstrumentor"]


class LiveKitAgentsInstrumentor(BaseInstrumentor):  # type: ignore[misc]
    """Emit one OpenInference audio turn for each LiveKit agent_turn.

    instrument kwargs are tracer_provider and config. config defaults to
    TraceConfig(). The tracer provider defaults to the global provider.
    Both entry points load this class:

    opentelemetry_instrumentor / livekit_agents
    openinference_instrumentor / livekit_agents

    The processor is added with add_span_processor and is not prepended.
    LiveKit's set_tracer_provider prepends its PII processor ahead of
    processors already on the provider, so this processor still runs
    after PII whether instrument() is called first or second.

    A second instrument() on the same instance enables the existing
    processor when the provider is the same object. A different provider
    gets its own processor. uninstrument disables the processor and leaves
    it registered. TracerProvider has no remove_span_processor.
    """

    _version = __version__
    _processor_type = LiveKitTurnProcessor

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        provider = kwargs.get("tracer_provider") or trace_api.get_tracer_provider()
        config = kwargs.get("config")
        if config is None:
            config = TraceConfig()
        if not isinstance(config, TraceConfig):
            raise TypeError("config must be a TraceConfig")
        if not isinstance(provider, TracerProvider):
            logger.warning(
                "LiveKit Agents instrumentation requires an SDK TracerProvider; "
                "pass the provider given to livekit.agents.telemetry.set_tracer_provider"
            )
            return

        processor = getattr(self, "_processor", None)
        if processor is not None and getattr(self, "_tracer_provider", None) is provider:
            processor.enable()
            return

        tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, provider),
            config=config,
        )
        processor = self._processor_type(tracer, config)
        provider.add_span_processor(processor)
        self._tracer_provider = provider
        self._processor = processor

    def _uninstrument(self, **kwargs: Any) -> None:
        processor = getattr(self, "_processor", None)
        if processor is not None:
            processor.disable()
