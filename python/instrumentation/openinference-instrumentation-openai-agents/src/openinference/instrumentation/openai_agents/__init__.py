import logging
from typing import Any, Collection, cast

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import Tracer

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.openai_agents.package import _instruments
from openinference.instrumentation.openai_agents.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class OpenAIAgentsInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for openai-agents
    """

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
        from agents import add_trace_processor

        from openinference.instrumentation.openai_agents._processor import (
            OpenInferenceTracingProcessor,
        )

        add_trace_processor(OpenInferenceTracingProcessor(cast(Tracer, tracer)))

    def _uninstrument(self, **kwargs: Any) -> None:
        # TODO
        pass
