import logging
from typing import Any, Collection, cast

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import Tracer
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.openai_agents.package import _instruments
from openinference.instrumentation.openai_agents.version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_REALTIME_MODULE = "agents.realtime.session"
_REALTIME_PUT_EVENT_ATTR = "RealtimeSession._put_event"
_REALTIME_SEND_AUDIO_ATTR = "RealtimeSession.send_audio"
_REALTIME_CLOSE_ATTR = "RealtimeSession.close"


class OpenAIAgentsInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for openai-agents
    """

    __slots__ = ("_original_put_event", "_original_send_audio", "_original_close")

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()
        if (exclusive_processor := kwargs.get("exclusive_processor")) is None:
            exclusive_processor = True
        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)
        tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        from openinference.instrumentation.openai_agents._processor import (
            OpenInferenceTracingProcessor,
        )

        if exclusive_processor:
            from agents import set_trace_processors

            set_trace_processors([OpenInferenceTracingProcessor(cast(Tracer, tracer))])
        else:
            from agents import add_trace_processor

            add_trace_processor(OpenInferenceTracingProcessor(cast(Tracer, tracer)))

        from openinference.instrumentation.openai_agents._realtime import (
            _load_realtime_events,
            make_close_wrapper,
            make_realtime_wrapper,
            make_send_audio_wrapper,
        )

        if _load_realtime_events():
            try:
                from agents.realtime.session import RealtimeSession

                self._original_put_event = RealtimeSession._put_event
                wrap_function_wrapper(
                    _REALTIME_MODULE,
                    _REALTIME_PUT_EVENT_ATTR,
                    make_realtime_wrapper(tracer, config),
                )
                self._original_send_audio = RealtimeSession.send_audio
                wrap_function_wrapper(
                    _REALTIME_MODULE,
                    _REALTIME_SEND_AUDIO_ATTR,
                    make_send_audio_wrapper(),
                )
                self._original_close = RealtimeSession.close
                wrap_function_wrapper(
                    _REALTIME_MODULE,
                    _REALTIME_CLOSE_ATTR,
                    make_close_wrapper(),
                )
                logger.debug(
                    "Realtime tracing enabled: patched _put_event, send_audio, close on %s",
                    RealtimeSession.__qualname__,
                )
            except Exception:
                logger.debug(
                    "Could not patch RealtimeSession — realtime tracing disabled",
                    exc_info=True,
                )
        else:
            logger.debug("agents.realtime.events not importable — realtime tracing disabled")

    def _uninstrument(self, **kwargs: Any) -> None:
        try:
            from agents.realtime.session import RealtimeSession

            original = getattr(self, "_original_put_event", None)
            if original is not None:
                RealtimeSession._put_event = original  # type: ignore[method-assign]

            original = getattr(self, "_original_send_audio", None)
            if original is not None:
                RealtimeSession.send_audio = original  # type: ignore[method-assign]

            original = getattr(self, "_original_close", None)
            if original is not None:
                RealtimeSession.close = original  # type: ignore[method-assign]
        except Exception:
            logger.debug("realtime uninstrument failed", exc_info=True)
