import logging
from typing import TYPE_CHECKING, Any, Collection, Optional, cast

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from opentelemetry.trace import Tracer
from wrapt import FunctionWrapper, wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.openai_agents.package import _instruments
from openinference.instrumentation.openai_agents.version import __version__

if TYPE_CHECKING:
    from agents.tracing import TracingProcessor

    from openinference.instrumentation.openai_agents._processor import (
        OpenInferenceTracingProcessor,
    )

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_REALTIME_MODULE = "agents.realtime.session"
_REALTIME_PUT_EVENT_ATTR = "RealtimeSession._put_event"
_REALTIME_SEND_AUDIO_ATTR = "RealtimeSession.send_audio"
_REALTIME_CLOSE_ATTR = "RealtimeSession.close"


def _patch_tool_execution() -> Optional[list[tuple[Any, str, Any]]]:
    """Wrap the SDK step that executes function tool calls, so function spans can report
    tool.description and tool.parameters.

    Every binding of the step is patched, not just the defining module, because the call
    sites import it by name and hold their own references. Returns what ``_uninstrument``
    needs to undo the patches, or ``None`` when nothing could be patched -- in which case
    those two attributes are simply not recorded, which is a missing enrichment rather
    than a failure.
    """
    from openinference.instrumentation.openai_agents._tool_schemas import (
        find_tool_execution_bindings,
        make_execute_function_tools_wrapper,
    )

    patched: list[tuple[Any, str, Any]] = []
    for owner, attribute in find_tool_execution_bindings():
        try:
            # Read from __dict__ so whatever the owner really holds is captured and can be
            # restored exactly, rather than anything getattr would synthesise on the way out.
            original = owner.__dict__[attribute]
            setattr(
                owner, attribute, FunctionWrapper(original, make_execute_function_tools_wrapper())
            )
        except Exception:
            logger.debug("could not patch %s.%s", owner, attribute, exc_info=True)
            continue
        logger.debug("Tool schema capture enabled: patched %s.%s", owner, attribute)
        patched.append((owner, attribute, original))
    if not patched:
        logger.debug(
            "No known function tool execution step could be patched -- tool.description and "
            "tool.parameters will not be recorded on function spans"
        )
        return None
    return patched


def _get_trace_processors() -> Optional[list["TracingProcessor"]]:
    """Return the processors registered with the Agents SDK, or ``None`` when they cannot
    be read.

    The SDK can add a processor or replace them all, but cannot list or remove one, so this
    reads the default trace provider's private list. A custom ``TraceProvider``, or an SDK
    that moves the list, gets ``None``, and callers fall back to what they can do without.
    """
    try:
        from agents.tracing import get_trace_provider

        multi_processor = getattr(get_trace_provider(), "_multi_processor", None)
        processors = getattr(multi_processor, "_processors", None)
    except Exception:
        logger.debug("could not read the registered trace processors", exc_info=True)
        return None
    if not isinstance(processors, (tuple, list)):
        logger.debug("could not read the registered trace processors")
        return None
    return list(processors)


def _unregister_processor(
    processor: "OpenInferenceTracingProcessor",
    displaced: list["TracingProcessor"],
) -> None:
    """Undo what ``_instrument`` did to the SDK's processors.

    Only ``processor`` is removed: processors registered after it are kept, and the ones an
    exclusive ``instrument()`` replaced (``displaced``) are put back ahead of them, in their
    original order. If the list was replaced since, it is left as it is. ``processor``
    itself stays registered, tracing no new run, until the runs it is tracing have ended --
    it has to receive their end events to end their spans.
    """
    from agents import set_trace_processors

    registered = _get_trace_processors()
    if registered is None:
        # Nothing can be put back or taken out, so the best left is to stop tracing.
        processor.drain()
        return
    if any(p is processor for p in registered):
        restored = [p for p in displaced if not any(p is r for r in registered)]
        set_trace_processors([*restored, *registered])

    def remove() -> None:
        registered = _get_trace_processors()
        if registered is not None and any(p is processor for p in registered):
            set_trace_processors([p for p in registered if p is not processor])

    processor.drain(remove)


class OpenAIAgentsInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for openai-agents
    """

    __slots__ = (
        "_original_put_event",
        "_original_send_audio",
        "_original_close",
        "_original_execute_function_tools",
        "_processor",
        "_displaced_processors",
    )

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

        # Raises on an unsupported span kind before anything is patched.
        processor = OpenInferenceTracingProcessor(
            cast(Tracer, tracer), tool_span_kinds=kwargs.get("tool_span_kinds")
        )

        from agents import add_trace_processor, set_trace_processors

        registered = _get_trace_processors() or []
        # A processor an earlier uninstrument() left draining a run still in flight keeps
        # receiving that run's events even in exclusive mode, so its spans still end.
        draining = [
            p for p in registered if isinstance(p, OpenInferenceTracingProcessor) and p.draining
        ]
        self._processor: Optional["OpenInferenceTracingProcessor"] = processor
        self._displaced_processors: list["TracingProcessor"] = []
        if exclusive_processor:
            self._displaced_processors = [
                p for p in registered if not any(p is d for d in draining)
            ]
            set_trace_processors([*draining, processor])
        else:
            add_trace_processor(processor)

        self._original_execute_function_tools = _patch_tool_execution()

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
        processor = getattr(self, "_processor", None)
        if processor is not None:
            try:
                _unregister_processor(processor, getattr(self, "_displaced_processors", []))
            except Exception:
                logger.debug("trace processor uninstrument failed", exc_info=True)
        self._processor = None
        self._displaced_processors = []

        for container, attribute, original in (
            getattr(self, "_original_execute_function_tools", None) or ()
        ):
            try:
                setattr(container, attribute, original)
            except Exception:
                logger.debug("tool execution uninstrument failed", exc_info=True)
        self._original_execute_function_tools = None

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
