import contextlib
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Callable, Collection, Generator

from opentelemetry.trace import StatusCode

from openinference.instrumentation._spans import OpenInferenceSpan

if TYPE_CHECKING:
    from beeai_framework.emitter import EventMeta

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)
from openinference.instrumentation.beeai._span import SpanWrapper
from openinference.instrumentation.beeai._utils import _datetime_to_span_time, exception_handler
from openinference.instrumentation.beeai.processors.base import Processor
from openinference.instrumentation.beeai.processors.locator import ProcessorLocator

logger = logging.getLogger(__name__)

_instruments = ("beeai-framework >= 0.1.32",)
try:
    __version__ = version("beeai-framework")
except PackageNotFoundError:
    __version__ = "unknown"


class BeeAIInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = ("_tracer", "_cleanup", "_processes", "_processes_deps")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup: Callable[[], None] = lambda: None
        self._processes: dict[str, Processor] = {}
        self._processes_deps: dict[str, list[Processor]] = {}

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        try:
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

            from beeai_framework.emitter import Emitter, EmitterOptions

            self._cleanup = Emitter.root().match(
                "*.*",
                self._handler,
                EmitterOptions(match_nested=True, is_blocking=True),
            )
        except Exception as e:
            logger.error("Instrumentation error", exc_info=e)

    def _uninstrument(self, **kwargs: Any) -> None:
        self._cleanup()
        self._processes.clear()
        self._processes_deps.clear()

    def _build_tree(self, processor: Processor) -> None:
        with self._build_tree_for_span(processor.span):
            for child in self._processes_deps.pop(processor.run_id):
                self._build_tree(child)
        self._processes.pop(processor.run_id)

    @contextlib.contextmanager
    def _build_tree_for_span(self, node: SpanWrapper) -> Generator[OpenInferenceSpan, None, None]:
        with self._tracer.start_as_current_span(
            name=node.name,
            openinference_span_kind=node.kind,
            attributes=node.attributes,
            start_time=_datetime_to_span_time(node.started_at) if node.started_at else None,
            end_on_exit=False,  # we do it manually
        ) as current_span:
            yield current_span

            for event in node.events:
                current_span.add_event(
                    name=event.name, attributes=event.attributes, timestamp=event.timestamp
                )

            for children in node.children:
                with self._build_tree_for_span(children):
                    pass

            current_span.set_status(node.status)
            if node.error is not None and node.status == StatusCode.ERROR:
                current_span.record_exception(node.error)

            current_span.end(_datetime_to_span_time(node.ended_at) if node.ended_at else None)

    @exception_handler
    async def _handler(self, data: Any, event: "EventMeta") -> None:
        if event.trace is None:
            return

        if event.trace.run_id not in self._processes:
            parent = (
                self._processes.get(event.trace.parent_run_id)
                if event.trace.parent_run_id
                else None
            )
            if event.trace.parent_run_id and not parent:
                raise ValueError(f"Parent run with ID {event.trace.parent_run_id} was not found!")

            self._processes_deps[event.trace.run_id] = []
            node = self._processes[event.trace.run_id] = ProcessorLocator.locate(data, event)
            if parent is not None:
                self._processes_deps[parent.run_id].append(node)
        else:
            node = self._processes[event.trace.run_id]

        from beeai_framework.context import RunContextFinishEvent

        if isinstance(data, RunContextFinishEvent):
            await node.end(data, event)
            if event.trace.parent_run_id is None:
                self._build_tree(node)
        else:
            if event.context.get("internal"):
                return

            await node.update(
                data,
                event,
            )
