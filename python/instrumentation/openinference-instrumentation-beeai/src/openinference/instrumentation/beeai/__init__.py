# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Callable, Collection, TypeVar

from beeai_framework.context import (
    RunContextFinishEvent,
)
from beeai_framework.emitter import Emitter, EmitterOptions, EventMeta
from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
)

from ._span import SpanWrapper
from ._utils import _datetime_to_span_time, exception_handler
from .processors.base import Processor
from .processors.locator import init_processor

logger = logging.getLogger(__name__)

_instruments = ("beeai-framework >= 0.1.31",)
try:
    __version__ = version("beeai-framework")
except PackageNotFoundError:
    __version__ = "unknown"

EventPair = tuple[Any, EventMeta]


class BeeAIInstrumentor(BaseInstrumentor):  # type: ignore
    __slots__ = ("_tracer", "_cleanup", "_processes")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cleanup: Callable[[], None] = lambda: None
        self._processes: dict[str, Processor] = {}

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        self._uninstrument()

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
        self._cleanup = Emitter.root().match(
            "*.*",
            self._handler,
            EmitterOptions(match_nested=True, is_blocking=True),
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        self._cleanup()
        self._processes.clear()

    def _build_tree(self, node: SpanWrapper) -> None:
        with self._tracer.start_as_current_span(
            name=node.name,
            openinference_span_kind=node.kind,
            attributes=node.attributes,
            start_time=_datetime_to_span_time(node.started_at) if node.started_at else None,
            end_on_exit=False,  # we do it manually
        ) as current_span:
            for event in node.events:
                current_span.add_event(
                    name=event.name, attributes=event.attributes, timestamp=event.timestamp
                )

            for children in node.children:
                self._build_tree(children)

            current_span.set_status(node.status)
            if node.error is not None:
                current_span.record_exception(node.error)

            current_span.end(_datetime_to_span_time(node.ended_at) if node.ended_at else None)

    @exception_handler
    async def _handler(self, data: Any, event: EventMeta) -> None:
        assert event.trace is not None, "Event must have a trace"

        if event.trace.run_id not in self._processes:
            parent = (
                self._processes.get(event.trace.parent_run_id)
                if event.trace.parent_run_id
                else None
            )
            if event.trace.parent_run_id and not parent:
                raise ValueError(f"Parent run with ID {event.trace.parent_run_id} was not found!")

            node = self._processes[event.trace.run_id] = init_processor(data, event)
            if parent is not None:
                parent.span.children.append(node.span)
        else:
            node = self._processes[event.trace.run_id]

        if isinstance(data, RunContextFinishEvent):
            await node.end(data, event)
            self._build_tree(node.span)
            self._processes.pop(event.trace.run_id)
        else:
            if event.context.get("internal"):
                return

            await node.update(
                data,
                event,
            )
