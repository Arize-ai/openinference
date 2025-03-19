from __future__ import annotations

import logging
from typing import (
    Any,
    Mapping,
    TypeVar,
)

from opentelemetry.trace import Span, Tracer

from openinference.instrumentation.bedrock.processerors import process_trace_event
from openinference.instrumentation.bedrock.utils import TraceContext, _finish
from openinference.semconv.trace import SpanAttributes

_AnyT = TypeVar("_AnyT")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _MessagesCallback:
    def __init__(
        self, span: Span, tracer: Tracer, request: Mapping[str, Any], idx: int = 0
    ) -> None:
        self._span = span
        self._request_attributes = request
        self.idx = idx
        self.trace_context = TraceContext()
        self.trace_processing_data: dict[str, Any] = {}
        self.tracer = tracer
        self.current_orchestration_data = {"span": None, "trace_id": None}

    def __call__(self, obj: _AnyT) -> _AnyT:
        try:
            span = self._span
            self.idx += 1
            if isinstance(obj, dict):
                if "chunk" in obj:
                    if "bytes" in obj["chunk"]:
                        output_text = obj["chunk"]["bytes"].decode("utf-8")
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_text)
                        span.set_attribute(f"chunk.{self.idx}.content", output_text)
                elif "trace" in obj:
                    process_trace_event(obj["trace"]["trace"], span, self)
            elif isinstance(obj, (StopIteration, StopAsyncIteration)):
                _finish(span, None, self._request_attributes)
            elif isinstance(obj, BaseException):
                _finish(span, obj, self._request_attributes)
        except Exception as e:
            logger.exception(e)
            logger.error(str(e))
        return obj
