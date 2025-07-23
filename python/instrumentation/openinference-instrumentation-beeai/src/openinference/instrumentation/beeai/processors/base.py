from typing import Any, ClassVar

from beeai_framework.context import RunContext, RunContextFinishEvent, RunContextStartEvent
from beeai_framework.emitter import EventMeta
from beeai_framework.utils.strings import to_json
from opentelemetry.trace import StatusCode

from instrumentation.beeai._span import SpanWrapper
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

StartEventPair = [RunContextStartEvent, EventMeta]


class Processor:
    kind: ClassVar[OpenInferenceSpanKindValues] = OpenInferenceSpanKindValues.UNKNOWN

    def __init__(self, event: RunContextStartEvent, meta: EventMeta):
        assert isinstance(meta.creator, RunContext)
        target_cls = type(meta.creator.instance)

        self.span = SpanWrapper(name=target_cls.__name__, kind=type(self).kind)
        self.span.started_at = meta.created_at
        self.span.attributes.update(
            {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: type(self).kind,
                SpanAttributes.INPUT_VALUE: to_json(
                    event.input, exclude_none=True, sort_keys=False
                ),
                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                f"{SpanAttributes.METADATA}.class_name": target_cls.__name__,
            }
        )

    async def update(
        self,
        event: Any,
        meta: EventMeta,
    ) -> None:
        pass

    async def end(self, event: RunContextFinishEvent, meta: EventMeta) -> None:
        if event.output is not None:
            if SpanAttributes.OUTPUT_VALUE not in self.span.attributes:
                self.span.attributes.update(
                    {
                        SpanAttributes.OUTPUT_VALUE: to_json(
                            event.output, exclude_none=True, sort_keys=False
                        ),
                        SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                    }
                )
            self.span.set_status(StatusCode.OK)

        if event.error is not None:
            self.span.set_status(StatusCode.ERROR)
            self.span.record_exception(event.error)

        self.span.ended_at = meta.created_at
