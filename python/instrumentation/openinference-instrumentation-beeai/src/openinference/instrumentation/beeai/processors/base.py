from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from beeai_framework.context import RunContextFinishEvent, RunContextStartEvent
    from beeai_framework.emitter import EventMeta

from opentelemetry.trace import StatusCode

from openinference.instrumentation.beeai._span import SpanWrapper
from openinference.instrumentation.beeai._utils import stringify
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


class Processor:
    kind: ClassVar[OpenInferenceSpanKindValues] = OpenInferenceSpanKindValues.UNKNOWN

    def __init__(self, event: "RunContextStartEvent", meta: "EventMeta"):
        from beeai_framework.context import RunContext

        assert isinstance(meta.creator, RunContext)
        target_cls = type(meta.creator.instance)

        assert meta.trace is not None
        self.run_id = meta.trace.run_id

        self.span = SpanWrapper(name=target_cls.__name__, kind=type(self).kind)
        self.span.started_at = meta.created_at
        self.span.attributes.update(
            {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: type(self).kind,
                SpanAttributes.INPUT_VALUE: stringify(event.input),
                SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                f"{SpanAttributes.METADATA}.class_name": target_cls.__name__,
            }
        )

    async def update(
        self,
        event: Any,
        meta: "EventMeta",
    ) -> None:
        pass

    async def end(self, event: "RunContextFinishEvent", meta: "EventMeta") -> None:
        if event.error is not None:
            self.span.record_exception(event.error)

        if event.output is not None:
            if SpanAttributes.OUTPUT_VALUE not in self.span.attributes:
                self.span.attributes.update(
                    {
                        SpanAttributes.OUTPUT_VALUE: stringify(event.output),
                        SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                    }
                )
            self.span.set_status(StatusCode.OK)

        self.span.ended_at = meta.created_at
