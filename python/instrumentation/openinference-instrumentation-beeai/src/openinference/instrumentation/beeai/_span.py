from datetime import datetime
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace import Event
from opentelemetry.trace import StatusCode

from openinference.instrumentation.beeai._utils import _datetime_to_span_time, _unpack_object

if TYPE_CHECKING:
    from beeai_framework.emitter import EventMeta

from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


class SpanWrapper:
    def __init__(self, *, name: str, kind: OpenInferenceSpanKindValues):
        self.name = name
        self.attributes: dict[str, Any] = {SpanAttributes.OPENINFERENCE_SPAN_KIND: kind}
        self.events: list[Event] = []
        self.status: StatusCode = StatusCode.OK
        self.error: Exception | None = None
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self.children: list["SpanWrapper"] = []
        self.kind = kind

    def child(
        self, name: str | None = None, event: tuple[Any, "EventMeta"] | None = None
    ) -> "SpanWrapper":
        child = SpanWrapper(name=name or f"{self.name}_child", kind=self.kind)
        if event is not None:
            from beeai_framework.utils.dicts import include_keys
            from beeai_framework.utils.strings import to_json

            value, meta = event

            child.started_at = meta.created_at
            child.ended_at = meta.created_at
            child.attributes.update(
                {
                    SpanAttributes.INPUT_VALUE: to_json(value) if value is not None else value,
                    SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
                    **_unpack_object(
                        include_keys(meta.model_dump(), {"id", "context", "path", "trace"}),
                        prefix=SpanAttributes.METADATA,
                    ),
                }
            )

        self.children.append(child)
        return child

    def add_event(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        self.events.append(
            Event(
                name=name,
                attributes=attributes or {},
                timestamp=_datetime_to_span_time(timestamp) if timestamp else None,
            )
        )

    def set_attribute(self, name: str, value: Any) -> None:
        self.attributes[name] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.attributes.update(attributes)

    def set_status(self, status: StatusCode) -> None:
        self.status = status

    def reset_exception(self) -> None:
        self.error = None
        self.set_status(StatusCode.OK)

    def record_exception(self, error: Exception) -> None:
        from beeai_framework.errors import FrameworkError

        self.error = error
        self.set_status(StatusCode.ERROR)
        self.set_attributes(
            {
                SpanAttributes.OUTPUT_VALUE: error.explain()
                if isinstance(error, FrameworkError)
                else str(error),
                SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
            }
        )
