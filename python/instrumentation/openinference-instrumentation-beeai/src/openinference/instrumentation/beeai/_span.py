from datetime import datetime
from typing import Any

from beeai_framework.emitter import EventMeta
from beeai_framework.utils.dicts import include_keys
from beeai_framework.utils.strings import to_json
from opentelemetry.sdk.trace import Event
from opentelemetry.trace import StatusCode

from instrumentation.beeai._utils import _datetime_to_span_time, _unpack_object
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
        self, name: str | None = None, event: tuple[Any, EventMeta] | None = None
    ) -> "SpanWrapper":
        child = SpanWrapper(name=name, kind=self.kind)
        if event is not None:
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
        attributes: dict[str, Any] = None,
        timestamp: datetime | None = None,
    ) -> None:
        self.events.append(
            Event(
                name=name,
                attributes=attributes,
                timestamp=_datetime_to_span_time(timestamp) if timestamp else None,
            )
        )

    def set_attribute(self, name: str, value: Any) -> None:
        self.attributes[name] = value

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        self.attributes.update(attributes)

    def set_status(self, status: StatusCode) -> None:
        self.status = status

    def record_exception(self, error: Exception):
        self.error = error
