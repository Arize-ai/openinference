from typing import Any

from beeai_framework.context import RunContextStartEvent
from beeai_framework.emitter import EventMeta
from typing_extensions import override

from openinference.instrumentation.beeai._utils import _unpack_object, stringify
from openinference.instrumentation.beeai.processors.agents.base import AgentProcessor
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes


class ReActAgentProcessor(AgentProcessor):
    def __init__(self, event: "RunContextStartEvent", meta: "EventMeta"):
        super().__init__(event, meta)
        self._last_chunk_index = 0

    @override
    async def update(
        self,
        event: Any,
        meta: "EventMeta",
    ) -> None:
        from beeai_framework.agents.react import (
            ReActAgentErrorEvent,
            ReActAgentRetryEvent,
            ReActAgentStartEvent,
            ReActAgentSuccessEvent,
            ReActAgentUpdateEvent,
        )
        from beeai_framework.backend import UserMessage

        await super().update(event, meta)

        self.span.add_event(f"{meta.name} ({meta.path})", timestamp=meta.created_at)

        match event:
            case ReActAgentStartEvent():
                last_user_message: UserMessage | None = next(
                    (
                        msg
                        for msg in reversed(event.memory.messages)
                        if isinstance(msg, UserMessage)
                    ),
                    None,
                )
                self.span.set_attributes(
                    {
                        SpanAttributes.INPUT_VALUE: last_user_message.text
                        if last_user_message
                        else "",
                        SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
                    }
                )
            case ReActAgentUpdateEvent():
                value = event.update.parsed_value or event.update.value
                self.span.set_attribute(
                    f"{SpanAttributes.METADATA}.iteration", event.meta.iteration
                )
                self.span.set_attributes(
                    _unpack_object(
                        {event.update.key: stringify(value)},
                        prefix=f"{SpanAttributes.METADATA}.iterations.{event.meta.iteration}",
                    )
                )

            case ReActAgentErrorEvent():
                span = self.span.child(meta.name, event=(event, meta))
                span.record_exception(event.error)
            case ReActAgentRetryEvent():
                self.span.child(meta.name, event=(event, meta))
            case ReActAgentSuccessEvent():
                self.span.set_attribute(SpanAttributes.OUTPUT_VALUE, event.data.text)
                self.span.set_attribute(
                    SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.TEXT.value
                )
            case _:
                self.span.child(meta.name, event=(event, meta))
