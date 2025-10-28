from typing import Any

from beeai_framework.agents.tool_calling import (
    ToolCallingAgentStartEvent,
    ToolCallingAgentSuccessEvent,
)
from beeai_framework.emitter import EventMeta
from typing_extensions import override

from openinference.instrumentation.beeai.processors.agents.base import AgentProcessor
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes


class ToolCallingAgentProcessor(AgentProcessor):
    @override
    async def update(
        self,
        event: Any,
        meta: "EventMeta",
    ) -> None:
        await super().update(event, meta)
        self.span.add_event(f"{meta.name} ({meta.path})", timestamp=meta.created_at)

        match event:
            case ToolCallingAgentStartEvent():
                pass
            case ToolCallingAgentSuccessEvent():
                if event.state.result is not None:
                    self.span.set_attribute(SpanAttributes.OUTPUT_VALUE, event.state.result.text)
                    self.span.set_attribute(
                        SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.TEXT.value
                    )
            case _:
                self.span.child(meta.name, event=(event, meta))
