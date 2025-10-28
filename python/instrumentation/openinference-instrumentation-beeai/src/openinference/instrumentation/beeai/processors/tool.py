from typing import Any, ClassVar

from beeai_framework.context import RunContext, RunContextStartEvent
from beeai_framework.emitter import EventMeta
from beeai_framework.tools import ToolErrorEvent, ToolRetryEvent, ToolStartEvent, ToolSuccessEvent
from beeai_framework.tools.tool import Tool
from typing_extensions import override

from openinference.instrumentation.beeai._utils import safe_dump_model_schema, stringify
from openinference.instrumentation.beeai.processors.base import Processor
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
)


class ToolProcessor(Processor):
    kind: ClassVar[OpenInferenceSpanKindValues] = OpenInferenceSpanKindValues.TOOL

    def __init__(self, event: "RunContextStartEvent", meta: "EventMeta") -> None:
        super().__init__(event, meta)

        assert isinstance(meta.creator, RunContext)
        assert isinstance(meta.creator.instance, Tool)

        tool = meta.creator.instance
        self.span.name = tool.name
        self.span.set_attributes(
            {
                SpanAttributes.TOOL_NAME: tool.name,
                SpanAttributes.TOOL_DESCRIPTION: tool.description,
                ToolAttributes.TOOL_JSON_SCHEMA: stringify(
                    safe_dump_model_schema(tool.input_schema)
                ),
            }
        )

    @override
    async def update(
        self,
        event: Any,
        meta: "EventMeta",
    ) -> None:
        await super().update(event, meta)

        self.span.add_event(f"{meta.name} ({meta.path})", timestamp=meta.created_at)

        match event:
            case ToolStartEvent():
                pass
            case None:  # finish event
                pass
            case ToolSuccessEvent():
                output_cls = type(event.output)

                self.span.reset_exception()
                self.span.set_attributes(
                    {
                        SpanAttributes.OUTPUT_VALUE: event.output.get_text_content(),
                        SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
                        f"{SpanAttributes.METADATA}.output_class": output_cls.__name__,
                        f"{SpanAttributes.METADATA}.is_empty": event.output.is_empty(),
                    }
                )
            case ToolErrorEvent():
                span = self.span.child(meta.name, event=(event, meta))
                span.record_exception(event.error)
                self.span.record_exception(event.error)
            case ToolRetryEvent():
                self.span.child(meta.name, event=(event, meta))
            case _:
                self.span.child(meta.name, event=(event, meta))
