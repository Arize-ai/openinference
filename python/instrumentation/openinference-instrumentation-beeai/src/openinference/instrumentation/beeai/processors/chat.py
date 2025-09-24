from datetime import datetime
from typing import Any, ClassVar

from beeai_framework.backend import (
    AnyMessage,
    ChatModel,
    MessageImageContent,
    MessageTextContent,
    MessageToolCallContent,
    MessageToolResultContent,
)
from beeai_framework.backend.events import (
    ChatModelStartEvent,
    ChatModelSuccessEvent,
)
from beeai_framework.context import RunContext, RunContextStartEvent
from beeai_framework.emitter import EventMeta
from beeai_framework.tools import AnyTool
from beeai_framework.utils.lists import remove_falsy
from typing_extensions import override

from openinference.instrumentation.beeai._utils import (
    _unpack_object,
    safe_dump_model_schema,
    stringify,
)
from openinference.instrumentation.beeai.processors.base import Processor
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


class ChatModelProcessor(Processor):
    kind: ClassVar[OpenInferenceSpanKindValues] = OpenInferenceSpanKindValues.LLM

    def __init__(self, event: RunContextStartEvent, meta: "EventMeta"):
        super().__init__(event, meta)

        self._last_updated_at = datetime.now()
        self._messages: dict[str, "AnyMessage"] = {}

        assert isinstance(meta.creator, RunContext)
        assert isinstance(meta.creator.instance, ChatModel)
        llm = meta.creator.instance
        self.span.set_attributes(
            {
                SpanAttributes.LLM_MODEL_NAME: llm.model_id,
                SpanAttributes.LLM_PROVIDER: llm.provider_id,
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

        if meta.name == "finish":
            return

        match event:
            case ChatModelStartEvent():
                assert isinstance(meta.creator, ChatModel)
                self._last_updated_at = meta.created_at
                self.span.set_attributes(
                    _process_messages(
                        event.input.messages,
                        prefix=SpanAttributes.LLM_INPUT_MESSAGES,
                    ),
                )
                self.span.set_attributes(
                    {
                        SpanAttributes.LLM_TOOLS: [t.name for t in (event.input.tools or [])],
                        SpanAttributes.LLM_INVOCATION_PARAMETERS: stringify(
                            meta.creator.parameters.model_dump(
                                exclude_none=True, exclude_unset=True
                            )
                            | event.input.model_dump(
                                exclude_none=True,
                                exclude_unset=True,
                                exclude={
                                    "tools",
                                    "messages",
                                },
                            ),
                        ),
                    }
                )

            case ChatModelSuccessEvent():
                if not self._messages:  # only when no streaming
                    self._add_new_messages(event.value.output)

                usage = event.value.usage
                if usage:
                    self.span.set_attributes(
                        {
                            SpanAttributes.LLM_TOKEN_COUNT_TOTAL: usage.total_tokens,
                            SpanAttributes.LLM_TOKEN_COUNT_PROMPT: usage.prompt_tokens,
                            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: usage.completion_tokens,
                        }
                    )

                cost = event.value.cost
                if cost:
                    self.span.set_attributes(
                        {
                            SpanAttributes.LLM_COST_COMPLETION: cost.completion_tokens_cost_usd,
                            SpanAttributes.LLM_COST_PROMPT: cost.prompt_tokens_usd,
                            SpanAttributes.LLM_COST_TOTAL: cost.total_cost_usd,
                        }
                    )

                self.span.set_attributes(
                    {
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: type(self).kind,
                        SpanAttributes.OUTPUT_VALUE: event.value.get_text_content(),
                        SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.TEXT.value,
                        f"{SpanAttributes.METADATA}.chunks_count": len(event.value.output),
                        **_unpack_object(
                            usage.model_dump(exclude_none=True) if usage else {},
                            prefix=f"{SpanAttributes.METADATA}.usage",
                        ),
                    }
                )
            case _:
                self.span.child(meta.name, event=(event, meta))

    def _add_new_messages(self, messages: list["AnyMessage"]) -> None:
        for new_msg in messages:
            msg_id = new_msg.meta.get("id") or f"latest_{new_msg.role}"
            if msg_id in self._messages:
                self._messages[msg_id].merge(new_msg)
            else:
                self._messages[msg_id] = new_msg.clone()

            _aggregate_msg_content(self._messages[msg_id])
            self.span.set_attributes(
                _process_messages(
                    [self._messages[msg_id]], prefix=SpanAttributes.LLM_OUTPUT_MESSAGES
                )
            )


def _process_tools(tools: list[AnyTool]) -> list[dict[str, str | Any]]:
    return [
        {
            SpanAttributes.TOOL_NAME: t.name,
            SpanAttributes.TOOL_DESCRIPTION: t.description,
            ToolAttributes.TOOL_JSON_SCHEMA: safe_dump_model_schema(t.input_schema),
        }
        for t in tools
    ]


def _process_messages(
    messages: list["AnyMessage"], prefix: str = "", offset: int = 0
) -> dict[str, Any]:
    if prefix and not prefix.endswith("."):
        prefix += "."

    output = {}
    for _i, msg in enumerate(messages):
        i = _i + offset
        output[f"{prefix}{i}.{MessageAttributes.MESSAGE_ROLE}"] = str(msg.role)

        output.update(
            _unpack_object(
                remove_falsy(
                    [
                        (
                            {
                                MessageContentAttributes.MESSAGE_CONTENT_TYPE: "text",
                                MessageContentAttributes.MESSAGE_CONTENT_TEXT: content.text,
                            }
                            if isinstance(content, MessageTextContent)
                            else {
                                MessageContentAttributes.MESSAGE_CONTENT_TYPE: "image",
                                MessageContentAttributes.MESSAGE_CONTENT_IMAGE: content.image_url[
                                    "url"
                                ],
                            }
                            if isinstance(content, MessageImageContent)
                            else {
                                MessageAttributes.MESSAGE_TOOL_CALL_ID: content.tool_call_id,
                                MessageContentAttributes.MESSAGE_CONTENT_TYPE: "text",
                                MessageContentAttributes.MESSAGE_CONTENT_TEXT: stringify(
                                    content.result, pretty=True
                                ),
                            }
                            if isinstance(content, MessageToolResultContent)
                            else None
                        )
                        for content in msg.content
                    ],
                ),
                prefix=f"{prefix}{i}.{MessageAttributes.MESSAGE_CONTENTS}",
            )
        )
        # )

        tool_calls: list[dict[str, Any]] = [
            {
                ToolCallAttributes.TOOL_CALL_ID: msg_call.id,
                ToolCallAttributes.TOOL_CALL_FUNCTION_NAME: msg_call.tool_name,
                ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON: msg_call.args,
            }
            for msg_call in msg.get_by_type(MessageToolCallContent)
        ]
        output.update(
            _unpack_object(tool_calls, prefix=f"{prefix}{i}.{MessageAttributes.MESSAGE_TOOL_CALLS}")
        )

    return output


def _aggregate_msg_content(message: "AnyMessage") -> None:
    from beeai_framework.backend import MessageTextContent, MessageToolCallContent

    contents = message.content.copy()
    aggregated_content: list[Any] = []

    for content in contents:
        last_content = aggregated_content[-1] if aggregated_content else None
        if isinstance(last_content, MessageTextContent) and isinstance(content, MessageTextContent):
            last_content.text += content.text
        elif isinstance(last_content, MessageToolCallContent) and isinstance(
            content, MessageToolCallContent
        ):
            last_content.args += content.args
        else:
            aggregated_content.append(content)

    message.content.clear()
    message.content.extend(aggregated_content)
