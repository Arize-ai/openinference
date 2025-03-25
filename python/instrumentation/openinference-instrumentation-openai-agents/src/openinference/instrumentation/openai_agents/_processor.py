from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, Optional, Sequence, Union

from agents.tracing import Span, Trace, TracingProcessor
from agents.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
)
from openai.types.responses import (
    EasyInputMessageParam,
    FunctionTool,
    Response,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallParam,
    ResponseFunctionWebSearch,
    ResponseInputContentParam,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputMessageParam,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
    Tool,
)
from openai.types.responses.response_input_item_param import FunctionCallOutput, Message
from openai.types.responses.response_output_message_param import Content
from opentelemetry.context import attach, detach
from opentelemetry.trace import Span as OtelSpan
from opentelemetry.trace import (
    Status,
    StatusCode,
    Tracer,
    set_span_in_context,
)
from opentelemetry.util.types import AttributeValue
from typing_extensions import assert_never

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

logger = logging.getLogger(__name__)


class OpenInferenceTracingProcessor(TracingProcessor):  # type: ignore[misc]
    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer
        self._root_spans: dict[str, OtelSpan] = {}
        self._otel_spans: dict[str, OtelSpan] = {}
        self._tokens: dict[str, object] = {}

    def on_trace_start(self, trace: Trace) -> None:
        """Called when a trace is started.

        Args:
            trace: The trace that started.
        """
        otel_span = self._tracer.start_span(
            name=trace.name,
            attributes={
                OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            },
        )
        self._root_spans[trace.trace_id] = otel_span

    def on_trace_end(self, trace: Trace) -> None:
        """Called when a trace is finished.

        Args:
            trace: The trace that started.
        """
        if root_span := self._root_spans.pop(trace.trace_id, None):
            root_span.set_status(Status(StatusCode.OK))
            root_span.end()

    def on_span_start(self, span: Span[Any]) -> None:
        """Called when a span is started.

        Args:
            span: The span that started.
        """
        if not span.started_at:
            return
        start_time = datetime.fromisoformat(span.started_at)
        parent_span = (
            self._otel_spans.get(span.parent_id)
            if span.parent_id
            else self._root_spans.get(span.trace_id)
        )
        context = set_span_in_context(parent_span) if parent_span else None
        span_name = _get_span_name(span)
        otel_span = self._tracer.start_span(
            name=span_name,
            context=context,
            start_time=_as_utc_nano(start_time),
            attributes={
                OPENINFERENCE_SPAN_KIND: _get_span_kind(span.span_data),
                LLM_SYSTEM: OpenInferenceLLMSystemValues.OPENAI.value,
            },
        )
        self._otel_spans[span.span_id] = otel_span
        self._tokens[span.span_id] = attach(set_span_in_context(otel_span))

    def on_span_end(self, span: Span[Any]) -> None:
        """Called when a span is finished. Should not block or raise exceptions.

        Args:
            span: The span that finished.
        """
        if token := self._tokens.pop(span.span_id, None):
            detach(token)  # type: ignore[arg-type]
        if not (otel_span := self._otel_spans.pop(span.span_id, None)):
            return
        otel_span.update_name(_get_span_name(span))
        # flatten_attributes: dict[str, AttributeValue] = dict(_flatten(span.export()))
        # otel_span.set_attributes(flatten_attributes)
        data = span.span_data
        if isinstance(data, ResponseSpanData):
            if hasattr(data, "response") and isinstance(response := data.response, Response):
                otel_span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                otel_span.set_attribute(OUTPUT_VALUE, response.model_dump_json())
                for k, v in _get_attributes_from_response(response):
                    otel_span.set_attribute(k, v)
            if hasattr(data, "input") and (input := data.input):
                if isinstance(input, str):
                    otel_span.set_attribute(INPUT_VALUE, input)
                elif isinstance(input, list):
                    otel_span.set_attribute(INPUT_MIME_TYPE, JSON)
                    otel_span.set_attribute(INPUT_VALUE, safe_json_dumps(input))
                    for k, v in _get_attributes_from_input(input):
                        otel_span.set_attribute(k, v)
                elif TYPE_CHECKING:
                    assert_never(input)
        elif isinstance(data, GenerationSpanData):
            for k, v in _get_attributes_from_generation_span_data(data):
                otel_span.set_attribute(k, v)
        elif isinstance(data, FunctionSpanData):
            for k, v in _get_attributes_from_function_span_data(data):
                otel_span.set_attribute(k, v)
        end_time: Optional[int] = None
        if span.ended_at:
            try:
                end_time = _as_utc_nano(datetime.fromisoformat(span.ended_at))
            except ValueError:
                pass
        otel_span.end(end_time)

    def force_flush(self) -> None:
        """Forces an immediate flush of all queued spans/traces."""
        # TODO
        pass

    def shutdown(self) -> None:
        """Called when the application stops."""
        # TODO
        pass


def _as_utc_nano(dt: datetime) -> int:
    return int(dt.astimezone(timezone.utc).timestamp() * 1_000_000_000)


def _get_span_name(obj: Span) -> str:
    if hasattr(data := obj.span_data, "name") and isinstance(name := data.name, str):
        return name
    if isinstance(obj.span_data, HandoffSpanData) and obj.span_data.to_agent:
        return f"handoff to {obj.span_data.to_agent}"
    return obj.span_data.type  # type: ignore[no-any-return]


def _get_span_kind(obj: SpanData) -> str:
    if isinstance(obj, AgentSpanData):
        return OpenInferenceSpanKindValues.AGENT.value
    if isinstance(obj, FunctionSpanData):
        return OpenInferenceSpanKindValues.TOOL.value
    if isinstance(obj, GenerationSpanData):
        return OpenInferenceSpanKindValues.LLM.value
    if isinstance(obj, ResponseSpanData):
        return OpenInferenceSpanKindValues.LLM.value
    if isinstance(obj, HandoffSpanData):
        return OpenInferenceSpanKindValues.TOOL.value
    if isinstance(obj, CustomSpanData):
        return OpenInferenceSpanKindValues.CHAIN.value
    if isinstance(obj, GuardrailSpanData):
        return OpenInferenceSpanKindValues.CHAIN.value
    return OpenInferenceSpanKindValues.CHAIN.value


def _get_attributes_from_input(
    obj: Iterable[ResponseInputItemParam],
    msg_idx: int = 1,
) -> Iterator[tuple[str, AttributeValue]]:
    for item in obj:
        prefix = f"{LLM_INPUT_MESSAGES}.{msg_idx}."
        if "type" not in item:
            if "role" in item and "content" in item:
                yield from _get_attributes_from_message_param(
                    {
                        "type": "message",
                        "role": item["role"],
                        "content": item["content"],
                    },
                    prefix,
                )
            continue
        if item["type"] == "message":
            yield from _get_attributes_from_message_param(item, prefix)
        elif item["type"] == "file_search_call":
            continue
        elif item["type"] == "computer_call":
            continue
        elif item["type"] == "computer_call_output":
            continue
        elif item["type"] == "web_search_call":
            continue
        elif item["type"] == "function_call":
            yield from _get_attributes_from_response_function_tool_call_param(item, prefix)
        elif item["type"] == "function_call_output":
            yield from _get_attributes_from_function_call_output(item, prefix)
        elif item["type"] == "reasoning":
            continue
        elif item["type"] == "item_reference":
            continue
        elif TYPE_CHECKING:
            assert_never(item["type"])
        msg_idx += 1


def _get_attributes_from_message_param(
    obj: Union[
        EasyInputMessageParam,
        Message,
        ResponseOutputMessageParam,
    ],
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    yield f"{prefix}{MESSAGE_ROLE}", obj["role"]
    if content := obj.get("content"):
        if isinstance(content, str):
            yield f"{prefix}{MESSAGE_CONTENT}", content
        elif isinstance(content, list):
            yield from _get_attributes_from_message_content_list(content, prefix)


def _get_attributes_from_response_function_tool_call_param(
    obj: ResponseFunctionToolCallParam,
    prefix: str,
) -> Iterator[tuple[str, AttributeValue]]:
    yield f"{prefix}{MESSAGE_ROLE}", "assistant"
    yield f"{prefix}{TOOL_CALL_ID}", obj["call_id"]
    yield f"{prefix}{TOOL_CALL_FUNCTION_NAME}", obj["name"]
    if obj["arguments"] != "{}":
        yield f"{prefix}{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}", obj["arguments"]


def _get_attributes_from_function_call_output(
    obj: FunctionCallOutput,
    prefix: str,
) -> Iterator[tuple[str, AttributeValue]]:
    yield f"{prefix}{MESSAGE_ROLE}", "tool"
    yield f"{prefix}{MESSAGE_TOOL_CALL_ID}", obj["call_id"]
    yield f"{prefix}{MESSAGE_CONTENT}", obj["output"]


def _get_attributes_from_generation_span_data(
    obj: GenerationSpanData,
) -> Iterator[tuple[str, AttributeValue]]:
    if isinstance(model := obj.model, str):
        yield LLM_MODEL_NAME, model
    if isinstance(obj.model_config, dict) and (
        param := {k: v for k, v in obj.model_config.items() if v is not None}
    ):
        yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(param)
        if base_url := param.get("base_url"):
            if "api.openai.com" in base_url:
                yield LLM_PROVIDER, OpenInferenceLLMProviderValues.OPENAI.value
    yield from _get_attributes_from_chat_completions_input(obj.input)
    yield from _get_attributes_from_chat_completions_output(obj.output)
    yield from _get_attributes_from_chat_completions_usage(obj.usage)


def _get_attributes_from_chat_completions_input(
    obj: Optional[Sequence[Mapping[str, Any]]],
) -> Iterator[tuple[str, AttributeValue]]:
    if not obj:
        return
    try:
        yield INPUT_VALUE, safe_json_dumps(obj)
        yield INPUT_MIME_TYPE, JSON
    except Exception:
        pass
    yield from _get_attributes_from_chat_completions_message_dicts(
        obj,
        f"{LLM_INPUT_MESSAGES}.",
    )


def _get_attributes_from_chat_completions_output(
    obj: Optional[Sequence[Mapping[str, Any]]],
) -> Iterator[tuple[str, AttributeValue]]:
    if not obj:
        return
    try:
        yield OUTPUT_VALUE, safe_json_dumps(obj)
        yield OUTPUT_MIME_TYPE, JSON
    except Exception:
        pass
    yield from _get_attributes_from_chat_completions_message_dicts(
        obj,
        f"{LLM_OUTPUT_MESSAGES}.",
    )


def _get_attributes_from_chat_completions_message_dicts(
    obj: Iterable[Mapping[str, Any]],
    prefix: str,
    msg_idx: int = 0,
    tool_call_idx: int = 0,
) -> Iterator[tuple[str, AttributeValue]]:
    if not isinstance(obj, Iterable):
        return
    for msg in obj:
        if isinstance(role := msg.get("role"), str):
            yield f"{prefix}{msg_idx}.{MESSAGE_ROLE}", role
        if content := msg.get("content"):
            yield from _get_attributes_from_chat_completions_message_content(
                content,
                f"{prefix}{msg_idx}.",
            )
        if isinstance(tool_call_id := msg.get("tool_call_id"), str):
            yield f"{prefix}{msg_idx}.{MESSAGE_TOOL_CALL_ID}", tool_call_id
        if isinstance(tool_calls := msg.get("tool_calls"), Iterable):
            for tc in tool_calls:
                yield from _get_attributes_from_chat_completions_tool_call_dict(
                    tc,
                    f"{prefix}{msg_idx}.{MESSAGE_TOOL_CALLS}.{tool_call_idx}.",
                )
                tool_call_idx += 1
        msg_idx += 1


def _get_attributes_from_chat_completions_message_content(
    obj: Union[str, list[Mapping[str, Any]]],
    prefix: str,
) -> Iterator[tuple[str, AttributeValue]]:
    if isinstance(obj, str):
        yield f"{prefix}{MESSAGE_CONTENT}", obj
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _get_attributes_from_chat_completions_message_content_item(
                item,
                f"{prefix}{MESSAGE_CONTENTS}.{i}.",
            )


def _get_attributes_from_chat_completions_message_content_item(
    obj: Mapping[str, Any],
    prefix: str,
) -> Iterator[tuple[str, AttributeValue]]:
    if obj.get("type") == "text" and (text := obj.get("text")):
        yield f"{prefix}{MESSAGE_CONTENT_TYPE}", "text"
        yield f"{prefix}{MESSAGE_CONTENT_TEXT}", text


def _get_attributes_from_chat_completions_tool_call_dict(
    obj: Mapping[str, Any],
    prefix: str,
) -> Iterator[tuple[str, AttributeValue]]:
    if id_ := obj.get("id"):
        yield f"{prefix}{TOOL_CALL_ID}", id_
    if function := obj.get("function"):
        if name := function.get("name"):
            yield f"{prefix}{TOOL_CALL_FUNCTION_NAME}", name
        if arguments := function.get("arguments"):
            if arguments != "{}":
                yield f"{prefix}{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}", arguments


def _get_attributes_from_chat_completions_usage(
    obj: Optional[Mapping[str, Any]],
) -> Iterator[tuple[str, AttributeValue]]:
    if not obj:
        return
    if input_tokens := obj.get("input_tokens"):
        yield LLM_TOKEN_COUNT_PROMPT, input_tokens
    if output_tokens := obj.get("output_tokens"):
        yield LLM_TOKEN_COUNT_COMPLETION, output_tokens


def _get_attributes_from_function_span_data(
    obj: FunctionSpanData,
) -> Iterator[tuple[str, AttributeValue]]:
    yield TOOL_NAME, obj.name
    if obj.input:
        yield INPUT_VALUE, obj.input
        yield INPUT_MIME_TYPE, JSON
    if obj.output is not None:
        yield OUTPUT_VALUE, obj.output
        if isinstance(obj.output, str) and obj.output[0] == "{" and obj.output[-1] == "}":
            yield OUTPUT_MIME_TYPE, JSON


def _get_attributes_from_message_content_list(
    obj: Iterable[Union[ResponseInputContentParam, Content]],
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    for i, item in enumerate(obj):
        if item["type"] == "input_text" or item["type"] == "output_text":
            yield f"{prefix}{MESSAGE_CONTENTS}.{i}.{MESSAGE_CONTENT_TYPE}", "text"
            yield f"{prefix}{MESSAGE_CONTENTS}.{i}.{MESSAGE_CONTENT_TEXT}", item["text"]
        elif item["type"] == "input_image":
            # TODO
            ...
        elif item["type"] == "input_file":
            # TODO
            ...
        elif item["type"] == "refusal":
            # TODO
            ...
        elif TYPE_CHECKING:
            assert_never(item["type"])


def _get_attributes_from_response(obj: Response) -> Iterator[tuple[str, AttributeValue]]:
    yield from _get_attributes_from_tools(obj.tools)
    yield from _get_attributes_from_usage(obj.usage)
    yield from _get_attributes_from_response_output(obj.output)
    yield from _get_attributes_from_response_instruction(obj.instructions)
    yield LLM_MODEL_NAME, obj.model
    param = obj.model_dump(exclude_none=True)
    param.pop("tools", None)
    param.pop("usage", None)
    param.pop("output", None)
    yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(param)


def _get_attributes_from_tools(
    tools: Optional[Iterable[Tool]],
) -> Iterator[tuple[str, AttributeValue]]:
    if not tools:
        return
    for i, tool in enumerate(tools):
        if isinstance(tool, FunctionTool):
            yield (
                f"{LLM_TOOLS}.{i}.{TOOL_JSON_SCHEMA}",
                safe_json_dumps(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                            "strict": tool.strict,
                        },
                    }
                ),
            )
        else:
            pass


def _get_attributes_from_response_output(
    obj: Iterable[ResponseOutputItem],
    msg_idx: int = 0,
) -> Iterator[tuple[str, AttributeValue]]:
    tool_call_idx = 0
    for i, item in enumerate(obj):
        if isinstance(item, ResponseOutputMessage):
            prefix = f"{LLM_OUTPUT_MESSAGES}.{msg_idx}."
            yield from _get_attributes_from_message(item, prefix)
            msg_idx += 1
        elif isinstance(item, ResponseFunctionToolCall):
            yield f"{LLM_OUTPUT_MESSAGES}.{msg_idx}.{MESSAGE_ROLE}", "assistant"
            prefix = f"{LLM_OUTPUT_MESSAGES}.{msg_idx}.{MESSAGE_TOOL_CALLS}.{tool_call_idx}."
            yield from _get_attributes_from_function_tool_call(item, prefix)
            tool_call_idx += 1
        elif isinstance(item, ResponseFileSearchToolCall):
            ...
        elif isinstance(item, ResponseFunctionWebSearch):
            ...
        elif isinstance(item, ResponseComputerToolCall):
            ...
        elif isinstance(item, ResponseReasoningItem):
            ...
        elif TYPE_CHECKING:
            assert_never(item)


def _get_attributes_from_response_instruction(
    instructions: Optional[str],
) -> Iterator[tuple[str, AttributeValue]]:
    if not instructions:
        return
    yield f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}", "system"
    yield f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}", instructions


def _get_attributes_from_function_tool_call(
    obj: ResponseFunctionToolCall,
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    yield f"{prefix}{TOOL_CALL_ID}", obj.call_id
    yield f"{prefix}{TOOL_CALL_FUNCTION_NAME}", obj.name
    if obj.arguments != "{}":
        yield f"{prefix}{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}", obj.arguments


def _get_attributes_from_message(
    obj: ResponseOutputMessage,
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    yield f"{prefix}{MESSAGE_ROLE}", obj.role
    for i, item in enumerate(obj.content):
        if isinstance(item, ResponseOutputText):
            yield f"{prefix}{MESSAGE_CONTENTS}.{i}.{MESSAGE_CONTENT_TYPE}", "text"
            yield f"{prefix}{MESSAGE_CONTENTS}.{i}.{MESSAGE_CONTENT_TEXT}", item.text
        elif isinstance(item, ResponseOutputRefusal):
            ...
        elif TYPE_CHECKING:
            assert_never(item)


def _get_attributes_from_usage(
    obj: Optional[ResponseUsage],
) -> Iterator[tuple[str, AttributeValue]]:
    if not obj:
        return
    yield LLM_TOKEN_COUNT_COMPLETION, obj.input_tokens
    yield LLM_TOKEN_COUNT_PROMPT, obj.output_tokens
    yield LLM_TOKEN_COUNT_TOTAL, obj.total_tokens


def _flatten(
    obj: Mapping[str, Any],
    prefix: str = "",
) -> Iterator[tuple[str, AttributeValue]]:
    for key, value in obj.items():
        if isinstance(value, dict):
            yield from _flatten(value, f"{prefix}{key}.")
        elif isinstance(value, (str, int, float, bool, str)):
            yield key, value
        else:
            yield key, str(value)


INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOOLS = SpanAttributes.LLM_TOOLS
METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS

MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID

TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID

TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA

JSON = OpenInferenceMimeTypeValues.JSON.value
