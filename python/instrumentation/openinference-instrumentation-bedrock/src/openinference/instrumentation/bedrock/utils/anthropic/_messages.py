from __future__ import annotations

import json
import logging
from dataclasses import asdict
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from dacite import from_dict
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue
from typing_extensions import assert_never

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.bedrock.__generated__.anthropic._types import (
    InputJSONDelta,
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    RawMessageStreamEvent,
    TextBlock,
    TextDelta,
    ToolUseBlock,
)
from openinference.instrumentation.bedrock.utils import _finish
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from anthropic.types import (
        DocumentBlockParam,
        ImageBlockParam,
        MessageParam,
        TextBlockParam,
        ToolResultBlockParam,
        ToolUseBlockParam,
    )

_AnyT = TypeVar("_AnyT")

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _AnthropicMessagesCallback:
    def __init__(
        self,
        span: Span,
        request: Mapping[str, Any],
    ) -> None:
        self._span = span
        self._snapshot: Optional[Message] = None
        self._request_attributes = dict(_attributes_from_request(request))

    def __call__(self, obj: _AnyT) -> _AnyT:
        span = self._span
        if isinstance(obj, dict):
            if "chunk" in obj and "bytes" in obj["chunk"]:
                payload = json.loads(obj["chunk"]["bytes"])
                if payload.get("type") == "message_stop" and isinstance(
                    metrics := payload.get("amazon-bedrock-invocationMetrics"), dict
                ):
                    if isinstance(v := metrics.get("inputTokenCount"), int):
                        span.set_attribute(LLM_TOKEN_COUNT_PROMPT, v)
                    if isinstance(v := metrics.get("outputTokenCount"), int):
                        span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, v)
                try:
                    self._snapshot = accumulate_event(
                        event=parse_event(payload),
                        current_snapshot=self._snapshot,
                    )
                except BaseException as e:
                    logger.warning(str(e))
                    pass
        elif isinstance(obj, (StopIteration, StopAsyncIteration)):
            if (message := self._snapshot) is None:
                _finish(span, None, self._request_attributes)
            else:
                for k, v in _attributes_from_message(message, f"{LLM_OUTPUT_MESSAGES}.{0}."):
                    span.set_attribute(k, v)
                _finish(span, asdict(message), self._request_attributes)
        elif isinstance(obj, BaseException):
            _finish(span, obj, self._request_attributes)
        return obj


def _stop_on_exception(
    wrapped: Callable[..., Iterator[tuple[str, Any]]],
) -> Callable[..., Iterator[tuple[str, Any]]]:
    def wrapper(*args: Any, **kwargs: Any) -> Iterator[tuple[str, Any]]:
        try:
            yield from wrapped(*args, **kwargs)
        except Exception as e:
            logger.warning(str(e))

    return wrapper


@_stop_on_exception
def _attributes_from_message_param(
    message: MessageParam,
    prefix: str,
    role: Optional[str] = None,
) -> Iterator[tuple[str, AttributeValue]]:
    if role:
        yield f"{prefix}{MESSAGE_ROLE}", role
    elif (
        isinstance(message["content"], list)
        and message["content"]
        and any(block["type"] == "tool_result" for block in message["content"])
    ):
        yield f"{prefix}{MESSAGE_ROLE}", "tool"
    else:
        try:
            yield f"{prefix}{MESSAGE_ROLE}", message["role"]
        except KeyError:
            pass
    if not (content := message.get("content")):
        return
    if isinstance(content, str):
        yield f"{prefix}{MESSAGE_CONTENTS}.{0}.{MESSAGE_CONTENT_TEXT}", content
        return
    if not isinstance(content, Iterable):
        return
    num_tool_calls = 0
    for i, block in enumerate(content):
        if TYPE_CHECKING:
            assert isinstance(block, dict)
        if "type" not in block:
            continue
        if block["type"] == "text":
            try:
                yield f"{prefix}{MESSAGE_CONTENTS}.{i}.{MESSAGE_CONTENT_TEXT}", block["text"]
            except KeyError:
                pass
        elif block["type"] == "tool_use":
            yield from _attributes_from_tool_call_param(
                block,
                f"{prefix}{MESSAGE_TOOL_CALLS}.{num_tool_calls}.",
            )
            num_tool_calls += 1
        elif block["type"] == "image":
            yield from _attributes_from_image_param(block, f"{prefix}{MESSAGE_CONTENTS}.{i}.")
        elif block["type"] == "tool_result":
            yield from _attributes_from_tool_result_param(block, f"{prefix}{MESSAGE_CONTENTS}.{i}.")
        elif block["type"] == "document":
            yield from _attributes_from_document_param(block, f"{prefix}{MESSAGE_CONTENTS}.{i}.")
        elif block["type"] == "thinking":
            pass  # TODO
        elif block["type"] == "redacted_thinking":
            pass  # TODO
        else:
            if TYPE_CHECKING:
                assert_never(block)  # type: ignore[arg-type]


@_stop_on_exception
def _attributes_from_message(
    message: Message,
    prefix: str,
    num_tool_calls: int = 0,
) -> Iterator[tuple[str, AttributeValue]]:
    if role := message.role:
        yield f"{prefix}{MESSAGE_ROLE}", role.value if isinstance(role, Enum) else role
    for block in message.content:
        if isinstance(block, TextBlock):
            yield f"{prefix}{MESSAGE_CONTENTS}.{0}.{MESSAGE_CONTENT_TEXT}", block.text
        elif isinstance(block, ToolUseBlock):
            yield from _attributes_from_tool_call(
                block, f"{prefix}{MESSAGE_TOOL_CALLS}.{num_tool_calls}."
            )
            num_tool_calls += 1
        else:
            if TYPE_CHECKING:
                assert_never(block)
    usage = message.usage
    # See https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#tracking-cache-performance
    # cache_creation_input_tokens: Number of tokens written to the cache when creating a new entry.
    # cache_read_input_tokens: Number of tokens retrieved from the cache for this request.
    # input_tokens: Number of input tokens which were not read from or used to create a cache.
    if prompt_tokens := (
        usage.input_tokens
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    ):
        yield LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if usage.output_tokens:
        yield LLM_TOKEN_COUNT_COMPLETION, usage.output_tokens


@_stop_on_exception
def _attributes_from_system_message(
    system: Union[str, Iterable[TextBlockParam]],
    prefix: str,
) -> Iterator[Tuple[str, AttributeValue]]:
    yield f"{prefix}{MESSAGE_ROLE}", "system"
    if isinstance(system, str):
        yield f"{prefix}{MESSAGE_CONTENTS}.{0}.{MESSAGE_CONTENT_TEXT}", system
        return
    for i, block in enumerate(system):
        try:
            yield f"{MESSAGE_CONTENTS}.{i}.{MESSAGE_CONTENT_TEXT}", block["text"]
        except KeyError:
            pass


@_stop_on_exception
def _attributes_from_tool_call_param(
    block: ToolUseBlockParam,
    prefix: str,
) -> Iterator[Tuple[str, AttributeValue]]:
    try:
        yield f"{prefix}{TOOL_CALL_ID}", block["id"]
    except KeyError:
        pass
    try:
        yield f"{prefix}{TOOL_CALL_FUNCTION_NAME}", block["name"]
    except KeyError:
        pass
    try:
        if isinstance(block["input"], str):
            args = block["input"]
        else:
            args = safe_json_dumps(block["input"])
        yield f"{prefix}{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}", args
    except KeyError:
        pass


@_stop_on_exception
def _attributes_from_tool_call(
    block: ToolUseBlock,
    prefix: str,
) -> Iterator[Tuple[str, AttributeValue]]:
    yield f"{prefix}{TOOL_CALL_ID}", block.id
    yield f"{prefix}{TOOL_CALL_FUNCTION_NAME}", block.name
    if isinstance(block.input, str):
        args = block.input
    else:
        args = safe_json_dumps(block.input)
    yield f"{prefix}{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}", args


@_stop_on_exception
def _attributes_from_tool(
    tool: Mapping[str, Any],
    prefix: str,
) -> Iterator[tuple[str, AttributeValue]]:
    yield f"{prefix}{TOOL_JSON_SCHEMA}", safe_json_dumps(tool)


@_stop_on_exception
def _attributes_from_image_param(
    block: ImageBlockParam,
    prefix: str,
) -> Iterator[Tuple[str, AttributeValue]]:
    source = block["source"]
    if source["type"] == "base64":
        media_type = source["media_type"]
        data = source["data"]
        type_ = source["type"]
        yield (
            f"{prefix}{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}",
            f"data:{media_type};{type_},{data}",
        )
    elif source["type"] == "url":
        yield (
            f"{prefix}{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}",
            source["url"],
        )
    elif TYPE_CHECKING:
        assert_never(source["type"])


@_stop_on_exception
def _attributes_from_tool_result_param(
    block: ToolResultBlockParam,
    prefix: str,
) -> Iterator[Tuple[str, AttributeValue]]:
    content = block.get("content")
    if isinstance(content, str):
        yield f"{prefix}{MESSAGE_CONTENT_TEXT}", content
    elif isinstance(content, list):
        texts = [b["text"] for b in content if b["type"] == "text"]
        yield f"{prefix}{MESSAGE_CONTENT_TEXT}", "\n\n".join(texts)


@_stop_on_exception
def _attributes_from_document_param(
    block: DocumentBlockParam,
    prefix: str,
) -> Iterator[Tuple[str, AttributeValue]]:
    yield from ()


@_stop_on_exception
def _attributes_from_request(
    request: Mapping[str, Any],
) -> Iterator[tuple[str, AttributeValue]]:
    try:
        yield LLM_MODEL_NAME, request["modelId"]
    except KeyError:
        pass
    num_messages = 0
    body = request["body"]
    if system := body.get("system"):
        yield from _attributes_from_system_message(system, f"{LLM_INPUT_MESSAGES}.0.")
        num_messages += 1
    if isinstance(messages := body.get("messages"), list):
        for i, message in enumerate(messages, num_messages):
            yield from _attributes_from_message_param(message, f"{LLM_INPUT_MESSAGES}.{i}.")
    yield LLM_INVOCATION_PARAMETERS, _invocation_parameters(body)
    if tools := body.get("tools"):
        for i, tool in enumerate(tools):
            yield from _attributes_from_tool(tool, f"{LLM_TOOLS}.{i}.")
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(body)
    yield OPENINFERENCE_SPAN_KIND, LLM


def _invocation_parameters(
    request: Mapping[str, Any],
) -> str:
    parameters = dict(request)
    parameters.pop("tools", None)
    parameters.pop("messages", None)
    parameters.pop("system", None)
    parameters.pop("anthropic_version", None)
    return safe_json_dumps(parameters)


IMAGE_URL = ImageAttributes.IMAGE_URL
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
JSON = OpenInferenceMimeTypeValues.JSON.value
LLM = OpenInferenceSpanKindValues.LLM.value
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOOLS = SpanAttributes.LLM_TOOLS
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA


def accumulate_event(
    *,
    event: RawMessageStreamEvent,
    current_snapshot: Optional[Message] = None,
) -> Message:
    """
    Based on
    https://github.com/anthropics/anthropic-sdk-python/blob/0f9ccca8e26cf27b969e38c02899fde4b3489c86/src/anthropic/lib/streaming/_messages.py#L370
    """
    if current_snapshot is None:
        if isinstance(event, RawMessageStartEvent):
            return event.message
        raise RuntimeError(f'Unexpected event order, got {event.type} before "message_start"')
    if isinstance(event, RawContentBlockStartEvent):
        # TODO: check index
        current_snapshot.content.append(event.content_block)
    elif isinstance(event, RawContentBlockDeltaEvent):
        content = current_snapshot.content[event.index]
        if isinstance(content, TextBlock) and isinstance(event.delta, TextDelta):
            content.text += event.delta.text
        elif isinstance(content, ToolUseBlock) and isinstance(event.delta, InputJSONDelta):
            if not isinstance(content.input, str):
                content.input = ""
            content.input += event.delta.partial_json
    elif isinstance(event, RawMessageDeltaEvent):
        current_snapshot.stop_reason = event.delta.stop_reason
        current_snapshot.stop_sequence = event.delta.stop_sequence
        current_snapshot.usage.output_tokens = event.usage.output_tokens
    return current_snapshot


def parse_event(obj: dict[str, Any]) -> RawMessageStreamEvent:
    if not isinstance(type_ := obj.get("type"), str):
        raise ValueError(f"Unknown event: {obj}")
    if type_ == "message_start":
        return from_dict(RawMessageStartEvent, obj)
    if type_ == "message_delta":
        return from_dict(RawMessageDeltaEvent, obj)
    if type_ == "message_stop":
        return from_dict(RawMessageStopEvent, obj)
    if type_ == "content_block_start":
        return from_dict(RawContentBlockStartEvent, obj)
    if type_ == "content_block_delta":
        return from_dict(RawContentBlockDeltaEvent, obj)
    if type_ == "content_block_stop":
        return from_dict(RawContentBlockStopEvent, obj)
    raise ValueError(f"Unknown event: {obj}")
