"""
Span attributes for the content blocks of Anthropic messages.

Shared by the input messages and by the output message of the streaming and non-streaming paths,
so that every path maps every block type the same way. Blocks can be request params (dicts) or
response models (e.g. a response passed back as input), so both are read with `_get`.
"""

import logging
from typing import Any, Callable, Iterable, Iterator, List, Mapping, Tuple

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Blocks in which the model, or the server on its behalf, calls a tool. An `mcp_tool_use` block
# keeps the tool's own name, so the tool call matches what the model emitted; its MCP server's
# name is recorded in the span metadata (see `_get_response_metadata`).
TOOL_CALL_BLOCK_TYPES = frozenset({"tool_use", "server_tool_use", "mcp_tool_use"})

_MISSING = object()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _require(obj: Any, key: str) -> Any:
    if (value := _get(obj, key, _MISSING)) is _MISSING:
        raise KeyError(key)
    return value


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        # by_alias, so e.g. a fallback block's `from_` is recorded as the API's `from`, and
        # without warnings, which the SDK's stand-in model for an unknown block type would raise
        return obj.model_dump(mode="json", by_alias=True, exclude_none=True, warnings=False)
    return obj


def _to_json(obj: Any) -> str:
    return safe_json_dumps(_to_jsonable(obj))


def _attributes_or_nothing(
    get_attributes: Callable[..., Iterator[Tuple[str, Any]]],
    *args: Any,
) -> List[Tuple[str, Any]]:
    """
    Collects all of the attributes, or none if getting them fails, so that one bad message or
    content block neither leaves partial attributes behind nor stops the rest from being recorded.
    """
    try:
        return list(get_attributes(*args))
    except Exception:
        logger.exception("Failed to get attributes.")
        return []


def _get_message_contents_attributes(
    message_prefix: str,
    blocks: Iterable[Any],
) -> Iterator[Tuple[str, Any]]:
    """
    Yields the attributes of each content block of the message at `message_prefix`, e.g.
    `llm.output_messages.0`. Content indices follow the block indices, and a block that cannot be
    mapped is recorded generically, so no index is skipped and no block is silently dropped.
    """
    tool_index = 0
    for j, block in enumerate(blocks):
        attributes = _attributes_or_nothing(
            _get_content_block_attributes, message_prefix, j, tool_index, block
        )
        if not attributes:
            attributes = _attributes_or_nothing(
                _get_generic_content_block_attributes,
                f"{message_prefix}.{MESSAGE_CONTENTS}.{j}",
                block,
            )
        elif _get(block, "type") in TOOL_CALL_BLOCK_TYPES:
            tool_index += 1
        yield from attributes


def _get_content_block_attributes(
    message_prefix: str,
    j: int,
    tool_index: int,
    block: Any,
) -> Iterator[Tuple[str, Any]]:
    block_type = _require(block, "type")
    prefix = f"{message_prefix}.{MESSAGE_CONTENTS}.{j}"
    if block_type == "text":
        yield f"{prefix}.{MESSAGE_CONTENT_TYPE}", "text"
        yield f"{prefix}.{MESSAGE_CONTENT_TEXT}", _require(block, "text")
    elif block_type == "thinking":
        yield f"{prefix}.{MESSAGE_CONTENT_TYPE}", "reasoning"
        yield f"{prefix}.{MESSAGE_CONTENT_TEXT}", _require(block, "thinking")
        if signature := _get(block, "signature"):
            yield f"{prefix}.{MESSAGE_CONTENT_SIGNATURE}", signature
    elif block_type == "redacted_thinking":
        yield f"{prefix}.{MESSAGE_CONTENT_TYPE}", "reasoning"
        yield f"{prefix}.{MESSAGE_CONTENT_DATA}", _require(block, "data")
    elif block_type in TOOL_CALL_BLOCK_TYPES:
        tool_call_id = _require(block, "id")
        name = _require(block, "name")
        arguments = safe_json_dumps(_require(block, "input"))
        tool_call_prefix = f"{message_prefix}.{MESSAGE_TOOL_CALLS}.{tool_index}"
        yield f"{tool_call_prefix}.{TOOL_CALL_ID}", tool_call_id
        yield f"{tool_call_prefix}.{TOOL_CALL_FUNCTION_NAME}", name
        yield f"{tool_call_prefix}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}", arguments
        yield f"{prefix}.{MESSAGE_CONTENT_TYPE}", "tool_use"
        yield f"{prefix}.{TOOL_CALL_ID}", tool_call_id
        yield f"{prefix}.{TOOL_CALL_FUNCTION_NAME}", name
        yield f"{prefix}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}", arguments
    elif block_type == "tool_result":
        yield (
            f"{message_prefix}.{MESSAGE_TOOL_CALL_ID}",
            _require(block, "tool_use_id"),
        )
        if (tool_result_content := _get(block, "content")) is not None:
            yield (
                f"{message_prefix}.{MESSAGE_CONTENT}",
                tool_result_content
                if isinstance(tool_result_content, str)
                else _to_json(tool_result_content),
            )
    elif block_type == "image":
        yield from _get_image_attributes(prefix, _require(block, "source"))
    elif block_type == "compaction":
        # The summary of the compacted context, which stands in for it from here on. Its content
        # is null if compaction failed.
        yield f"{prefix}.{MESSAGE_CONTENT_TYPE}", "text"
        if (content := _get(block, "content")) is not None:
            yield f"{prefix}.{MESSAGE_CONTENT_TEXT}", content
        if signature := _get(block, "signature"):
            yield f"{prefix}.{MESSAGE_CONTENT_SIGNATURE}", signature
    else:
        # e.g. server tool results, MCP tool results and listings, container uploads, fallback
        # boundaries, documents, search results, tool additions and removals, and types newer
        # than this instrumentation
        yield from _get_generic_content_block_attributes(prefix, block)


def _get_image_attributes(prefix: str, source: Any) -> Iterator[Tuple[str, Any]]:
    source_type = _get(source, "type")
    if source_type == "base64":
        url = f"data:{_get(source, 'media_type')};base64,{_require(source, 'data')}"
    elif source_type == "url":
        url = _require(source, "url")
    elif source_type == "file":
        # A file uploaded with the Files API has no URL, so it is referenced by its id.
        yield f"{prefix}.{MESSAGE_CONTENT_TYPE}", "image"
        yield f"{prefix}.{MESSAGE_CONTENT_ID}", _require(source, "file_id")
        return
    else:
        raise ValueError(f"Unknown image source type: {source_type}")
    yield f"{prefix}.{MESSAGE_CONTENT_TYPE}", "image"
    yield f"{prefix}.{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}", url


def _get_generic_content_block_attributes(prefix: str, block: Any) -> Iterator[Tuple[str, Any]]:
    """Records a block's type and, as its text, the block as JSON."""
    yield f"{prefix}.{MESSAGE_CONTENT_TYPE}", str(_get(block, "type") or "unknown")
    yield f"{prefix}.{MESSAGE_CONTENT_TEXT}", _to_json(block)


def _get_output_message_attributes(message: Any) -> Iterator[Tuple[str, Any]]:
    yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", message.role
    if stop_reason := message.stop_reason:
        yield LLM_FINISH_REASON, stop_reason
    yield from _get_message_contents_attributes(f"{LLM_OUTPUT_MESSAGES}.0", message.content)


def _get_inline_tool_definitions(messages: Iterable[Any]) -> Iterator[Any]:
    """
    Yields the tools defined inline in the messages by `tool_addition` blocks, which make them
    available to the model from that point in the conversation onward, in addition to the
    request's `tools`. A `tool_addition` that references a tool by name defines nothing new.
    """
    for message in messages:
        try:
            content = _get(message, "content")
            if isinstance(content, str) or not isinstance(content, Iterable):
                continue
            for block in content:
                if _get(block, "type") != "tool_addition":
                    continue
                tool = _get(block, "tool")
                if _get(tool, "type") == "tool_definition" and (
                    definition := _get(tool, "definition")
                ):
                    yield _to_jsonable(definition)
        except Exception:
            logger.exception("Failed to get the tools defined in a message.")


IMAGE_URL = ImageAttributes.IMAGE_URL
LLM_FINISH_REASON = SpanAttributes.LLM_FINISH_REASON
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID
MESSAGE_CONTENT_DATA = MessageContentAttributes.MESSAGE_CONTENT_DATA
MESSAGE_CONTENT_ID = MessageContentAttributes.MESSAGE_CONTENT_ID
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_SIGNATURE = MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
