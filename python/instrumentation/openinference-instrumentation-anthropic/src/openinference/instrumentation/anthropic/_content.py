"""
Span attributes for the content blocks of Anthropic messages.

Shared by the input messages and by the output message of the streaming and non-streaming paths,
so that every path maps every block type the same way. Blocks can be request params (dicts) or
response models (e.g. a response passed back as input), so both are read with `_get`.
"""

import json
import logging
import warnings
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import TraceConfig, safe_json_dumps
from openinference.instrumentation.config import REDACTED_VALUE
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
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, Mapping):
        return {key: _to_jsonable(value) for key, value in obj.items()}
    # not a model class, e.g. the output_format of messages.parse(), which has no instance to dump
    if hasattr(obj, "model_dump") and not isinstance(obj, type):
        # by_alias, so e.g. a fallback block's `from_` is recorded as the API's `from`, and
        # without warnings, which the SDK's stand-in model for an unknown block type would raise
        try:
            return obj.model_dump(mode="json", by_alias=True, exclude_none=True, warnings=False)
        except (TypeError, ValueError):
            # Pydantic v1, whose model_dump (the SDK's shim) does not take `warnings`
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return obj.model_dump(mode="json", by_alias=True, exclude_none=True)
    return obj


def _redact_sources(obj: Any, hide_images: bool) -> Any:
    """
    Redacts the nested sources that hold data, so that recording a block as JSON neither copies
    base64 images and documents, e.g. whole PDFs, nor bypasses the trace config's image settings:
    every base64 source, and every image source if images are hidden. A source keeps its type and
    media type.
    """
    if isinstance(obj, list):
        return [_redact_sources(item, hide_images) for item in obj]
    if not isinstance(obj, dict):
        return obj
    return {
        key: _redacted_source(value)
        if key == "source"
        and isinstance(value, dict)
        and (value.get("type") == "base64" or (hide_images and obj.get("type") == "image"))
        else _redact_sources(value, hide_images)
        for key, value in obj.items()
    }


def _redacted_source(source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value if key in ("type", "media_type") else REDACTED_VALUE
        for key, value in source.items()
    }


def _to_json(obj: Any, hide_images: bool = False) -> str:
    return safe_json_dumps(_redact_sources(_to_jsonable(obj), hide_images))


def _hides_input_images(config: Optional[TraceConfig]) -> bool:
    return bool(config and (config.hide_inputs or config.hide_input_images))


def _get_request_params(kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """
    The request's parameters, to be recorded, without the ones that were not given, e.g. the
    `omit` defaults that the tool runner passes explicitly.
    """
    from anthropic import NotGiven, Omit

    return {
        key: value
        for key, value in kwargs.items()
        if not isinstance(value, (NotGiven, Omit))
        and not (key in ("extra_query", "extra_body") and value is None)
    }


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
        logger.debug("Failed to get attributes.", exc_info=True)
        return []


def _get_message_contents_attributes(
    message_prefix: str,
    blocks: Iterable[Any],
    hide_images: bool = False,
) -> Iterator[Tuple[str, Any]]:
    """
    Yields the attributes of each content block of the message at `message_prefix`, e.g.
    `llm.output_messages.0`. Content indices follow the block indices, and a block that cannot be
    mapped is recorded generically, so no index is skipped and no block is silently dropped.
    `hide_images` redacts the images nested in blocks that are recorded as JSON.
    """
    tool_index = 0
    for j, block in enumerate(blocks):
        attributes = _attributes_or_nothing(
            _get_content_block_attributes, message_prefix, j, tool_index, block, hide_images
        )
        if not attributes:
            attributes = _attributes_or_nothing(
                _get_generic_content_block_attributes,
                f"{message_prefix}.{MESSAGE_CONTENTS}.{j}",
                block,
                hide_images,
            )
        elif _get(block, "type") in TOOL_CALL_BLOCK_TYPES:
            tool_index += 1
        yield from attributes


def _get_content_block_attributes(
    message_prefix: str,
    j: int,
    tool_index: int,
    block: Any,
    hide_images: bool = False,
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
                else _to_json(tool_result_content, hide_images),
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
        yield from _get_generic_content_block_attributes(prefix, block, hide_images)


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


def _get_generic_content_block_attributes(
    prefix: str,
    block: Any,
    hide_images: bool = False,
) -> Iterator[Tuple[str, Any]]:
    """Records a block's type and, as its text, the block as JSON, its data sources redacted."""
    yield f"{prefix}.{MESSAGE_CONTENT_TYPE}", str(_get(block, "type") or "unknown")
    yield f"{prefix}.{MESSAGE_CONTENT_TEXT}", _to_json(block, hide_images)


def _get_output_message_attributes(message: Any) -> Iterator[Tuple[str, Any]]:
    yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", message.role
    if stop_reason := message.stop_reason:
        yield LLM_FINISH_REASON, stop_reason
    yield from _get_message_contents_attributes(f"{LLM_OUTPUT_MESSAGES}.0", message.content)


def _get_response_metadata(
    message: Any,
    config: Optional[TraceConfig] = None,
    request_messages: Any = None,
) -> Dict[str, Any]:
    """
    Collects the provider-specific data of a response that has no semantic convention of its own,
    to be recorded in the span metadata. Each item is guarded, so one cannot lose the others.

    - stop_details: why the output stopped, e.g. the category of a refusal
    - cache_miss_reason: why the prompt cache could not reuse a previous request's prefix
    - container_id: the container that a container tool, e.g. code execution, ran in
    - context_management_applied_edits: the context management edits that were applied
    - input_transformations: the changes the API made to the request's input
    - usage_iterations: the token usage of each sampling and compaction iteration. A compaction
      iteration's tokens are not included in the top-level usage, so they are not included in
      the token counts either.
    - server_tool_use: the number of server tool requests, e.g. web searches, by tool, when any
    - citations: the citations of each text block, keyed by its index in the output contents
    - mcp_server_names: the MCP server of each MCP tool call, keyed by its tool call id
    - tool_callers: the caller of each tool call that was not called directly by the model,
      e.g. by code execution, keyed by its tool call id, from the request's messages and the
      response, which wins
    - toolset_names: the toolset of each tool call and tool result, keyed by its tool call id,
      likewise

    The metadata is not masked by the trace config, so the items that quote the input or the
    output are left out here when the config hides it: the citations, which quote the cited
    input, and the explanation in the stop details, which describes the output.
    """
    metadata: Dict[str, Any] = {}
    hide_output_text = bool(config and (config.hide_outputs or config.hide_output_text))
    hide_citations = hide_output_text or bool(
        config and (config.hide_inputs or config.hide_input_messages or config.hide_input_text)
    )

    def add(key: str, get_value: Callable[[], Any]) -> None:
        try:
            if value := get_value():
                metadata[key] = _to_jsonable(value)
        except Exception:
            logger.exception("Failed to get the %s of the response.", key)

    add("stop_details", lambda: _get_stop_details(message, hide_output_text))
    add("cache_miss_reason", lambda: _get(_get(message, "diagnostics"), "cache_miss_reason"))
    add("container_id", lambda: _get(_get(message, "container"), "id"))
    add(
        "context_management_applied_edits",
        lambda: _get(_get(message, "context_management"), "applied_edits"),
    )
    add("input_transformations", lambda: _get(message, "input_transformations"))
    add("usage_iterations", lambda: _get(_get(message, "usage"), "iterations"))
    add("server_tool_use", lambda: _get_server_tool_use(message))
    if not hide_citations:
        add("citations", lambda: _get_citations(message))
    add("mcp_server_names", lambda: _get_mcp_server_names(message))
    tool_calls = _ToolCallMetadata()
    tool_calls.add_messages(request_messages)
    tool_calls.add_blocks(_get(message, "content"))
    add("tool_callers", lambda: tool_calls.callers)
    add("toolset_names", lambda: tool_calls.toolset_names)
    return metadata


class _ToolCallMetadata:
    """
    Collects the callers and toolsets of tool calls, keyed by tool call id. A later block's entry
    replaces an earlier one's, so the response's win over the request's.
    """

    def __init__(self) -> None:
        self.callers: Dict[str, Any] = {}
        self.toolset_names: Dict[str, Any] = {}

    def add_messages(self, messages: Any) -> None:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Iterable):
            return
        for message in messages:
            content = _get(message, "content")
            if not isinstance(content, str):
                self.add_blocks(content)

    def add_blocks(self, blocks: Any) -> None:
        if isinstance(blocks, (str, bytes)) or not isinstance(blocks, Iterable):
            return
        for block in blocks:
            try:
                self._add_block(block)
            except Exception:
                logger.debug("Failed to get the caller or toolset of a block.", exc_info=True)

    def _add_block(self, block: Any) -> None:
        block_type = _get(block, "type")
        if block_type in TOOL_CALL_BLOCK_TYPES:
            tool_call_id = _get(block, "id")
            caller = _to_jsonable(_get(block, "caller"))
            # a call made directly by the model, the default, is not recorded
            if caller and caller != {"type": "direct"}:
                self.callers[tool_call_id] = caller
        elif block_type == "tool_result":
            tool_call_id = _get(block, "tool_use_id")
        else:
            return
        if toolset_name := _get(block, "toolset_name"):
            self.toolset_names[tool_call_id] = toolset_name


def _get_stop_details(message: Any, hide_explanation: bool) -> Any:
    if not (stop_details := _to_jsonable(_get(message, "stop_details"))):
        return None
    if hide_explanation and isinstance(stop_details, dict):
        stop_details.pop("explanation", None)
    return stop_details


def _get_server_tool_use(message: Any) -> Any:
    # only when a server tool was used, since the API returns zero counts on most responses
    counts = _to_jsonable(_get(_get(message, "usage"), "server_tool_use"))
    return counts if isinstance(counts, dict) and any(counts.values()) else None


def _get_citations(message: Any) -> Dict[str, Any]:
    return {
        str(j): citations
        for j, block in enumerate(_get(message, "content") or ())
        if _get(block, "type") == "text" and (citations := _get(block, "citations"))
    }


def _get_mcp_server_names(message: Any) -> Dict[str, Any]:
    return {
        _get(block, "id"): _get(block, "server_name")
        for block in _get(message, "content") or ()
        if _get(block, "type") == "mcp_tool_use"
    }


def _get_metadata_attributes(
    message: Any,
    context_metadata: Optional[AttributeValue] = None,
    config: Optional[TraceConfig] = None,
    request_messages: Any = None,
) -> Iterator[Tuple[str, Any]]:
    """
    Yields the span metadata: the response's metadata merged with the metadata from the context
    that the span was started in, which takes precedence, since this replaces it on the span.
    """
    if not (metadata := _get_response_metadata(message, config, request_messages)):
        return
    if isinstance(context_metadata, str):
        try:
            parsed = json.loads(context_metadata)
        except Exception:
            parsed = None
        if not isinstance(parsed, dict):
            # the context's metadata cannot be merged, so it is kept as it is
            return
        metadata.update(parsed)
    yield METADATA, safe_json_dumps(metadata)


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
METADATA = SpanAttributes.METADATA
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
