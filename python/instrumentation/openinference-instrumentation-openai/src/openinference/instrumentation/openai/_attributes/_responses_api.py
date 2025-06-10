from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Tuple, Union

from opentelemetry.util.types import AttributeValue
from typing_extensions import assert_never

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from openai.types import responses

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def stop_on_exception(
    wrapped: Callable[..., Iterator[Tuple[str, Any]]],
) -> Callable[..., Iterator[Tuple[str, Any]]]:
    def wrapper(*args: Any, **kwargs: Any) -> Iterator[Tuple[str, Any]]:
        try:
            yield from wrapped(*args, **kwargs)
        except Exception:
            logger.exception(f"Failed to get attribute in {wrapped.__name__}.")

    return wrapper


class _ResponsesApiAttributes:
    @classmethod
    @stop_on_exception
    def _get_attributes_from_message_param(
        cls,
        obj: Union[
            responses.easy_input_message_param.EasyInputMessageParam,
            responses.response_input_param.Message,
            responses.response_output_message_param.ResponseOutputMessageParam,
        ],
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (role := obj.get("role")) is not None:
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", role
        if (content := obj.get("content")) is not None:
            if isinstance(content, str):
                yield f"{prefix}{MessageAttributes.MESSAGE_CONTENT}", content
            elif isinstance(content, Iterable):
                if TYPE_CHECKING:
                    assert not isinstance(content, str)
                yield from cls._get_attributes_from_message_param_content_list(content, prefix)

    @classmethod
    @stop_on_exception
    def _get_attributes_from_message_param_content_list(
        cls,
        obj: Union[
            Iterable[responses.response_input_message_content_list_param.ResponseInputContentParam],
            Iterable[responses.response_output_message_param.Content],
        ],
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        for i, item in enumerate(obj):
            if "type" not in item:
                continue
            inner_prefix = f"{prefix}{MessageAttributes.MESSAGE_CONTENTS}.{i}."
            if item["type"] == "input_text":
                yield from cls._get_attributes_from_response_input_text_param(item, inner_prefix)
            elif item["type"] == "output_text":
                yield from cls._get_attributes_from_response_output_text_param(item, inner_prefix)
            elif item["type"] == "input_image":
                yield from cls._get_attributes_from_response_input_image_param(item, inner_prefix)
            elif item["type"] == "input_file":
                # TODO: Handle input file
                pass
            elif item["type"] == "refusal":
                yield from cls._get_attributes_from_response_output_refusal_param(
                    item, inner_prefix
                )
            elif TYPE_CHECKING:
                assert_never(item["type"])

    @classmethod
    @stop_on_exception
    def _get_attributes_from_output_message_content(
        cls,
        obj: Union[
            responses.response_output_text.ResponseOutputText,
            responses.response_output_refusal.ResponseOutputRefusal,
        ],
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        from openai.types import responses

        if isinstance(obj, responses.ResponseOutputText):
            yield from cls._get_attributes_from_response_output_text(obj, prefix)
        elif isinstance(obj, responses.ResponseOutputRefusal):
            yield from cls._get_attributes_from_response_output_refusal(obj, prefix)
        elif TYPE_CHECKING:
            assert_never(obj)

    @classmethod
    @stop_on_exception
    def _get_attributes_from_output_message_content_list(
        cls,
        obj: Iterable[
            Union[
                responses.response_output_text.ResponseOutputText,
                responses.response_output_refusal.ResponseOutputRefusal,
            ]
        ],
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        for i, item in enumerate(obj):
            yield from cls._get_attributes_from_output_message_content(
                item, f"{prefix}{MessageAttributes.MESSAGE_CONTENTS}.{i}."
            )

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response(
        cls,
        obj: responses.response.Response,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.LLM_MODEL_NAME, obj.model
        if obj.usage:
            yield from cls._get_attributes_from_response_usage(obj.usage)
        if isinstance(obj.output, Iterable):
            for i, item in enumerate(obj.output):
                yield from cls._get_attributes_from_response_output_item(
                    item,
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{i}.",
                )
        if isinstance(obj.tools, Iterable):
            for i, tool in enumerate(obj.tools):
                yield from cls._get_attributes_from_response_tool(
                    tool,
                    f"{SpanAttributes.LLM_TOOLS}.{i}.",
                )

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_computer_tool_call(
        cls,
        obj: responses.response_computer_tool_call.ResponseComputerToolCall,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "tool"
        if (call_id := obj.call_id) is not None:
            yield f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALL_ID}", call_id

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_computer_tool_call_param(
        cls,
        obj: responses.response_computer_tool_call_param.ResponseComputerToolCallParam,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "tool"
        if (call_id := obj.get("call_id")) is not None:
            yield f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALL_ID}", call_id

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_create_param_base(
        cls,
        obj: responses.response_create_params.ResponseCreateParamsBase,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        invocation_params = dict(obj)
        invocation_params.pop("input", None)
        invocation_params.pop("instructions", None)
        if isinstance((tools := invocation_params.pop("tools", None)), Iterable):
            for i, tool in enumerate(tools):
                yield from cls._get_attributes_from_response_tool_param(
                    tool, f"{SpanAttributes.LLM_TOOLS}.{i}."
                )
        yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)
        if (model := obj.get("model")) is not None:
            yield SpanAttributes.LLM_MODEL_NAME, model
        if (instructions := obj.get("instructions")) is not None:
            yield (
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
                "system",
            )
            yield (
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}",
                instructions,
            )
        if (input := obj.get("input")) is not None:
            if isinstance(input, str):
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}",
                    "user",
                )
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}",
                    input,
                )
            elif isinstance(input, list):
                yield from cls._get_attributes_from_response_input_item_params(input)

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_file_search_tool_call(
        cls,
        obj: responses.response_file_search_tool_call.ResponseFileSearchToolCall,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (id := obj.id) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_ID}", id
        if (type := obj.type) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", type

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_file_search_tool_call_param(
        cls,
        obj: responses.response_file_search_tool_call_param.ResponseFileSearchToolCallParam,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (id := obj.get("id")) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_ID}", id
        if (type := obj.get("type")) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", type

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_tool(
        cls,
        obj: responses.tool.Tool,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield (f"{prefix}{ToolAttributes.TOOL_JSON_SCHEMA}", obj.model_dump_json())

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_tool_param(
        cls,
        obj: responses.tool_param.ToolParam,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield (f"{prefix}{ToolAttributes.TOOL_JSON_SCHEMA}", safe_json_dumps(obj))

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_function_tool_call(
        cls,
        obj: responses.ResponseFunctionToolCall,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (call_id := obj.call_id) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_ID}", call_id
        if (name := obj.name) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", name
        if obj.arguments != "{}":
            yield (
                f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                obj.arguments,
            )

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_function_tool_call_param(
        cls,
        obj: responses.response_function_tool_call_param.ResponseFunctionToolCallParam,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (call_id := obj.get("call_id")) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_ID}", call_id
        if (name := obj.get("name")) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", name
        if (arguments := obj.get("arguments")) is not None:
            if arguments != "{}":
                yield (
                    f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                    arguments,
                )

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_function_web_search(
        cls,
        obj: responses.response_function_web_search.ResponseFunctionWebSearch,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (id := obj.id) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_ID}", id
        if (type := obj.type) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", type

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_function_web_search_param(
        cls,
        obj: responses.response_function_web_search_param.ResponseFunctionWebSearchParam,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (id := obj.get("id")) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_ID}", id
        if (type := obj.get("type")) is not None:
            yield f"{prefix}{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", type

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_input_image_param(
        cls,
        obj: responses.response_input_image_param.ResponseInputImageParam,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if "image_url" in obj and (image_url := obj["image_url"]):
            yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "image"
            yield (
                f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}",
                image_url,
            )

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_input_item_param(
        cls,
        obj: responses.response_input_param.ResponseInputItemParam,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if "type" not in obj:
            if "role" in obj and "content" in obj:
                yield from cls._get_attributes_from_message_param(
                    {
                        "type": "message",
                        "role": obj["role"],  # type: ignore[typeddict-item]
                        "content": obj["content"],  # type: ignore[typeddict-item]
                    },
                    prefix,
                )
        elif obj["type"] == "message":
            yield from cls._get_attributes_from_message_param(obj, prefix)
        elif obj["type"] == "function_call":
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "assistant"
            yield from cls._get_attributes_from_response_function_tool_call_param(
                obj,
                f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALLS}.0.",
            )
        elif obj["type"] == "function_call_output":
            yield from cls._get_attributes_from_response_input_param_function_call_output(
                obj, prefix
            )
        elif obj["type"] == "reasoning":
            yield from cls._get_attributes_from_response_reasoning_item_param(obj, prefix)
        elif obj["type"] == "item_reference":
            # TODO: Handle item reference
            pass
        elif obj["type"] == "file_search_call":
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "assistant"
            yield from cls._get_attributes_from_response_file_search_tool_call_param(
                obj,
                f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALLS}.0.",
            )
        elif obj["type"] == "computer_call":
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "assistant"
            yield from cls._get_attributes_from_response_computer_tool_call_param(
                obj,
                f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALLS}.0.",
            )
        elif obj["type"] == "computer_call_output":
            yield from cls._get_attributes_from_response_input_param_computer_call_output(
                obj, prefix
            )
        elif obj["type"] == "web_search_call":
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "assistant"
            yield from cls._get_attributes_from_response_function_web_search_param(
                obj,
                f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALLS}.0.",
            )
        elif obj["type"] == "image_generation_call":
            # TODO: Handle image generation call
            pass
        elif obj["type"] == "code_interpreter_call":
            # TODO: Handle code interpreter call
            pass
        elif obj["type"] == "local_shell_call":
            # TODO: Handle local shell call
            pass
        elif obj["type"] == "local_shell_call_output":
            # TODO: Handle local shell call output
            pass
        elif obj["type"] == "mcp_list_tools":
            # TODO: Handle mcp list tools
            pass
        elif obj["type"] == "mcp_approval_request":
            # TODO: Handle mcp approval request
            pass
        elif obj["type"] == "mcp_approval_response":
            # TODO: Handle mcp approval response
            pass
        elif obj["type"] == "mcp_call":
            # TODO: Handle mcp call
            pass
        elif TYPE_CHECKING and obj["type"] is not None:
            assert_never(obj["type"])

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_input_item_params(
        cls,
        obj: Iterable[responses.response_input_param.ResponseInputItemParam],
        msg_idx: int = 1,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        for i, item in enumerate(obj, msg_idx):
            prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}."
            yield from cls._get_attributes_from_response_input_item_param(item, prefix)

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_input_param_computer_call_output(
        cls,
        obj: responses.response_input_param.ComputerCallOutput,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "tool"
        if (call_id := obj.get("call_id")) is not None:
            yield f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALL_ID}", call_id

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_input_param_function_call_output(
        cls,
        obj: responses.response_input_param.FunctionCallOutput,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "tool"
        if (call_id := obj.get("call_id")) is not None:
            yield f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALL_ID}", call_id
        if (output := obj.get("output")) is not None:
            yield f"{prefix}{MessageAttributes.MESSAGE_CONTENT}", output

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_input_text_param(
        cls,
        obj: responses.response_input_text_param.ResponseInputTextParam,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (text := obj.get("text")) is not None:
            yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", text
            yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_output_item(
        cls,
        obj: responses.response_output_item.ResponseOutputItem,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if obj.type == "message":
            yield from cls._get_attributes_from_response_output_message(obj, prefix)
        elif obj.type == "function_call":
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "assistant"
            yield from cls._get_attributes_from_response_function_tool_call(
                obj,
                f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALLS}.0.",
            )
        elif obj.type == "file_search_call":
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "assistant"
            yield from cls._get_attributes_from_response_file_search_tool_call(
                obj,
                f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALLS}.0.",
            )
        elif obj.type == "computer_call":
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "assistant"
            yield from cls._get_attributes_from_response_computer_tool_call(
                obj,
                f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALLS}.0.",
            )
        elif obj.type == "reasoning":
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "assistant"
            yield from cls._get_attributes_from_response_reasoning_item(obj, prefix)
        elif obj.type == "web_search_call":
            yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "assistant"
            yield from cls._get_attributes_from_response_function_web_search(
                obj,
                f"{prefix}{MessageAttributes.MESSAGE_TOOL_CALLS}.0.",
            )
        elif obj.type == "image_generation_call":
            # TODO: Handle image generation call
            pass
        elif obj.type == "code_interpreter_call":
            # TODO: Handle code interpreter call
            pass
        elif obj.type == "local_shell_call":
            # TODO: Handle local shell call
            pass
        elif obj.type == "mcp_call":
            # TODO: Handle mcp call
            pass
        elif obj.type == "mcp_list_tools":
            # TODO: Handle mcp list tools
            pass
        elif obj.type == "mcp_approval_request":
            # TODO: Handle mcp approval request
            pass
        elif TYPE_CHECKING:
            assert_never(obj.type)

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_output_message(
        cls,
        obj: responses.response_output_message.ResponseOutputMessage,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", obj.role
        yield from cls._get_attributes_from_output_message_content_list(obj.content, prefix)

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_output_refusal(
        cls,
        obj: responses.response_output_refusal.ResponseOutputRefusal,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"
        yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", obj.refusal

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_output_refusal_param(
        cls,
        obj: responses.response_output_refusal_param.ResponseOutputRefusalParam,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (refusal := obj.get("refusal")) is not None:
            yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", refusal
            yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_output_text(
        cls,
        obj: responses.response_output_text.ResponseOutputText,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"
        yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", obj.text

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_output_text_param(
        cls,
        obj: responses.response_output_text_param.ResponseOutputTextParam,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (text := obj.get("text")) is not None:
            yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", text
            yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_reasoning_item(
        cls,
        obj: responses.response_reasoning_item.ResponseReasoningItem,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if isinstance(obj.summary, Iterable):
            for i, item in enumerate(obj.summary):
                yield from cls._get_attributes_from_response_reasoning_item_summary(
                    item,
                    f"{prefix}{MessageAttributes.MESSAGE_CONTENTS}.{i}.",
                )

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_reasoning_item_param(
        cls,
        obj: responses.response_reasoning_item_param.ResponseReasoningItemParam,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield f"{prefix}{MessageAttributes.MESSAGE_ROLE}", "assistant"
        if isinstance((summary := obj.get("summary")), Iterable):
            for i, item in enumerate(summary):
                if "type" not in item:
                    continue
                if item["type"] == "summary_text":
                    yield from cls._get_attributes_from_response_reasoning_item_param_summary(
                        item,
                        f"{prefix}{MessageAttributes.MESSAGE_CONTENTS}.{i}.",
                    )
                elif TYPE_CHECKING:
                    assert_never(item["type"])

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_reasoning_item_param_summary(
        cls,
        obj: responses.response_reasoning_item_param.Summary,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (text := obj.get("text")) is not None:
            yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", text
            yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_reasoning_item_summary(
        cls,
        obj: responses.response_reasoning_item.Summary,
        prefix: str = "",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"
        yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", obj.text

    @classmethod
    @stop_on_exception
    def _get_attributes_from_response_usage(
        cls,
        obj: responses.response_usage.ResponseUsage,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, obj.total_tokens
        yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, obj.input_tokens
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, obj.output_tokens
        yield (
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
            obj.output_tokens_details.reasoning_tokens,
        )
        yield (
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
            obj.input_tokens_details.cached_tokens,
        )
