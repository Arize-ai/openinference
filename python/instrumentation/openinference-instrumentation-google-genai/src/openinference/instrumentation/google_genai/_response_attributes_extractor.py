import logging
from typing import Any, Iterable, Iterator, Mapping, Tuple

from google.genai import types
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.google_genai._utils import (
    _as_output_attributes,
    _io_value_and_type,
)
from openinference.semconv.trace import MessageAttributes, SpanAttributes, ToolCallAttributes

__all__ = ("_ResponseAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def get_attributes(self, response: Any) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _as_output_attributes(
            _io_value_and_type(response),
        )

    def get_extra_attributes(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from self._get_attributes_from_generate_content(
            response=response,
            request_parameters=request_parameters,
        )

    def _get_attributes_from_generate_content(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # https://github.com/googleapis/python-genai/blob/e9e84aa38726e7b65796812684d9609461416b11/google/genai/types.py#L2981  # noqa: E501
        if model_version := getattr(response, "model_version", None):
            yield SpanAttributes.LLM_MODEL_NAME, model_version
        if usage_metadata := getattr(response, "usage_metadata", None):
            yield from self._get_attributes_from_generate_content_usage(usage_metadata)
        if (candidates := getattr(response, "candidates", None)) and isinstance(
            candidates, Iterable
        ):
            index = -1
            for candidate in candidates:
                # TODO: This is a hack to get the index of the candidate.
                #       Might be a better way to do this.
                # Keep track of previous index to increment if index not found
                index = (
                    index + 1
                    if getattr(candidate, "index") is None
                    else getattr(candidate, "index")
                )
                if content := getattr(candidate, "content", None):
                    for key, value in self._get_attributes_from_generate_content_content(content):
                        yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value

        # Handle automatic function calling history
        # For automatic function calling, the function call details are stored separately
        if automatic_history := getattr(response, "automatic_function_calling_history", None):
            yield from self._get_attributes_from_automatic_function_calling_history(
                automatic_history
            )

    def _get_attributes_from_generate_content_content(
        self,
        content: object,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if content_parts := getattr(content, "parts", None):
            yield from self._get_attributes_from_content_parts(content_parts)
        if role := getattr(content, "role", None):
            yield MessageAttributes.MESSAGE_ROLE, role

    def _get_attributes_from_content_parts(
        self,
        content_parts: Iterable[object],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # https://github.com/googleapis/python-genai/blob/e9e84aa38726e7b65796812684d9609461416b11/google/genai/types.py#L565  # noqa: E501
        text_content = []
        tool_call_index = 0

        for part in content_parts:
            if text := getattr(part, "text", None):
                text_content.append(text)
            elif function_call := getattr(part, "function_call", None):
                # Handle tool/function calls
                yield from self._get_attributes_from_function_call(function_call, tool_call_index)
                tool_call_index += 1

        # Always yield message content for consistency, even if empty
        # This ensures Phoenix can properly display the message structure
        content = "\n".join(text_content) if text_content else ""
        yield MessageAttributes.MESSAGE_CONTENT, content

    def _get_attributes_from_function_call(
        self,
        function_call: object,
        tool_call_index: int,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract attributes from a function call in the response."""
        try:
            if function_name := getattr(function_call, "name", None):
                yield (
                    f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_call_index}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                    function_name,
                )

            if function_args := getattr(function_call, "args", None):
                # Serialize the function arguments
                try:
                    args_json = safe_json_dumps(function_args)
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_call_index}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        args_json,
                    )
                except Exception:
                    logger.exception(
                        f"Failed to serialize function call args for tool call {tool_call_index}"
                    )
        except Exception:
            logger.exception(
                f"Failed to extract function call attributes for tool call {tool_call_index}"
            )

    def _get_attributes_from_generate_content_usage(
        self,
        obj: types.GenerateContentResponseUsageMetadata,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if total := obj.total_token_count:
            yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total
        if obj.prompt_tokens_details:
            prompt_details_audio = 0
            for modality_token_count in obj.prompt_tokens_details:
                if (
                    modality_token_count.modality is types.MediaModality.AUDIO
                    and modality_token_count.token_count
                ):
                    prompt_details_audio += modality_token_count.token_count
            if prompt_details_audio:
                yield (
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO,
                    prompt_details_audio,
                )
        if prompt := obj.prompt_token_count:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt
        if obj.candidates_tokens_details:
            completion_details_audio = 0
            for modality_token_count in obj.candidates_tokens_details:
                if (
                    modality_token_count.modality is types.MediaModality.AUDIO
                    and modality_token_count.token_count
                ):
                    completion_details_audio += modality_token_count.token_count
            if completion_details_audio:
                yield (
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO,
                    completion_details_audio,
                )
        completion = 0
        if candidates := obj.candidates_token_count:
            completion += candidates
        if thoughts := obj.thoughts_token_count:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, thoughts
            completion += thoughts
        if completion:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion

    def _get_attributes_from_automatic_function_calling_history(
        self,
        history: Iterable[object],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract function call information from automatic_function_calling_history.

        This history contains the sequence of model->function call->function response
        that happened during automatic function calling.
        """
        tool_call_index = 0

        for content_entry in history:
            # Each entry is a Content object with parts
            if not hasattr(content_entry, "parts") or not hasattr(content_entry, "role"):
                continue

            # Look for model responses that contain function calls
            if getattr(content_entry, "role") == "model":
                parts = getattr(content_entry, "parts", [])
                for part in parts:
                    if function_call := getattr(part, "function_call", None):
                        # Extract function call details for the span
                        yield from self._get_attributes_from_function_call(
                            function_call, tool_call_index
                        )
                        tool_call_index += 1
