import logging
from typing import Any, Iterable, Iterator, Mapping, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.google_genai._utils import (
    _as_output_attributes,
    _io_value_and_type,
)
from openinference.semconv.trace import MessageAttributes, SpanAttributes

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
        for part in content_parts:
            if text := getattr(part, "text", None):
                yield MessageAttributes.MESSAGE_CONTENT, text

    def _get_attributes_from_generate_content_usage(
        self,
        usage: object,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (total_token_count := getattr(usage, "total_token_count", None)) is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_token_count
        if (prompt_token_count := getattr(usage, "prompt_token_count", None)) is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_token_count
        if (candidates_token_count := getattr(usage, "candidates_token_count", None)) is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, candidates_token_count
