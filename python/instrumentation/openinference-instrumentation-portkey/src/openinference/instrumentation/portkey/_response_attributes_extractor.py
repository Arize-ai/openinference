import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    """Extracts attributes from response objects."""

    def __init__(self, response: Any) -> None:
        self.response = response

    def extract(self) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract attributes from response objects."""
        yield from self._extract_output_attributes()
        yield from self._extract_usage_attributes()

    def _extract_output_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract output-related attributes."""
        # TODO: Implement output attribute extraction for Portkey AI
        # This will depend on how Portkey AI structures its response objects
        # For example:
        # if hasattr(self.response, "choices"):
        #     for i, choice in enumerate(self.response.choices):
        #         if hasattr(choice, "message"):
        #             message = choice.message
        #             if hasattr(message, "role"):
        #                 yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{i}.{MessageAttributes.MESSAGE_ROLE}", message.role
        #             if hasattr(message, "content"):
        #                 yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{i}.{MessageAttributes.MESSAGE_CONTENT}", message.content
        pass

    def _extract_usage_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        """Extract usage-related attributes."""
        # TODO: Implement usage attribute extraction for Portkey AI
        # This will depend on how Portkey AI structures its usage information
        # For example:
        # if hasattr(self.response, "usage"):
        #     usage = self.response.usage
        #     if hasattr(usage, "prompt_tokens"):
        #         yield SpanAttributes.LLM_PROMPT_TOKENS, usage.prompt_tokens
        #     if hasattr(usage, "completion_tokens"):
        #         yield SpanAttributes.LLM_COMPLETION_TOKENS, usage.completion_tokens
        #     if hasattr(usage, "total_tokens"):
        #         yield SpanAttributes.LLM_TOTAL_TOKENS, usage.total_tokens
        pass 