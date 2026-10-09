"""Attribute extractors for embedding spans.

Embedding spans use EMBEDDING_* semantic conventions (EMBEDDING_TEXT,
EMBEDDING_VECTOR, EMBEDDING_MODEL_NAME) rather than LLM message attributes.

EMBEDDING_TEXT is extracted from the captured SDK request dict (not the raw
request parameters) because the SDK transforms contents differently depending
on the backend:
  - Gemini API: each Content with its parts is sent as a separate request entry
  - Vertex AI: text parts are extracted into flat strings, one per instance

The captured dict reflects these differences, so reading from it produces
correctly aligned EMBEDDING_TEXT entries that match the EMBEDDING_VECTOR
entries in the response.
"""

import logging
from typing import Any, Iterator, Mapping

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.google_genai._context import get_captured_request
from openinference.semconv.trace import (
    EmbeddingAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _EmbeddingRequestAttributesExtractor:
    def get_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[tuple[str, AttributeValue]]:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.EMBEDDING.value
        yield SpanAttributes.LLM_PROVIDER, OpenInferenceLLMProviderValues.GOOGLE.value
        if model := request_parameters.get("model"):
            if isinstance(model, str):
                yield SpanAttributes.EMBEDDING_MODEL_NAME, model
        # Embedding text is extracted from the captured request dict
        # (set by CapturedRequestScope after the SDK call) rather than
        # here, because the SDK's transform differs between Gemini API
        # and Vertex AI.

    @staticmethod
    def get_embedding_text_attributes() -> Iterator[tuple[str, AttributeValue]]:
        """Extract EMBEDDING_TEXT from the captured SDK request dict."""
        request = get_captured_request()
        if not isinstance(request, dict):
            return
        # Gemini API: {"requests": [{"content": {"parts": [{"text": ...}]}, ...}]}
        if requests := request.get("requests"):
            if isinstance(requests, list):
                for index, req in enumerate(requests):
                    if not isinstance(req, dict):
                        continue
                    content = req.get("content")
                    if isinstance(content, dict):
                        parts = content.get("parts")
                        if isinstance(parts, list):
                            texts = [
                                p["text"] for p in parts if isinstance(p, dict) and "text" in p
                            ]
                            if texts:
                                yield (
                                    f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{index}"
                                    f".{EmbeddingAttributes.EMBEDDING_TEXT}",
                                    "\n\n".join(texts),
                                )
                return
        # Vertex AI: {"instances": [{"content": "plain text", ...}]}
        if instances := request.get("instances"):
            if isinstance(instances, list):
                for index, inst in enumerate(instances):
                    if not isinstance(inst, dict):
                        continue
                    content = inst.get("content")
                    if isinstance(content, str):
                        yield (
                            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{index}"
                            f".{EmbeddingAttributes.EMBEDDING_TEXT}",
                            content,
                        )


class _EmbeddingResponseAttributesExtractor:
    def get_attributes(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[tuple[str, AttributeValue]]:
        if not response:
            return
        embeddings = getattr(response, "embeddings", None)
        if not embeddings:
            return
        for index, embedding in enumerate(embeddings):
            values = getattr(embedding, "values", None)
            if values is not None:
                yield (
                    f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.{index}"
                    f".{EmbeddingAttributes.EMBEDDING_VECTOR}",
                    values,
                )
