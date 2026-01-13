import logging
from typing import Any, Iterable, Iterator, Mapping, Optional, Tuple

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.portkey._utils import _as_output_attributes, _io_value_and_type
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    SpanAttributes,
)

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
        yield from self._get_attributes_from_chat_completion(
            completion=response,
            request_parameters=request_parameters,
        )

    def _get_attributes_from_chat_completion(
        self,
        completion: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if model := getattr(completion, "model", None):
            yield SpanAttributes.LLM_MODEL_NAME, model

            if provider := infer_llm_provider_from_model(model):
                yield SpanAttributes.LLM_PROVIDER, provider.value

                if system := _PROVIDER_TO_SYSTEM.get(provider.value):
                    yield SpanAttributes.LLM_SYSTEM, system

        if usage := getattr(completion, "usage", None):
            yield from self._get_attributes_from_completion_usage(usage)
        if (choices := getattr(completion, "choices", None)) and isinstance(choices, Iterable):
            for choice in choices:
                if (index := getattr(choice, "index", None)) is None:
                    continue
                if message := getattr(choice, "message", None):
                    for key, value in self._get_attributes_from_chat_completion_message(message):
                        yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value

    def _get_attributes_from_chat_completion_message(
        self,
        message: object,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if role := getattr(message, "role", None):
            yield MessageAttributes.MESSAGE_ROLE, role
        if content := getattr(message, "content", None):
            yield MessageAttributes.MESSAGE_CONTENT, content
        if function_call := getattr(message, "function_call", None):
            if name := getattr(function_call, "name", None):
                yield MessageAttributes.MESSAGE_FUNCTION_CALL_NAME, name
            if arguments := getattr(function_call, "arguments", None):
                yield MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON, arguments

    def _get_attributes_from_completion_usage(
        self,
        usage: object,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if (total_tokens := getattr(usage, "total_tokens", None)) is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens
        if (prompt_tokens := getattr(usage, "prompt_tokens", None)) is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens
        if (completion_tokens := getattr(usage, "completion_tokens", None)) is not None:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens


def infer_llm_provider_from_model(
    model_name: Optional[str] = None,
) -> Optional[OpenInferenceLLMProviderValues]:
    """Infer the LLM provider from a model identifier when possible."""
    if not model_name:
        return None

    model = model_name.lower()

    # OpenAI
    if model.startswith(("gpt-", "gpt.", "o1", "o3", "o4")):
        return OpenInferenceLLMProviderValues.OPENAI

    # Anthropic
    if model.startswith(("claude-", "anthropic.claude")):
        return OpenInferenceLLMProviderValues.ANTHROPIC

    # Google / Vertex / Gemini
    if model.startswith(
        (
            "gemini",
            "google",
            "vertex",
            "vertexai",
            "google_genai",
            "google_vertexai",
            "google_anthropic_vertex",
        )
    ):
        return OpenInferenceLLMProviderValues.GOOGLE

    # AWS Bedrock
    if model.startswith(("bedrock", "bedrock_converse")):
        return OpenInferenceLLMProviderValues.AWS

    # Mistral
    if model.startswith(("mistral", "mixtral", "mistralai")):
        return OpenInferenceLLMProviderValues.MISTRALAI

    # Cohere
    if model.startswith(("command", "cohere", "cohere.command")):
        return OpenInferenceLLMProviderValues.COHERE

    # xAI
    if model.startswith(("grok", "xai")):
        return OpenInferenceLLMProviderValues.XAI

    # DeepSeek
    if model.startswith("deepseek"):
        return OpenInferenceLLMProviderValues.DEEPSEEK

    return None


_NA = None
_PROVIDER_TO_SYSTEM = {
    "anthropic": OpenInferenceLLMSystemValues.ANTHROPIC.value,
    "azure": OpenInferenceLLMSystemValues.OPENAI.value,
    "azure_ai": OpenInferenceLLMSystemValues.OPENAI.value,
    "azure_openai": OpenInferenceLLMSystemValues.OPENAI.value,
    "bedrock": _NA,
    "bedrock_converse": _NA,
    "cohere": OpenInferenceLLMSystemValues.COHERE.value,
    "deepseek": _NA,
    "fireworks": _NA,
    "google": OpenInferenceLLMSystemValues.VERTEXAI.value,
    "google_anthropic_vertex": OpenInferenceLLMSystemValues.ANTHROPIC.value,
    "google_genai": OpenInferenceLLMSystemValues.VERTEXAI.value,
    "google_vertexai": OpenInferenceLLMSystemValues.VERTEXAI.value,
    "groq": OpenInferenceLLMSystemValues.OPENAI.value,
    "huggingface": _NA,
    "ibm": _NA,
    "mistralai": OpenInferenceLLMSystemValues.MISTRALAI.value,
    "ollama": OpenInferenceLLMSystemValues.OPENAI.value,
    "openai": OpenInferenceLLMSystemValues.OPENAI.value,
    "perplexity": _NA,
    "together": _NA,
    "vertex": OpenInferenceLLMSystemValues.VERTEXAI.value,
    "vertexai": OpenInferenceLLMSystemValues.VERTEXAI.value,
    "xai": _NA,
}
