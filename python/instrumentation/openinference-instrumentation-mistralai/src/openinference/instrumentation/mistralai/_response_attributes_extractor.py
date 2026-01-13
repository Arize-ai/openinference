import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    Tuple,
)

from opentelemetry.util.types import AttributeValue

from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    SpanAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from mistralai.models import ChatCompletionResponse

__all__ = ("_ResponseAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def get_attributes_from_response(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _get_attributes_from_chat_completion_response(response)


def _get_attributes_from_chat_completion_response(
    response: "ChatCompletionResponse",
) -> Iterator[Tuple[str, AttributeValue]]:
    if model := getattr(response, "model", None):
        yield SpanAttributes.LLM_MODEL_NAME, model

        if provider := infer_llm_provider_from_model(model):
            yield SpanAttributes.LLM_PROVIDER, provider.value

            if system := _PROVIDER_TO_SYSTEM.get(provider.value):
                yield SpanAttributes.LLM_SYSTEM, system

    if usage := getattr(response, "usage", None):
        yield from _get_attributes_from_completion_usage(usage)
    if (choices := getattr(response, "choices", None)) and isinstance(choices, Iterable):
        for choice in choices:
            if (index := _get_attribute_or_value(choice, "index")) is None:
                continue
            if message := _get_attribute_or_value(choice, "message"):
                for key, value in _get_attributes_from_chat_completion_message(message):
                    yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value


class _StreamResponseAttributesExtractor:
    def get_attributes_from_response(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _get_attributes_from_stream_chat_completion_response(response)


def _get_attributes_from_stream_chat_completion_response(
    response: Any,
) -> Iterator[Tuple[str, AttributeValue]]:
    data = response.data
    if model := data.get("model", None):
        yield SpanAttributes.LLM_MODEL_NAME, model

        if provider := infer_llm_provider_from_model(model):
            yield SpanAttributes.LLM_PROVIDER, provider.value

            if system := _PROVIDER_TO_SYSTEM.get(provider.value):
                yield SpanAttributes.LLM_SYSTEM, system

    if usage := data.get("usage", None):
        yield from _get_attributes_from_completion_usage(usage)
    if (choices := data.get("choices", None)) and isinstance(choices, Iterable):
        for choice in choices:
            if (index := _get_attribute_or_value(choice, "index")) is None:
                continue
            if message := _get_attribute_or_value(choice, "message"):
                for key, value in _get_attributes_from_chat_completion_message(message):
                    yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value


def _get_attributes_from_chat_completion_message(
    message: "ChatCompletionResponse",
) -> Iterator[Tuple[str, AttributeValue]]:
    if role := _get_attribute_or_value(message, "role"):
        yield MessageAttributes.MESSAGE_ROLE, role
    if content := _get_attribute_or_value(message, "content"):
        yield MessageAttributes.MESSAGE_CONTENT, content
    if (tool_calls := _get_attribute_or_value(message, "tool_calls")) and isinstance(
        tool_calls, Iterable
    ):
        for index, tool_call in enumerate(tool_calls):
            if function := _get_attribute_or_value(tool_call, "function"):
                if name := _get_attribute_or_value(function, "name"):
                    yield (
                        (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
                        ),
                        name,
                    )
                if arguments := _get_attribute_or_value(function, "arguments"):
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        arguments,
                    )


def _get_attributes_from_completion_usage(
    usage: object,
) -> Iterator[Tuple[str, AttributeValue]]:
    # openai.types.CompletionUsage
    # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/completion_usage.py#L8  # noqa: E501
    if (total_tokens := _get_attribute_or_value(usage, "total_tokens")) is not None:
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens
    if (prompt_tokens := _get_attribute_or_value(usage, "prompt_tokens")) is not None:
        yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if (completion_tokens := _get_attribute_or_value(usage, "completion_tokens")) is not None:
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens


def _get_attribute_or_value(
    obj: Any,
    attribute_name: str,
) -> Any:
    if (value := getattr(obj, attribute_name, None)) is not None or (
        hasattr(obj, "get") and callable(obj.get) and (value := obj.get(attribute_name)) is not None
    ):
        return value
    return None


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
