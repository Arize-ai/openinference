import pytest
from typing import Dict, Any, Optional

from openinference.instrumentation.langchain._tracer import parse_provider_and_model, _llm_provider, _llm_system
from openinference.semconv.trace import OpenInferenceLLMProviderValues, OpenInferenceLLMSystemValues, SpanAttributes


@pytest.mark.parametrize(
    "model_str,expected_provider,expected_model",
    [
        ("openai/gpt-4", "openai", "gpt-4"),
        ("text-completion-openai/gpt-3.5-turbo-instruct", "openai", "gpt-3.5-turbo-instruct"),
        ("anthropic/claude-2", "anthropic", "claude-2"),
        ("gpt-4", None, "gpt-4"),
        ("claude-2", None, "claude-2"),
        (None, None, None),
        ("", None, None),
    ],
)
def test_parse_provider_and_model(
    model_str: Optional[str], expected_provider: Optional[str], expected_model: Optional[str]
) -> None:
    provider, model = parse_provider_and_model(model_str)
    assert provider == expected_provider
    assert model == expected_model


@pytest.mark.parametrize(
    "extra,expected_provider",
    [
        # Provider directly in invocation_params
        (
            {"invocation_params": {"provider": "openai"}},
            OpenInferenceLLMProviderValues.OPENAI.value,
        ),
        (
            {"invocation_params": {"client_name": "OpenAIClient"}},
            OpenInferenceLLMProviderValues.OPENAI.value,
        ),
        # Provider from model name
        (
            {"invocation_params": {"model_name": "openai/gpt-4"}},
            OpenInferenceLLMProviderValues.OPENAI.value,
        ),
        (
            {"invocation_params": {"model": "text-completion-openai/gpt-3.5-turbo-instruct"}},
            OpenInferenceLLMProviderValues.OPENAI.value,
        ),
        # Provider from class name
        (
            {"id": ["langchain", "llms", "openai", "OpenAI"]},
            OpenInferenceLLMProviderValues.OPENAI.value,
        ),
        (
            {"id": ["langchain", "llms", "anthropic", "ChatAnthropic"]},
            OpenInferenceLLMProviderValues.ANTHROPIC.value,
        ),
        # No provider info
        (
            {"invocation_params": {"model_name": "gpt-4"}},
            None,
        ),
        (
            {},
            None,
        ),
        (
            None,
            None,
        ),
    ],
)
def test_llm_provider(extra: Optional[Dict[str, Any]], expected_provider: Optional[str]) -> None:
    provider_items = list(_llm_provider(extra))
    if expected_provider is None:
        assert len(provider_items) == 0
    else:
        assert len(provider_items) == 1
        attribute, value = provider_items[0]
        assert attribute == SpanAttributes.LLM_PROVIDER
        assert value == expected_provider


@pytest.mark.parametrize(
    "extra,expected_system",
    [
        # System from model name with provider prefix
        (
            {"invocation_params": {"model_name": "openai/gpt-4"}},
            OpenInferenceLLMSystemValues.OPENAI.value,
        ),
        (
            {"invocation_params": {"model": "anthropic/claude-2"}},
            OpenInferenceLLMSystemValues.ANTHROPIC.value,
        ),
        (
            {"invocation_params": {"model_name": "google/gemini-pro"}},
            OpenInferenceLLMSystemValues.VERTEXAI.value,
        ),
        # System from model name pattern
        (
            {"invocation_params": {"model_name": "gpt-4"}},
            OpenInferenceLLMSystemValues.OPENAI.value,
        ),
        (
            {"invocation_params": {"model": "claude-2"}},
            OpenInferenceLLMSystemValues.ANTHROPIC.value,
        ),
        (
            {"invocation_params": {"model_name": "gemini-pro"}},
            OpenInferenceLLMSystemValues.VERTEXAI.value,
        ),
        # System from class name
        (
            {"id": ["langchain", "llms", "openai", "OpenAI"]},
            OpenInferenceLLMSystemValues.OPENAI.value,
        ),
        (
            {"id": ["langchain", "llms", "anthropic", "ChatAnthropic"]},
            OpenInferenceLLMSystemValues.ANTHROPIC.value,
        ),
        (
            {"id": ["langchain", "llms", "vertex", "ChatVertexAI"]},
            OpenInferenceLLMSystemValues.VERTEXAI.value,
        ),
        # No system info
        (
            {"invocation_params": {"model_name": "unknown-model"}},
            None,
        ),
        (
            {},
            None,
        ),
        (
            None,
            None,
        ),
    ],
)
def test_llm_system(extra: Optional[Dict[str, Any]], expected_system: Optional[str]) -> None:
    system_items = list(_llm_system(extra))
    if expected_system is None:
        assert len(system_items) == 0
    else:
        assert len(system_items) == 1
        attribute, value = system_items[0]
        assert attribute == SpanAttributes.LLM_SYSTEM
        assert value == expected_system 