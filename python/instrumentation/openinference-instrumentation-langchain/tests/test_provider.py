from typing import Any, Dict, Optional

import pytest

from openinference.instrumentation.langchain._tracer import (
    _llm_provider,
    _llm_system,
)
from openinference.semconv.trace import (
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    SpanAttributes,
)


@pytest.mark.parametrize(
    "extra,expected_provider",
    [
        # Provider from ls_provider in metadata (LangChain's source of truth)
        (
            {"metadata": {"ls_provider": "openai"}},
            OpenInferenceLLMProviderValues.OPENAI.value,
        ),
        (
            {"metadata": {"ls_provider": "anthropic"}},
            OpenInferenceLLMProviderValues.ANTHROPIC.value,
        ),
        (
            {"metadata": {"ls_provider": "google"}},
            OpenInferenceLLMProviderValues.GOOGLE.value,
        ),
        (
            {"metadata": {"ls_provider": "azure"}},
            OpenInferenceLLMProviderValues.AZURE.value,
        ),
        (
            {"metadata": {"ls_provider": "cohere"}},
            "cohere",
        ),
        (
            {"metadata": {"ls_provider": "mistralai"}},
            "mistralai",
        ),
        (
            {"metadata": {"ls_provider": "ollama"}},
            "ollama",
        ),
        # Unknown provider (should use raw value)
        (
            {"metadata": {"ls_provider": "unknown_provider"}},
            "unknown_provider",
        ),
        # No provider info - no ls_provider in metadata
        (
            {"metadata": {"other_field": "value"}},
            None,
        ),
        # No metadata at all
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
        # System from ls_provider in metadata (LangChain's source of truth)
        (
            {"metadata": {"ls_provider": "openai"}},
            OpenInferenceLLMSystemValues.OPENAI.value,
        ),
        (
            {"metadata": {"ls_provider": "anthropic"}},
            OpenInferenceLLMSystemValues.ANTHROPIC.value,
        ),
        (
            {"metadata": {"ls_provider": "google"}},
            OpenInferenceLLMSystemValues.VERTEXAI.value,
        ),
        (
            {"metadata": {"ls_provider": "google_genai"}},
            OpenInferenceLLMSystemValues.VERTEXAI.value,
        ),
        (
            {"metadata": {"ls_provider": "vertex"}},
            OpenInferenceLLMSystemValues.VERTEXAI.value,
        ),
        (
            {"metadata": {"ls_provider": "vertexai"}},
            OpenInferenceLLMSystemValues.VERTEXAI.value,
        ),
        # Provider not mapped to system (should return None)
        (
            {"metadata": {"ls_provider": "cohere"}},
            None,
        ),
        (
            {"metadata": {"ls_provider": "unknown_provider"}},
            None,
        ),
        # No system info - no ls_provider in metadata
        (
            {"metadata": {"other_field": "value"}},
            None,
        ),
        # No metadata at all
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
def test_llm_system(extra: Optional[Dict[str, Any]], expected_system: Optional[str]) -> None:
    system_items = list(_llm_system(extra))
    if expected_system is None:
        assert len(system_items) == 0
    else:
        assert len(system_items) == 1
        attribute, value = system_items[0]
        assert attribute == SpanAttributes.LLM_SYSTEM
        assert value == expected_system
