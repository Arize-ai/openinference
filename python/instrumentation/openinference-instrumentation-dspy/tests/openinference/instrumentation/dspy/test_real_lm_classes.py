"""Test provider and model extraction with real DSPy LM classes."""

from typing import Optional

import dspy
import pytest

from openinference.instrumentation.dspy import (
    _llm_model_name,
    _llm_provider,
    parse_provider_and_model,
)


@pytest.mark.parametrize(
    "model_string,expected_provider,expected_model",
    [
        pytest.param("openai/gpt-4", "openai", "gpt-4", id="openai-gpt4"),
        pytest.param("openai/gpt-3.5-turbo", "openai", "gpt-3.5-turbo", id="openai-gpt35"),
        pytest.param(
            "anthropic/claude-3-opus-20240229",
            "anthropic",
            "claude-3-opus-20240229",
            id="anthropic-claude3",
        ),
        pytest.param("cohere/command-r-plus", "cohere", "command-r-plus", id="cohere-command"),
        pytest.param(
            "databricks/databricks-meta-llama-3-1-70b-instruct",
            "databricks",
            "databricks-meta-llama-3-1-70b-instruct",
            id="databricks-llama",
        ),
        pytest.param(
            "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "together",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            id="together-llama",
        ),
        pytest.param("gpt-4", None, "gpt-4", id="simple-model-string"),
        pytest.param("claude-3", None, "claude-3", id="simple-claude"),
    ],
)
def test_parse_provider_and_model(
    model_string: str,
    expected_provider: Optional[str],
    expected_model: Optional[str],
) -> None:
    """Test parse_provider_and_model function with various model string formats."""
    provider, model_name = parse_provider_and_model(model_string)
    assert provider == expected_provider, f"Expected provider {expected_provider}, got {provider}"
    assert model_name == expected_model, f"Expected model {expected_model}, got {model_name}"


@pytest.mark.parametrize(
    "model_string,expected_provider,expected_model",
    [
        pytest.param("openai/gpt-4", "openai", "gpt-4", id="openai-gpt4"),
        pytest.param("openai/gpt-3.5-turbo", "openai", "gpt-3.5-turbo", id="openai-gpt35"),
    ],
)
def test_llm_provider_and_model_extraction(
    model_string: str,
    expected_provider: str,
    expected_model: str,
    openai_api_key: str,
) -> None:
    """Test _llm_provider and _llm_model_name functions with real LM instances."""
    # Create LM instance
    lm = dspy.LM(model_string)

    # Test _llm_provider function
    provider_results = list(_llm_provider(lm))
    assert len(provider_results) == 1, f"Expected 1 provider result, got {len(provider_results)}"
    assert provider_results[0][0] == "llm.provider"
    assert provider_results[0][1] == expected_provider

    # Test _llm_model_name function
    model_results = list(_llm_model_name(lm))
    assert len(model_results) == 1, f"Expected 1 model result, got {len(model_results)}"
    assert model_results[0][0] == "llm.model_name"
    assert model_results[0][1] == expected_model


def test_simple_model_string_extraction(openai_api_key: str) -> None:
    """Test extraction with a simple model string (no provider prefix)."""
    lm = dspy.LM("gpt-4")

    # Test _llm_model_name - should extract model name
    model_results = list(_llm_model_name(lm))
    assert len(model_results) == 1
    assert model_results[0][0] == "llm.model_name"
    assert model_results[0][1] == "gpt-4"

    # Test _llm_provider - might not extract provider for simple strings
    provider_results = list(_llm_provider(lm))
    # Provider extraction depends on the LM implementation
    assert len(provider_results) <= 1


def test_provider_class_name_extraction(openai_api_key: str) -> None:
    """Test that we correctly extract provider names from provider class names."""
    lm = dspy.LM("openai/gpt-4")

    # Verify the provider attribute exists and has the expected class name
    assert hasattr(lm, "provider")
    assert lm.provider.__class__.__name__ == "OpenAIProvider"

    # Test provider extraction
    provider_results = list(_llm_provider(lm))
    assert len(provider_results) == 1
    assert provider_results[0][0] == "llm.provider"
    assert provider_results[0][1] == "openai"


def test_edge_cases_parse_provider_and_model() -> None:
    """Test edge cases for parse_provider_and_model function."""
    # Test with trailing slash
    provider, model = parse_provider_and_model("openai/gpt-4/")
    assert provider == "openai"
    assert model == "gpt-4"

    # Test with empty string
    provider, model = parse_provider_and_model("")
    assert provider is None
    assert model is None

    # Test with None
    provider, model = parse_provider_and_model(None)
    assert provider is None
    assert model is None

    # Test with prefixed format
    provider, model = parse_provider_and_model("text-completion-openai/gpt-3.5-turbo-instruct")
    assert provider == "openai"
    assert model == "gpt-3.5-turbo-instruct"


def test_functions_yield_once(openai_api_key: str) -> None:
    """Test that _llm_provider and _llm_model_name only yield once."""
    lm = dspy.LM("openai/gpt-4")

    # Test _llm_provider yields exactly once
    provider_results = list(_llm_provider(lm))
    assert len(provider_results) == 1

    # Test _llm_model_name yields exactly once
    model_results = list(_llm_model_name(lm))
    assert len(model_results) == 1
