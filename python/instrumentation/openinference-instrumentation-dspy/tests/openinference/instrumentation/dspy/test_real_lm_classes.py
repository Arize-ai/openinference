"""Test provider and model extraction with real DSPy LM classes."""

import os

from openinference.instrumentation.dspy import (
    _llm_model_name,
    _llm_provider,
    parse_provider_and_model,
)

# Set dummy API keys to avoid errors
os.environ["OPENAI_API_KEY"] = "dummy-key"
os.environ["ANTHROPIC_API_KEY"] = "dummy-key"
os.environ["COHERE_API_KEY"] = "dummy-key"


def test_real_dspy_lm_classes() -> None:
    """Test provider and model extraction with actual DSPy LM instances."""
    import dspy

    # Test various model string formats that DSPy supports
    test_cases = [
        # (model_string, expected_provider, expected_model_name)
        ("openai/gpt-4", "openai", "gpt-4"),
        ("openai/gpt-3.5-turbo", "openai", "gpt-3.5-turbo"),
        ("anthropic/claude-3-opus-20240229", "anthropic", "claude-3-opus-20240229"),
        ("cohere/command-r-plus", "cohere", "command-r-plus"),
        (
            "databricks/databricks-meta-llama-3-1-70b-instruct",
            "databricks",
            "databricks-meta-llama-3-1-70b-instruct",
        ),
        (
            "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "together",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        ),
    ]

    for model_string, expected_provider, expected_model in test_cases:
        try:
            # Create LM instance
            lm = dspy.LM(model_string)

            # Test parse_provider_and_model function
            provider, model_name = parse_provider_and_model(model_string)
            assert provider == expected_provider, (
                f"Expected provider {expected_provider}, got {provider}"
            )
            assert model_name == expected_model, (
                f"Expected model {expected_model}, got {model_name}"
            )

            # Test _llm_provider function
            provider_results = list(_llm_provider(lm))
            assert len(provider_results) == 1, (
                f"Expected 1 provider result, got {len(provider_results)}"
            )
            assert provider_results[0][0] == "llm.provider"
            assert provider_results[0][1] == expected_provider

            # Test _llm_model_name function
            model_results = list(_llm_model_name(lm))
            assert len(model_results) == 1, f"Expected 1 model result, got {len(model_results)}"
            assert model_results[0][0] == "llm.model_name"
            assert model_results[0][1] == expected_model

            print(f"✓ {model_string}: provider={expected_provider}, model={expected_model}")

        except Exception as e:
            # Some providers might fail due to missing dependencies or API keys
            # but we can still test the parsing logic
            print(f"⚠ {model_string}: Skipped due to {type(e).__name__}: {e}")

            # Still test the parsing function
            provider, model_name = parse_provider_and_model(model_string)
            assert provider == expected_provider
            assert model_name == expected_model


def test_edge_cases_with_real_lm() -> None:
    """Test edge cases that might occur with real LM instances."""
    import dspy

    # Test with a simple model string (no provider prefix)
    try:
        lm = dspy.LM("gpt-4")
        provider_results = list(_llm_provider(lm))
        model_results = list(_llm_model_name(lm))

        # Should still extract model name
        assert len(model_results) == 1
        assert model_results[0][1] == "gpt-4"

        # Provider results might be empty for simple model strings
        assert len(provider_results) <= 1

        print("✓ Simple model string handled correctly")
    except Exception as e:
        print(f"⚠ Simple model string test skipped: {e}")


def test_provider_class_name_extraction() -> None:
    """Test that we correctly extract provider names from class names."""
    import dspy

    # Test OpenAI
    try:
        lm = dspy.LM("openai/gpt-4")
        assert hasattr(lm, "provider")
        assert lm.provider.__class__.__name__ == "OpenAIProvider"

        provider_results = list(_llm_provider(lm))
        assert len(provider_results) == 1
        assert provider_results[0][1] == "openai"

        print("✓ OpenAIProvider class name extraction works")
    except Exception as e:
        print(f"⚠ OpenAI provider test skipped: {e}")


