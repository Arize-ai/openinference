from types import SimpleNamespace

import pytest
from openinference.semconv.trace import OpenInferenceLLMProviderValues

from openinference.instrumentation.autogen._utils import (
    extract_llm_model_name_from_agent,
    infer_llm_provider_from_model,
)


@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("gpt-4o", OpenInferenceLLMProviderValues.OPENAI.value),
        ("gpt.4", OpenInferenceLLMProviderValues.OPENAI.value),
        ("o3-mini", OpenInferenceLLMProviderValues.OPENAI.value),
        ("o4-preview", OpenInferenceLLMProviderValues.OPENAI.value),
        ("claude-3-opus", OpenInferenceLLMProviderValues.ANTHROPIC.value),
        ("anthropic.claude-v2", OpenInferenceLLMProviderValues.ANTHROPIC.value),
        ("mistral-large", OpenInferenceLLMProviderValues.MISTRALAI.value),
        ("mixtral-8x7b", OpenInferenceLLMProviderValues.MISTRALAI.value),
        ("command-r", OpenInferenceLLMProviderValues.COHERE.value),
        ("cohere.command-light", OpenInferenceLLMProviderValues.COHERE.value),
        ("gemini-pro", OpenInferenceLLMProviderValues.GOOGLE.value),
        ("grok-2", OpenInferenceLLMProviderValues.XAI.value),
        ("deepseek-chat", OpenInferenceLLMProviderValues.DEEPSEEK.value),
    ],
)
def test_infer_llm_provider_from_model_known(model_name, expected):
    assert infer_llm_provider_from_model(model_name).value == expected


@pytest.mark.parametrize(
    "model_name",
    [
        None,
        "",
        "unknown-model",
        "custom-llm-v1",
    ],
)
def test_infer_llm_provider_from_model_unknown(model_name):
    assert infer_llm_provider_from_model(model_name) is None


def test_infer_llm_provider_is_case_insensitive():
    assert (
        infer_llm_provider_from_model("GPT-4O").value == OpenInferenceLLMProviderValues.OPENAI.value
    )


def test_extract_model_from_direct_model_field():
    agent = SimpleNamespace(
        llm_config={
            "model": "gpt-4o",
        }
    )

    assert extract_llm_model_name_from_agent(agent) == "gpt-4o"


def test_extract_model_from_config_list():
    agent = SimpleNamespace(
        llm_config={
            "config_list": [
                {"model": "claude-3-opus"},
            ]
        }
    )

    assert extract_llm_model_name_from_agent(agent) == "claude-3-opus"


def test_extract_model_prefers_direct_model_over_config_list():
    agent = SimpleNamespace(
        llm_config={
            "model": "gpt-4o",
            "config_list": [
                {"model": "claude-3-opus"},
            ],
        }
    )

    assert extract_llm_model_name_from_agent(agent) == "gpt-4o"


def test_extract_model_with_invalid_llm_config_type():
    agent = SimpleNamespace(llm_config="not-a-dict")

    assert extract_llm_model_name_from_agent(agent) is None


def test_extract_model_with_missing_llm_config():
    agent = SimpleNamespace()

    assert extract_llm_model_name_from_agent(agent) is None


def test_extract_model_with_empty_config_list():
    agent = SimpleNamespace(
        llm_config={
            "config_list": [],
        }
    )

    assert extract_llm_model_name_from_agent(agent) is None


def test_extract_model_with_non_dict_config_list_entry():
    agent = SimpleNamespace(
        llm_config={
            "config_list": ["not-a-dict"],
        }
    )

    assert extract_llm_model_name_from_agent(agent) is None
