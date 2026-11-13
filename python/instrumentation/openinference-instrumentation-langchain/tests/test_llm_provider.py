import pytest

from openinference.instrumentation.langchain._tracer import _llm_provider, _llm_system
from openinference.semconv.trace import (
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    SpanAttributes,
)


@pytest.mark.parametrize(
    "ls_provider",
    [
        pytest.param("amazon_bedrock", id="ChatBedrock-ChatBedrockConverse-BedrockLLM"),
        pytest.param("anthropic-bedrock", id="ChatAnthropicBedrock"),
        pytest.param("anthropic-mantle", id="ChatAnthropicMantle"),
        pytest.param("openai-mantle", id="ChatOpenAIMantle"),
    ],
)
def test_langchain_aws_models_have_aws_provider(ls_provider: str) -> None:
    extra = {"metadata": {"ls_provider": ls_provider}}
    assert dict(_llm_provider(extra)) == {
        SpanAttributes.LLM_PROVIDER: OpenInferenceLLMProviderValues.AWS.value
    }


@pytest.mark.parametrize(
    "ls_provider",
    [
        pytest.param("mistral", id="ChatMistralAI"),
        pytest.param("mistralai", id="init_chat_model"),
    ],
)
def test_langchain_mistralai_models_have_mistralai_provider_and_system(ls_provider: str) -> None:
    extra = {"metadata": {"ls_provider": ls_provider}}
    assert dict(_llm_provider(extra)) == {
        SpanAttributes.LLM_PROVIDER: OpenInferenceLLMProviderValues.MISTRALAI.value
    }
    assert dict(_llm_system(extra)) == {
        SpanAttributes.LLM_SYSTEM: OpenInferenceLLMSystemValues.MISTRALAI.value
    }
