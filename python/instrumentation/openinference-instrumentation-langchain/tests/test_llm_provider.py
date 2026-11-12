import pytest

from openinference.instrumentation.langchain._tracer import _llm_provider
from openinference.semconv.trace import OpenInferenceLLMProviderValues, SpanAttributes


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
