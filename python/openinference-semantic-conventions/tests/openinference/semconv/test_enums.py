from openinference.semconv.trace import (
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
)


class TestOpenInferenceSpanKindValues:
    def test_values(self) -> None:
        assert {e: e.value for e in OpenInferenceSpanKindValues} == {
            OpenInferenceSpanKindValues.AGENT: "AGENT",
            OpenInferenceSpanKindValues.CHAIN: "CHAIN",
            OpenInferenceSpanKindValues.EMBEDDING: "EMBEDDING",
            OpenInferenceSpanKindValues.EVALUATOR: "EVALUATOR",
            OpenInferenceSpanKindValues.GUARDRAIL: "GUARDRAIL",
            OpenInferenceSpanKindValues.LLM: "LLM",
            OpenInferenceSpanKindValues.RERANKER: "RERANKER",
            OpenInferenceSpanKindValues.RETRIEVER: "RETRIEVER",
            OpenInferenceSpanKindValues.TOOL: "TOOL",
            OpenInferenceSpanKindValues.UNKNOWN: "UNKNOWN",
        }


class TestOpenInferenceMimeTypeValues:
    def test_values(self) -> None:
        assert {e: e.value for e in OpenInferenceMimeTypeValues} == {
            OpenInferenceMimeTypeValues.JSON: "application/json",
            OpenInferenceMimeTypeValues.TEXT: "text/plain",
        }


class TestOpenInferenceLLMSystemValues:
    def test_values(self) -> None:
        assert {e: e.value for e in OpenInferenceLLMSystemValues} == {
            OpenInferenceLLMSystemValues.ANTHROPIC: "anthropic",
            OpenInferenceLLMSystemValues.COHERE: "cohere",
            OpenInferenceLLMSystemValues.MISTRALAI: "mistralai",
            OpenInferenceLLMSystemValues.OPENAI: "openai",
            OpenInferenceLLMSystemValues.VERTEXAI: "vertexai",
        }


class TestOpenInferenceLLMProviderValues:
    def test_values(self) -> None:
        assert {e: e.value for e in OpenInferenceLLMProviderValues} == {
            OpenInferenceLLMProviderValues.ANTHROPIC: "anthropic",
            OpenInferenceLLMProviderValues.AWS: "aws",
            OpenInferenceLLMProviderValues.AZURE: "azure",
            OpenInferenceLLMProviderValues.COHERE: "cohere",
            OpenInferenceLLMProviderValues.GOOGLE: "google",
            OpenInferenceLLMProviderValues.MISTRALAI: "mistralai",
            OpenInferenceLLMProviderValues.OPENAI: "openai",
            OpenInferenceLLMProviderValues.XAI: "xai",
            OpenInferenceLLMProviderValues.DEEPSEEK: "deepseek",
        }
