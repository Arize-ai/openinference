from openinference.semconv.trace import (
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
)


class TestOpenInferenceSpanKindValues:
    def test_values(self) -> None:
        assert {e.name: e.value for e in OpenInferenceSpanKindValues} == {
            "AGENT": "AGENT",
            "CHAIN": "CHAIN",
            "EMBEDDING": "EMBEDDING",
            "EVALUATOR": "EVALUATOR",
            "GUARDRAIL": "GUARDRAIL",
            "LLM": "LLM",
            "RERANKER": "RERANKER",
            "RETRIEVER": "RETRIEVER",
            "TOOL": "TOOL",
            "UNKNOWN": "UNKNOWN",
        }


class TestOpenInferenceMimeTypeValues:
    def test_values(self) -> None:
        assert {e.name: e.value for e in OpenInferenceMimeTypeValues} == {
            "JSON": "application/json",
            "TEXT": "text/plain",
        }


class TestOpenInferenceLLMSystemValues:
    def test_values(self) -> None:
        assert {e.name: e.value for e in OpenInferenceLLMSystemValues} == {
            "ANTHROPIC": "anthropic",
            "COHERE": "cohere",
            "MISTRALAI": "mistralai",
            "OPENAI": "openai",
            "VERTEXAI": "vertexai",
        }


class TestOpenInferenceLLMProviderValues:
    def test_values(self) -> None:
        assert {e.name: e.value for e in OpenInferenceLLMProviderValues} == {
            "ANTHROPIC": "anthropic",
            "AWS": "aws",
            "AZURE": "azure",
            "COHERE": "cohere",
            "GOOGLE": "google",
            "MISTRALAI": "mistralai",
            "OPENAI": "openai",
        }
