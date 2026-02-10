from typing import Any, Dict, Mapping, Optional, cast

import openai
import pytest
from openai.types.chat import ChatCompletionUserMessageParam
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

# Import OpenLLMetry constants from the official package
from opentelemetry.semconv_ai import (
    SpanAttributes as GenAISpanAttributes,
)
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation.openllmetry import OpenInferenceSpanProcessor
from openinference.instrumentation.openllmetry._span_processor import (
    _extract_llm_provider_and_system,
)
from openinference.semconv.trace import (
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


@pytest.fixture
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-0123456789")


def is_openinference_span(span: ReadableSpan) -> bool:
    """Check if a span is an OpenInference span."""
    if span.attributes is None:
        return False
    return SpanAttributes.OPENINFERENCE_SPAN_KIND in span.attributes


def remove_all_vcr_request_headers(request: Any) -> Any:
    """
    Removes all request headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_request_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    request.headers.clear()
    return request


def remove_all_vcr_response_headers(response: dict[str, Any]) -> dict[str, Any]:
    """
    Removes all response headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_response_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    response["headers"] = {}
    return response


class TestOpenLLMetryInstrumentor:
    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
    )
    def test_openllmetry_instrumentor(
        self,
        openai_api_key: str,
    ) -> None:
        in_memory_span_exporter = InMemorySpanExporter()
        in_memory_span_exporter.clear()

        # Set up the tracer provider
        tracer_provider = TracerProvider()

        tracer_provider.add_span_processor(OpenInferenceSpanProcessor())

        tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

        # Define and invoke your OpenAI model
        client = openai.OpenAI()

        messages: list[ChatCompletionUserMessageParam] = [
            {"role": "user", "content": "What is the capital of Yemen?"}
        ]

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
        )

        # Basic assertion on the result
        assert response.choices[0].message.content is not None
        assert (
            "Sanaa" in response.choices[0].message.content
            or "Sana'a" in response.choices[0].message.content
        )

        # Get spans
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]

        # Get attributes
        attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

        # OpenInference span kind
        assert is_openinference_span(span)
        assert (
            attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]
            == OpenInferenceSpanKindValues.LLM.value
        )

        # Input / Output
        assert isinstance(attributes[SpanAttributes.INPUT_VALUE], str)
        assert attributes[SpanAttributes.INPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
        assert isinstance(attributes[SpanAttributes.OUTPUT_VALUE], str)
        assert attributes[SpanAttributes.OUTPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value

        # LLM identity
        assert attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-4.1"
        assert attributes[SpanAttributes.LLM_SYSTEM] == OpenInferenceLLMSystemValues.OPENAI.value
        assert isinstance(attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS], str)
        total_tokens = attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL)
        assert isinstance(total_tokens, (int, float))
        assert total_tokens > 0

        # LLM messages
        assert attributes["llm.input_messages.0.message.role"] == "user"
        assert isinstance(attributes["llm.input_messages.0.message.content"], str)
        assert attributes["llm.output_messages.0.message.role"] == "assistant"
        assert isinstance(attributes["llm.output_messages.0.message.content"], str)

    def test_openinference_span_processor_creation(self) -> None:
        """Test that the OpenInference span processor can be created and used."""
        # Create the span processor
        span_processor = OpenInferenceSpanProcessor()

        # Verify it's created successfully
        assert span_processor is not None
        assert hasattr(span_processor, "on_start")
        assert hasattr(span_processor, "on_end")

        # Test that it can be added to a tracer provider
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(span_processor)

        assert True


def _set_span_attributes(
    *,
    provider: Any = None,
    system: Any = None,
) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    attrs[GenAIAttributes.GEN_AI_PROVIDER_NAME] = provider
    attrs[GenAISpanAttributes.LLM_SYSTEM] = system
    return attrs


@pytest.mark.parametrize(
    "attrs, expected_provider, expected_system",
    [
        # OpenAI system only
        (
            _set_span_attributes(system="openai"),
            None,
            OpenInferenceLLMSystemValues.OPENAI.value,
        ),
        # Anthropic system only
        (
            _set_span_attributes(system="Anthropic"),
            None,
            OpenInferenceLLMSystemValues.ANTHROPIC.value,
        ),
        # Mistral system only
        (
            _set_span_attributes(system="MISTRALAI"),
            None,
            OpenInferenceLLMSystemValues.MISTRALAI.value,
        ),
        # Vertex AI system only
        (
            _set_span_attributes(system="VertexAI"),
            None,
            OpenInferenceLLMSystemValues.VERTEXAI.value,
        ),
        # Google Generative AI (provider present)
        (
            _set_span_attributes(provider="Google", system="VertexAI"),
            OpenInferenceLLMProviderValues.GOOGLE.value,
            OpenInferenceLLMSystemValues.VERTEXAI.value,
        ),
        # Valid provider, Invalid system
        (
            _set_span_attributes(provider="OpenAI", system="Langchain"),
            OpenInferenceLLMProviderValues.OPENAI.value,
            None,
        ),
        # Invalid provider, Valid system
        (
            _set_span_attributes(provider="CrewAI", system="OpenAI"),
            None,
            OpenInferenceLLMSystemValues.OPENAI.value,
        ),
        # Both invalid values
        (
            _set_span_attributes(provider="HuggingFace", system="HuggingFace"),
            None,
            None,
        ),
        # Explicit None values
        (
            _set_span_attributes(provider=None, system=None),
            None,
            None,
        ),
        # Empty span attributes
        (
            {},
            None,
            None,
        ),
    ],
)
def test_extract_llm_provider_and_system(
    attrs: Dict[str, Any],
    expected_provider: Optional[str],
    expected_system: Optional[str],
) -> None:
    provider, system = _extract_llm_provider_and_system(attrs)

    assert provider == expected_provider
    assert system == expected_system
