from typing import Any, Dict, Mapping, Optional, cast

import openlit  # type: ignore[import-untyped]
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import KernelArguments

from openinference.instrumentation.openlit import (
    OpenInferenceSpanProcessor,
)
from openinference.instrumentation.openlit._span_processor import (
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


@pytest.fixture
def openai_global_llm_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLOBAL_LLM_SERVICE", "OpenAI")


@pytest.fixture
def openai_chat_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_CHAT_MODEL_ID", "gpt-4o-mini")


@pytest.fixture
def openai_text_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_TEXT_MODEL_ID", "gpt-4o-mini")


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


class TestOpenLitInstrumentor:
    @pytest.mark.vcr(
        before_record_request=remove_all_vcr_request_headers,
        before_record_response=remove_all_vcr_response_headers,
        decode_compressed_response=True,
        filter_headers=["authorization"],
    )
    @pytest.mark.asyncio
    @pytest.mark.skip(
        reason="OpenLIT v1.36.8 has async generator bug preventing initialization. "
        "See https://github.com/openlit/openlit/issues/997. "
        "TODO: Re-enable when upstream bug is fixed."
    )
    async def test_openlit_instrumentor(
        self,
        openai_api_key: None,
        openai_global_llm_service: None,
        openai_chat_model_id: None,
        openai_text_model_id: None,
    ) -> None:
        in_memory_span_exporter = InMemorySpanExporter()
        in_memory_span_exporter.clear()

        # Set up the tracer provider
        tracer_provider = TracerProvider()

        tracer_provider.add_span_processor(OpenInferenceSpanProcessor())

        tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

        tracer = tracer_provider.get_tracer(__name__)

        # Initialize OpenLit with the tracer
        openlit.init(
            otel_tracer=tracer,
            otlp_endpoint=None,
        )

        # Set up Semantic Kernel
        kernel = Kernel()
        kernel.remove_all_services()

        service_id = "default"
        kernel.add_service(
            OpenAIChatCompletion(
                service_id=service_id,
                ai_model_id="gpt-4o-mini",
            ),
        )

        # Create a simple function that makes a joke request
        async def run_joke() -> Any:
            joke = await kernel.invoke_prompt(
                prompt="Tell me a short joke about programming",
                arguments=KernelArguments(),
            )
            return joke

        # Execute the joke function
        result = await run_joke()

        # Basic assertion on the result
        assert result is not None
        assert hasattr(result, "value")
        assert result.value is not None

        # Get spans
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) > 0

        for span in spans:
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
            assert (
                attributes[SpanAttributes.INPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
            )
            assert isinstance(attributes[SpanAttributes.OUTPUT_VALUE], str)
            assert (
                attributes[SpanAttributes.OUTPUT_MIME_TYPE]
                == OpenInferenceMimeTypeValues.JSON.value
            )

            # LLM identity
            assert attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-4o-mini"
            assert (
                attributes[SpanAttributes.LLM_SYSTEM] == OpenInferenceLLMSystemValues.OPENAI.value
            )
            assert isinstance(attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS], str)
            completion_tokens = attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION)
            assert isinstance(completion_tokens, (int, float))
            assert completion_tokens > 0
            prompt_tokens = attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT)
            assert isinstance(prompt_tokens, (int, float))
            assert prompt_tokens > 0

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
    attrs["gen_ai.llm.provider"] = provider
    attrs["gen_ai.system"] = system
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
