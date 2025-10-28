import os
import sys
from typing import Any, Mapping, cast

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
from openinference.semconv.trace import SpanAttributes


@pytest.fixture
def openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Only set fake API key if no real one is available
    real_api_key = os.environ.get("OPENAI_API_KEY")
    if not real_api_key or real_api_key == "sk-0123456789":
        monkeypatch.setenv("OPENAI_API_KEY", "sk-0123456789")
    # If real API key exists, don't override it


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
    @pytest.mark.skipif(
        sys.version_info < (3, 10), reason="semantic-kernel>=1.0.0 requires Python>=3.10"
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
            tracer=tracer,
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

        # Check that we have OpenInference spans
        openinference_spans = [span for span in spans if is_openinference_span(span)]
        assert len(openinference_spans) > 0, "No OpenInference spans found"

        for span in openinference_spans:
            # Check span attributes
            attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

            # Check that we have input and output values
            assert SpanAttributes.INPUT_VALUE in attributes
            assert SpanAttributes.OUTPUT_VALUE in attributes

            # Verify the model name is captured (exact attribute may vary)
            model_name_attr = next(
                (k for k in attributes if "model" in k.lower() and "name" in k.lower()), None
            )
            assert model_name_attr is not None, "Model name attribute not found"
            assert "gpt-4" in str(attributes[model_name_attr])

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
