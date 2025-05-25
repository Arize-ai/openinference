# type: ignore
from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.llama_index._handler import _Span
from openinference.semconv.trace import OpenInferenceLLMProviderValues, SpanAttributes


class TestLLMProvider:
    """Test LLM provider detection for various LlamaIndex LLM implementations."""

    def test_openai_provider_detection(
        self,
        tracer_provider: TracerProvider,
    ) -> None:
        """Test that OpenAI LLM instances are correctly identified with 'openai' provider."""
        from llama_index.llms.openai import OpenAI

        # Create a span directly and test the provider detection
        tracer = tracer_provider.get_tracer(__name__)
        otel_span = tracer.start_span("test")
        span = _Span(otel_span=otel_span)

        llm = OpenAI(model="gpt-4o-mini", api_key="sk-")
        span.process_instance(llm)

        assert span._attributes.get(LLM_PROVIDER) == OpenInferenceLLMProviderValues.OPENAI.value
        otel_span.end()

    def test_azure_openai_provider_detection(
        self,
        tracer_provider: TracerProvider,
    ) -> None:
        """Test that Azure OpenAI LLM instances are correctly identified with 'azure' provider."""
        from llama_index.llms.azure_openai import AzureOpenAI

        # Create a span directly and test the provider detection
        tracer = tracer_provider.get_tracer(__name__)
        otel_span = tracer.start_span("test")
        span = _Span(otel_span=otel_span)

        llm = AzureOpenAI(
            model="gpt-4o-mini",
            engine="gpt-4o-mini",  # Required parameter
            api_key="sk-",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2024-02-01",
        )
        span.process_instance(llm)

        assert span._attributes.get(LLM_PROVIDER) == OpenInferenceLLMProviderValues.AZURE.value
        otel_span.end()

    def test_anthropic_provider_detection(
        self,
        tracer_provider: TracerProvider,
    ) -> None:
        """Test that Anthropic LLM instances are correctly identified with 'anthropic' provider."""
        from llama_index.llms.anthropic import Anthropic

        # Create a span directly and test the provider detection
        tracer = tracer_provider.get_tracer(__name__)
        otel_span = tracer.start_span("test")
        span = _Span(otel_span=otel_span)

        llm = Anthropic(model="claude-3-5-haiku-20241022", api_key="sk-ant-")
        span.process_instance(llm)

        assert span._attributes.get(LLM_PROVIDER) == OpenInferenceLLMProviderValues.ANTHROPIC.value
        otel_span.end()

    def test_vertex_provider_detection(
        self,
        tracer_provider: TracerProvider,
    ) -> None:
        """Test that Vertex AI LLM instances are correctly identified with 'google' provider."""
        from llama_index.llms.vertex import Vertex

        # Create a span directly and test the provider detection
        tracer = tracer_provider.get_tracer(__name__)
        otel_span = tracer.start_span("test")
        span = _Span(otel_span=otel_span)

        llm = Vertex(model="gemini-1.5-flash", project="test-project", location="us-central1")
        span.process_instance(llm)

        assert span._attributes.get(LLM_PROVIDER) == OpenInferenceLLMProviderValues.GOOGLE.value
        otel_span.end()

    def test_fallback_provider_detection_by_class_name(
        self,
        tracer_provider: TracerProvider,
    ) -> None:
        """Test fallback provider detection based on class name for unknown LLM types."""
        from llama_index.core.base.llms.base import BaseLLM

        # Create a span directly and test the provider detection
        tracer = tracer_provider.get_tracer(__name__)
        otel_span = tracer.start_span("test")
        span = _Span(otel_span=otel_span)

        # Create a mock LLM class with a provider name in the class name
        class MockAnthropicLLM(BaseLLM):
            def __init__(self):
                super().__init__()

            @property
            def metadata(self):
                return None

            def chat(self, messages, **kwargs):
                pass

            def complete(self, prompt, **kwargs):
                pass

            def stream_chat(self, messages, **kwargs):
                pass

            def stream_complete(self, prompt, **kwargs):
                pass

            async def achat(self, messages, **kwargs):
                pass

            async def acomplete(self, prompt, **kwargs):
                pass

            async def astream_chat(self, messages, **kwargs):
                pass

            async def astream_complete(self, prompt, **kwargs):
                pass

            def _as_query_component(self, **kwargs):
                pass

        # Test with a mock LLM that has "anthropic" in the class name
        llm = MockAnthropicLLM()
        span.process_instance(llm)

        assert span._attributes.get(LLM_PROVIDER) == OpenInferenceLLMProviderValues.ANTHROPIC.value
        otel_span.end()

    def test_unknown_provider_no_attribute(
        self,
        tracer_provider: TracerProvider,
    ) -> None:
        """Test that unknown LLM types don't set a provider attribute."""
        from llama_index.core.base.llms.base import BaseLLM

        # Create a span directly and test the provider detection
        tracer = tracer_provider.get_tracer(__name__)
        otel_span = tracer.start_span("test")
        span = _Span(otel_span=otel_span)

        # Create a mock LLM class with no recognizable provider name
        class MockUnknownLLM(BaseLLM):
            def __init__(self):
                super().__init__()

            @property
            def metadata(self):
                return None

            def chat(self, messages, **kwargs):
                pass

            def complete(self, prompt, **kwargs):
                pass

            def stream_chat(self, messages, **kwargs):
                pass

            def stream_complete(self, prompt, **kwargs):
                pass

            async def achat(self, messages, **kwargs):
                pass

            async def acomplete(self, prompt, **kwargs):
                pass

            async def astream_chat(self, messages, **kwargs):
                pass

            async def astream_complete(self, prompt, **kwargs):
                pass

            def _as_query_component(self, **kwargs):
                pass

        # Test with a mock LLM that has no recognizable provider name
        llm = MockUnknownLLM()
        span.process_instance(llm)

        # Should not have LLM_PROVIDER attribute for unknown providers
        assert LLM_PROVIDER not in span._attributes
        otel_span.end()

    def test_google_class_name_fallback(
        self,
        tracer_provider: TracerProvider,
    ) -> None:
        """Test fallback provider detection for Google/Vertex class names."""
        from llama_index.core.base.llms.base import BaseLLM

        # Create a span directly and test the provider detection
        tracer = tracer_provider.get_tracer(__name__)
        otel_span = tracer.start_span("test")
        span = _Span(otel_span=otel_span)

        # Create a mock LLM class with "google" in the class name
        class MockGoogleLLM(BaseLLM):
            def __init__(self):
                super().__init__()

            @property
            def metadata(self):
                return None

            def chat(self, messages, **kwargs):
                pass

            def complete(self, prompt, **kwargs):
                pass

            def stream_chat(self, messages, **kwargs):
                pass

            def stream_complete(self, prompt, **kwargs):
                pass

            async def achat(self, messages, **kwargs):
                pass

            async def acomplete(self, prompt, **kwargs):
                pass

            async def astream_chat(self, messages, **kwargs):
                pass

            async def astream_complete(self, prompt, **kwargs):
                pass

            def _as_query_component(self, **kwargs):
                pass

        # Test with a mock LLM that has "google" in the class name
        llm = MockGoogleLLM()
        span.process_instance(llm)

        assert span._attributes.get(LLM_PROVIDER) == OpenInferenceLLMProviderValues.GOOGLE.value
        otel_span.end()


# Import the attribute constants
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
