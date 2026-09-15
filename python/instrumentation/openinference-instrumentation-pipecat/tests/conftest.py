"""Pytest configuration and shared fixtures for Pipecat instrumentation tests."""

from typing import Generator
from unittest.mock import Mock

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.pipecat import PipecatInstrumentor
from openinference.instrumentation.pipecat._observer import OpenInferenceObserver


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    """Create an in-memory span exporter for testing."""
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    """Create a TracerProvider configured with in-memory span export."""
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def tracer(tracer_provider: TracerProvider) -> OITracer:
    """Create an OITracer instance for testing."""
    config = TraceConfig()
    return OITracer(
        trace_api.get_tracer(__name__, "1.0.0", tracer_provider),
        config=config,
    )


@pytest.fixture()
def config() -> TraceConfig:
    """Create a TraceConfig instance for testing."""
    return TraceConfig()


@pytest.fixture()
def observer(tracer: OITracer, config: TraceConfig) -> OpenInferenceObserver:
    """Create an OpenInferenceObserver instance for testing."""
    return OpenInferenceObserver(tracer=tracer, config=config)


@pytest.fixture()
def setup_pipecat_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    """Set up and tear down Pipecat instrumentation for tests."""
    PipecatInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    PipecatInstrumentor().uninstrument()


@pytest.fixture()
def mock_llm_service() -> Mock:
    """Create a mock LLM service for testing."""

    # Create a real subclass to get proper MRO for service type detection
    class TestLLMService(LLMService):
        pass

    service = Mock(spec=TestLLMService)
    service.__class__ = TestLLMService
    service.__class__.__module__ = "pipecat.services.openai"
    service.model_name = "gpt-4"
    service._settings = {"temperature": 0.7, "max_tokens": 100}
    return service


@pytest.fixture()
def mock_stt_service() -> Mock:
    """Create a mock STT service for testing."""

    # Create a real subclass to get proper MRO for service type detection
    class TestSTTService(STTService):
        pass

    service = Mock(spec=TestSTTService)
    service.__class__ = TestSTTService
    service.__class__.__module__ = "pipecat.services.deepgram"
    service.model_name = "nova-2"
    service.sample_rate = 16000
    return service


@pytest.fixture()
def mock_tts_service() -> Mock:
    """Create a mock TTS service for testing."""

    # Create a real subclass to get proper MRO for service type detection
    class TestTTSService(TTSService):
        pass

    service = Mock(spec=TestTTSService)
    service.__class__ = TestTTSService
    service.__class__.__module__ = "pipecat.services.cartesia"
    service.model_name = "sonic-english"
    service._voice_id = "female-1"
    service.sample_rate = 22050
    return service
