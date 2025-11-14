"""
Shared test fixtures for Pipecat instrumentation tests.
"""

import asyncio
from typing import AsyncGenerator, List

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    Frame,
    LLMMessagesUpdateFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

# Mock Services for Testing


class MockLLMService(LLMService):
    """Mock LLM service for testing"""

    def __init__(self, *, model: str = "mock-model", provider: str = "mock", **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._model_name = model  # Set the private attribute directly
        self._provider = provider
        self.processed_frames = []
        # Set module to simulate provider
        self.__class__.__module__ = f"pipecat.services.{provider}.llm"

    async def process_frame(self, frame: Frame, direction):
        self.processed_frames.append(frame)
        if isinstance(frame, LLMMessagesUpdateFrame):
            # Simulate LLM response
            response = TextFrame(text="Mock LLM response")
            await self.push_frame(response, direction)
        return await super().process_frame(frame, direction)


class MockTTSService(TTSService):
    """Mock TTS service for testing"""

    def __init__(
        self,
        *,
        model: str = "mock-tts",
        voice: str = "mock-voice",
        provider: str = "mock",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model
        self._model_name = model  # Set the private attribute directly
        self._voice = voice
        self._voice_id = voice  # Real Pipecat services use _voice_id
        self._sample_rate = 16000  # Use private attribute for sample_rate
        self._provider = provider
        self.processed_texts = []

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Convert text to mock audio frames"""
        self.processed_texts.append(text)
        # Simulate audio frame generation
        audio_data = b"\x00" * 1024  # Mock audio data
        yield AudioRawFrame(audio=audio_data, sample_rate=16000, num_channels=1)


class MockSTTService(STTService):
    """Mock STT service for testing"""

    def __init__(self, *, model: str = "mock-stt", provider: str = "mock", **kwargs):
        super().__init__(**kwargs)
        self._model = model
        self._model_name = model  # Set the private attribute directly
        self._provider = provider
        self._user_id = "test-user"  # Add user_id for STT metadata extraction
        self._sample_rate = 16000  # Use private attribute for sample_rate
        self._muted = False  # Use private attribute for is_muted
        self.processed_audio = []

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Convert audio to mock transcription"""
        self.processed_audio.append(audio)
        # Simulate transcription
        yield TranscriptionFrame(text="Mock transcription", user_id="test-user", timestamp=0)


# Service Factory Functions - Better approach than multiple mock classes


def create_mock_service(service_class, provider: str, service_type: str, **kwargs):
    """
    Factory function to create mock services with proper provider attribution.

    Args:
        service_class: Base service class (MockLLMService, MockTTSService, MockSTTService)
        provider: Provider name (openai, anthropic, elevenlabs, deepgram)
        service_type: Service type (llm, tts, stt)
        **kwargs: Additional arguments passed to service constructor
    """
    # Create instance
    service = service_class(provider=provider, **kwargs)

    # Set module path to simulate real provider service
    service.__class__.__module__ = f"pipecat.services.{provider}.{service_type}"

    return service


# Convenience factory functions for common providers
def create_openai_llm(model: str = "gpt-4", **kwargs):
    """Create mock OpenAI LLM service"""
    return create_mock_service(MockLLMService, "openai", "llm", model=model, **kwargs)


def create_openai_tts(model: str = "tts-1", voice: str = "alloy", **kwargs):
    """Create mock OpenAI TTS service"""
    return create_mock_service(MockTTSService, "openai", "tts", model=model, voice=voice, **kwargs)


def create_openai_stt(model: str = "whisper-1", **kwargs):
    """Create mock OpenAI STT service"""
    return create_mock_service(MockSTTService, "openai", "stt", model=model, **kwargs)


def create_anthropic_llm(model: str = "claude-3-5-sonnet-20241022", **kwargs):
    """Create mock Anthropic LLM service"""
    return create_mock_service(MockLLMService, "anthropic", "llm", model=model, **kwargs)


def create_elevenlabs_tts(
    voice_id: str = "mock-voice-id", model: str = "eleven_turbo_v2", **kwargs
):
    """Create mock ElevenLabs TTS service"""
    service = create_mock_service(
        MockTTSService, "elevenlabs", "tts", model=model, voice=voice_id, **kwargs
    )
    service._voice_id = voice_id
    return service


def create_deepgram_stt(model: str = "nova-2", **kwargs):
    """Create mock Deepgram STT service"""
    return create_mock_service(MockSTTService, "deepgram", "stt", model=model, **kwargs)


def create_cartesia_tts(model: str = "sonic-english", voice_id: str = "mock-voice", **kwargs):
    """Create mock Cartesia TTS service"""
    return create_mock_service(
        MockTTSService, "cartesia", "tts", model=model, voice=voice_id, **kwargs
    )


# Fixtures


@pytest.fixture
def in_memory_span_exporter():
    """Create an in-memory span exporter for testing"""
    exporter = InMemorySpanExporter()
    yield exporter
    # Clear spans after each test
    exporter.clear()


@pytest.fixture
def tracer_provider(in_memory_span_exporter):
    """Create a tracer provider with in-memory exporter"""
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    trace_api.set_tracer_provider(provider)
    return provider


@pytest.fixture
def tracer(tracer_provider):
    """Create a tracer for testing"""
    return tracer_provider.get_tracer(__name__)


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service"""
    return MockLLMService()


@pytest.fixture
def mock_tts_service():
    """Create a mock TTS service"""
    return MockTTSService()


@pytest.fixture
def mock_stt_service():
    """Create a mock STT service"""
    return MockSTTService()


@pytest.fixture
def mock_openai_llm():
    """Create a mock OpenAI LLM service"""
    return create_openai_llm()


@pytest.fixture
def mock_openai_tts():
    """Create a mock OpenAI TTS service"""
    return create_openai_tts()


@pytest.fixture
def mock_openai_stt():
    """Create a mock OpenAI STT service"""
    return create_openai_stt()


@pytest.fixture
def mock_anthropic_llm():
    """Create a mock Anthropic LLM service"""
    return create_anthropic_llm()


@pytest.fixture
def mock_elevenlabs_tts():
    """Create a mock ElevenLabs TTS service"""
    return create_elevenlabs_tts()


@pytest.fixture
def mock_deepgram_stt():
    """Create a mock Deepgram STT service"""
    return create_deepgram_stt()


@pytest.fixture
def simple_pipeline(mock_stt_service, mock_llm_service, mock_tts_service):
    """Create a simple pipeline with STT -> LLM -> TTS"""
    return Pipeline([mock_stt_service, mock_llm_service, mock_tts_service])


@pytest.fixture
def openai_pipeline(mock_openai_stt, mock_openai_llm, mock_openai_tts):
    """Create a pipeline with OpenAI services"""
    return Pipeline([mock_openai_stt, mock_openai_llm, mock_openai_tts])


@pytest.fixture
def mixed_provider_pipeline(mock_deepgram_stt, mock_anthropic_llm, mock_elevenlabs_tts):
    """Create a pipeline with mixed service providers"""
    return Pipeline([mock_deepgram_stt, mock_anthropic_llm, mock_elevenlabs_tts])


@pytest.fixture
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def pipeline_task(simple_pipeline):
    """Create a pipeline task"""
    return PipelineTask(simple_pipeline)


def get_spans_by_name(exporter: InMemorySpanExporter, name: str) -> List:
    """Helper to get spans by name from exporter"""
    return [span for span in exporter.get_finished_spans() if span.name.startswith(name)]


def get_span_attributes(span) -> dict:
    """Helper to get span attributes as dict"""
    return dict(span.attributes) if span.attributes else {}


def assert_span_has_attributes(span, expected_attributes: dict):
    """Assert that span has expected attributes"""
    actual = get_span_attributes(span)
    for key, value in expected_attributes.items():
        assert key in actual, f"Attribute {key} not found in span"
        assert actual[key] == value, f"Expected {key}={value}, got {actual[key]}"


def assert_span_hierarchy(spans: List, expected_hierarchy: List[str]):
    """
    Assert that spans form the expected parent-child hierarchy.
    expected_hierarchy is a list of span names from root to leaf.
    """
    span_by_name = {span.name: span for span in spans}

    for i in range(len(expected_hierarchy) - 1):
        parent_name = expected_hierarchy[i]
        child_name = expected_hierarchy[i + 1]

        assert parent_name in span_by_name, f"Parent span {parent_name} not found"
        assert child_name in span_by_name, f"Child span {child_name} not found"

        parent_span = span_by_name[parent_name]
        child_span = span_by_name[child_name]

        assert child_span.parent.span_id == parent_span.context.span_id, (
            f"{child_name} is not a child of {parent_name}"
        )


async def run_pipeline_task(task: PipelineTask, *frames: Frame, send_start_frame: bool = True):
    """
    Helper to run a pipeline task with given frames.

    This simulates pipeline execution by manually triggering frame processing
    through the observers, which is sufficient for testing instrumentation.

    Args:
        task: The PipelineTask to run
        *frames: Frames to queue before running the task
        send_start_frame: Whether to send StartFrame first (default: True)
    """
    from pipecat.processors.frame_processor import FrameDirection

    # Mock data class for frame push events
    class MockFramePushData:
        def __init__(self, source, frame):
            import time

            self.source = source
            self.frame = frame
            self.destination = None
            self.direction = FrameDirection.DOWNSTREAM
            self.timestamp = time.time_ns()  # Nanoseconds for TurnTrackingObserver
            # Ensure frame has an id attribute for TurnTrackingObserver compatibility
            if not hasattr(frame, "id"):
                frame.id = id(frame)

    # Get the pipeline processors (services)
    # The structure is: task._pipeline._processors contains [Source, Pipeline, Sink]
    # The actual services are in the nested Pipeline._processors
    processors = []
    if hasattr(task, "_pipeline"):
        pipeline = task._pipeline
        if hasattr(pipeline, "_processors") and len(pipeline._processors) > 1:
            # The middle item is the actual Pipeline containing the services
            nested_pipeline = pipeline._processors[1]
            if hasattr(nested_pipeline, "_processors"):
                processors = nested_pipeline._processors

    # Get all observers from the task
    # The task has a TaskObserver wrapper which contains the actual observers
    observers = []
    if hasattr(task, "_observer") and task._observer:
        task_observer = task._observer
        # TaskObserver has _observers list containing the real observers
        if hasattr(task_observer, "_observers") and task_observer._observers:
            observers.extend(task_observer._observers)

    # Send StartFrame first to initialize first turn
    if send_start_frame:
        for processor in processors:
            for observer in observers:
                if hasattr(observer, "on_push_frame"):
                    await observer.on_push_frame(MockFramePushData(processor, StartFrame()))

    # Trigger observer callbacks for each frame through each processor
    for frame in frames:
        for processor in processors:
            # Notify all observers about this frame push
            for observer in observers:
                if hasattr(observer, "on_push_frame"):
                    await observer.on_push_frame(MockFramePushData(processor, frame))

    # Always send EndFrame to finish spans
    for processor in processors:
        for observer in observers:
            if hasattr(observer, "on_push_frame"):
                await observer.on_push_frame(MockFramePushData(processor, EndFrame()))
