"""OpenInference observer for Pipecat pipelines."""

import logging
from typing import Optional

from opentelemetry import trace as trace_api
from opentelemetry import context as context_api
from pipecat.observers.base_observer import BaseObserver, FramePushed, FrameProcessed

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.pipecat._attributes import _FrameAttributeExtractor
from openinference.instrumentation.pipecat._service_detector import _ServiceDetector
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    EndFrame,
    ErrorFrame,
)

logger = logging.getLogger(__name__)


class OpenInferenceObserver(BaseObserver):
    """
    Observer that creates OpenInference spans for Pipecat frame processing.

    Observes frame flow through pipeline and creates spans for LLM, TTS, and STT services.
    Implements proper span hierarchy with session ID propagation.
    """

    def __init__(
        self,
        tracer: OITracer,
        config: TraceConfig,
        conversation_id: Optional[str] = None,
    ):
        """
        Initialize the observer.

        Args:
            tracer: OpenInference tracer
            config: Trace configuration
            conversation_id: Optional conversation/session ID to link all spans
        """
        super().__init__()
        self._tracer = tracer
        self._config = config
        self._detector = _ServiceDetector()
        self._attribute_extractor = _FrameAttributeExtractor()

        # Session management
        self._conversation_id = conversation_id

        # Track active spans per service instance
        # Key: id(service), Value: {"span": span, "frame_count": int}
        self._active_spans = {}

        # Track the last frame seen from each service to detect completion
        self._last_frames = {}

        # Turn tracking state
        self._turn_active = False
        self._turn_span = None
        self._turn_context_token = None  # Token for turn span context
        self._turn_number = 0
        self._turn_user_text = []
        self._turn_bot_text = []
        self._bot_speaking = False
        self._user_speaking = False

    async def on_push_frame(self, data: FramePushed):
        """
        Called when a frame is pushed between processors.

        Args:
            data: FramePushed event data with source, destination, frame, direction
        """
        try:

            frame = data.frame

            # Handle turn tracking frames
            if isinstance(frame, UserStartedSpeakingFrame):
                # If bot is speaking, this is an interruption
                if self._bot_speaking and self._turn_active:
                    await self._finish_turn(interrupted=True)
                # Start a new turn when user begins speaking (if not already active)
                if not self._turn_active:
                    await self._start_turn()
            elif isinstance(frame, TranscriptionFrame):
                # Collect user input during turn
                if self._turn_active and frame.text:
                    self._turn_user_text.append(frame.text)
            elif isinstance(frame, BotStartedSpeakingFrame):
                self._bot_speaking = True
                # Start a new turn when bot begins speaking (if not already active)
                # This handles the case where bot speaks first (e.g., greeting)
                if not self._turn_active:
                    await self._start_turn()
            elif isinstance(frame, TextFrame):
                # Collect bot output during turn
                if self._turn_active and self._bot_speaking and frame.text:
                    self._turn_bot_text.append(frame.text)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._bot_speaking = False
                # Turn ends when bot finishes speaking
                await self._finish_turn(interrupted=False)

            # Detect if source is a service we care about
            service_type = self._detector.detect_service_type(data.source)

            if service_type:
                await self._handle_service_frame(data, service_type)

        except Exception as e:
            logger.debug(f"Error in observer: {e}")

    async def on_process_frame(self, data: FrameProcessed):
        """
        Called when a frame is being processed.

        Args:
            data: FrameProcessed event data
        """
        # For now, we only care about push events
        pass

    async def _handle_service_frame(self, data: FramePushed, service_type: str):
        """
        Handle frame from an LLM, TTS, or STT service.

        Args:
            data: FramePushed event data
            service_type: "llm", "tts", or "stt"
        """
        from pipecat.frames.frames import EndFrame, ErrorFrame

        service = data.source
        service_id = id(service)
        frame = data.frame

        # Check if we already have a span for this service
        if service_id not in self._active_spans:
            # Create new span and set as active
            span = self._create_service_span(service, service_type)
            self._active_spans[service_id] = {
                "span": span,
                "frame_count": 0,
                "service_type": service_type,
            }

        # Increment frame count for this service
        span_info = self._active_spans[service_id]
        span_info["frame_count"] += 1

        # Extract and add attributes from this frame to the span
        span = span_info["span"]
        frame_attrs = self._attribute_extractor.extract_from_frame(frame)
        for key, value in frame_attrs.items():
            span.set_attribute(key, value)

        # Store this as the last frame from this service
        self._last_frames[service_id] = frame

        # Finish span only on completion frames (EndFrame or ErrorFrame)
        if isinstance(frame, (EndFrame, ErrorFrame)):
            self._finish_span(service_id)

    def _create_service_span(self, service, service_type: str):
        """
        Create a span for a service.

        Args:
            service: The service instance
            service_type: "llm", "tts", or "stt"

        Returns:
            The created span
        """
        # Extract metadata
        metadata = self._detector.extract_service_metadata(service)

        # Create span name
        span_name = f"pipecat.{service_type}"

        # Build attributes - use LLM span kind for LLM services, CHAIN for others
        if service_type == "llm":
            span_kind = OpenInferenceSpanKindValues.LLM.value
        else:
            span_kind = OpenInferenceSpanKindValues.CHAIN.value

        attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: span_kind,
            "service.name": metadata.get("provider", "unknown"),
        }

        # Add session.id if conversation_id is available
        if self._conversation_id:
            attributes[SpanAttributes.SESSION_ID] = self._conversation_id

        # Add LLM-specific attributes
        if service_type == "llm":
            if "provider" in metadata:
                attributes[SpanAttributes.LLM_PROVIDER] = metadata["provider"]
            if "model" in metadata:
                attributes[SpanAttributes.LLM_MODEL_NAME] = metadata["model"]
        # Add model for non-LLM services
        elif "model" in metadata:
            attributes["model"] = metadata["model"]

        # Add voice if available (TTS)
        if "voice" in metadata:
            attributes["voice"] = metadata["voice"]

        if "voice_id" in metadata:
            attributes["voice_id"] = metadata["voice_id"]

        # Create span - it will automatically be a child of the current context (turn span)
        # The turn context was already set via context_api.attach() in _start_turn()
        span = self._tracer.start_span(
            name=span_name,
            attributes=attributes,
        )

        logger.debug(
            f"Created {span_kind} span {span_name} for {metadata.get('provider')} {service_type}"
        )

        return span

    def _finish_span(self, service_id: int):
        """
        Finish a span for a service.

        Args:
            service_id: The id() of the service instance
        """
        if service_id not in self._active_spans:
            return

        span_info = self._active_spans.pop(service_id)
        span = span_info["span"]

        # End the span with OK status
        span.set_status(trace_api.Status(trace_api.StatusCode.OK))
        span.end()

        logger.debug(
            f"Finished span {span.name} after {span_info['frame_count']} frames"
        )

        # Clean up last frame tracking
        self._last_frames.pop(service_id, None)

    async def _start_turn(self):
        """Start a new conversation turn and set it as parent context."""
        # Increment turn number
        self._turn_number += 1

        # Create turn span - use ROOT context to avoid inheriting from any active span
        # This ensures turn spans are top-level spans (only inheriting session.id from context attributes)
        from opentelemetry.trace import set_span_in_context, INVALID_SPAN
        from opentelemetry.context import get_current

        span_name = "pipecat.conversation.turn"
        attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "conversation.turn_number": self._turn_number,
        }

        # Add session.id if conversation_id is available
        if self._conversation_id:
            attributes[SpanAttributes.SESSION_ID] = self._conversation_id

        # Create a context with no parent span (ROOT context)
        # This will still inherit context attributes like session.id
        root_context = set_span_in_context(INVALID_SPAN, get_current())

        self._turn_span = self._tracer.start_span(
            name=span_name,
            attributes=attributes,
            context=root_context,
        )

        # Set turn span as active context so service spans become children
        ctx = trace_api.set_span_in_context(self._turn_span)
        self._turn_context_token = context_api.attach(ctx)

        # Reset turn state
        self._turn_active = True
        self._turn_user_text = []
        self._turn_bot_text = []

        logger.debug(f"Started turn {self._turn_number} (span context set as parent)")

    async def _finish_turn(self, interrupted: bool = False):
        """
        Finish the current conversation turn and detach context.

        Args:
            interrupted: Whether the turn was interrupted
        """
        if not self._turn_active or not self._turn_span:
            return

        # Finish any active service spans before finishing the turn
        # This ensures service spans are closed even if EndFrame doesn't reach them
        service_ids_to_finish = list(self._active_spans.keys())
        for service_id in service_ids_to_finish:
            self._finish_span(service_id)

        # Set input/output attributes
        if self._turn_user_text:
            user_input = " ".join(self._turn_user_text)
            self._turn_span.set_attribute(SpanAttributes.INPUT_VALUE, user_input)

        if self._turn_bot_text:
            bot_output = " ".join(self._turn_bot_text)
            self._turn_span.set_attribute(SpanAttributes.OUTPUT_VALUE, bot_output)

        # Set end reason
        end_reason = "interrupted" if interrupted else "completed"
        self._turn_span.set_attribute("conversation.end_reason", end_reason)

        # Finish span
        self._turn_span.set_status(trace_api.Status(trace_api.StatusCode.OK))
        self._turn_span.end()

        # Detach turn context
        if self._turn_context_token is not None:
            context_api.detach(self._turn_context_token)
            self._turn_context_token = None

        logger.debug(
            f"Finished turn {self._turn_number} ({end_reason}) - "
            f"input: {len(self._turn_user_text)} chunks, "
            f"output: {len(self._turn_bot_text)} chunks"
        )

        # Reset turn state
        self._turn_active = False
        self._turn_span = None
