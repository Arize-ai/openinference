"""OpenInference observer for Pipecat pipelines."""

import logging
import json
from datetime import datetime
from typing import Optional

from opentelemetry import trace as trace_api
from pipecat.observers.base_observer import BaseObserver, FramePushed, FrameProcessed

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.pipecat._attributes import _FrameAttributeExtractor
from openinference.instrumentation.pipecat._service_detector import _ServiceDetector
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
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
        debug_log_filename: Optional[str] = None,
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

        # Debug logging to file
        self._debug_log_file = None
        if debug_log_filename:
            # Write log to current working directory (where the script is running)
            try:
                self._debug_log_file = open(debug_log_filename, "w")
                self._log_debug(f"=== Observer initialized for conversation {conversation_id} ===")
                self._log_debug(f"=== Log file: {debug_log_filename} ===")
            except Exception as e:
                logger.error(f"Could not open debug log file: {e}")

        # Track active spans per service instance
        # Key: id(service), Value: {"span": span, "frame_count": int}
        self._active_spans = {}

        # Track the last frame seen from each service to detect completion
        self._last_frames = {}

        # Turn tracking state
        self._turn_active = False
        self._turn_span = None
        self._last_speaking_frame_id = None  # Deduplicate speaking frames from propagation
        self._turn_context_token = None  # Token for turn span context
        self._turn_number = 0
        self._turn_user_text = []
        self._turn_bot_text = []
        self._bot_speaking = False
        self._user_speaking = False

    def _log_debug(self, message: str):
        """Log debug message to file and logger."""
        if self._debug_log_file:
            timestamp = datetime.now().isoformat()
            log_line = f"[{timestamp}] {message}\n"
            self._debug_log_file.write(log_line)
            self._debug_log_file.flush()
        logger.debug(message)

    def __del__(self):
        """Clean up debug log file."""
        if self._debug_log_file:
            try:
                self._log_debug("=== Observer destroyed ===")
                self._debug_log_file.close()
            except:
                pass

    async def on_push_frame(self, data: FramePushed):
        """
        Called when a frame is pushed between processors.

        Args:
            data: FramePushed event data with source, destination, frame, direction
        """
        try:
            frame = data.frame
            frame_type = frame.__class__.__name__
            source_name = data.source.__class__.__name__ if data.source else "Unknown"

            # Log every frame
            self._log_debug(f"FRAME: {frame_type} from {source_name}")

            # Log frame details
            frame_details = {
                "type": frame_type,
                "source": source_name,
                "has_text": hasattr(frame, "text"),
            }
            if hasattr(frame, "text"):
                frame_details["text_preview"] = str(frame.text)[:50] if frame.text else None
            self._log_debug(f"  Details: {json.dumps(frame_details)}")

            # Service-based turn tracking: Use service frames to define turn boundaries
            # This avoids duplicate turn creation from frame propagation through pipeline
            source_name = data.source.__class__.__name__ if data.source else "Unknown"
            service_type = self._detector.detect_service_type(data.source)

            # Handle turn tracking using service-specific frames
            # Start turn: When STT produces transcription (user input received)
            if isinstance(frame, TranscriptionFrame) and service_type == "stt":
                # Check for interruption
                if self._bot_speaking and self._turn_active:
                    self._log_debug(f"  User interruption detected via TranscriptionFrame")
                    await self._finish_turn(interrupted=True)
                # Start new turn when user input arrives
                if not self._turn_active:
                    self._log_debug(f"  Starting turn via TranscriptionFrame from {source_name}")
                    self._turn_context_token = await self._start_turn()
                # Always collect user text
                if frame.text:
                    self._turn_user_text.append(frame.text)

            # Collect user input (from TranscriptionFrame without service check for backwards compat)
            elif isinstance(frame, TranscriptionFrame):
                if self._turn_active and frame.text:
                    self._turn_user_text.append(frame.text)

            # Handle bot-initiated conversations (greeting without user input)
            elif isinstance(frame, BotStartedSpeakingFrame):
                self._bot_speaking = True
                # Start turn if bot speaks first (no user input)
                if not self._turn_active:
                    self._log_debug(f"  Starting turn via BotStartedSpeakingFrame (bot-initiated)")
                    self._turn_context_token = await self._start_turn()

            # Collect bot output text from LLM streaming (LLMTextFrame) and TTS (TextFrame)
            elif isinstance(frame, (LLMTextFrame, TextFrame)):
                if self._turn_active and frame.text:
                    # LLMTextFrame arrives during streaming, TextFrame during TTS
                    self._turn_bot_text.append(frame.text)

            # End turn: When LLM finishes response (semantic completion)
            elif isinstance(frame, LLMFullResponseEndFrame) and service_type == "llm":
                self._log_debug(f"  Ending turn via LLMFullResponseEndFrame from {source_name}")
                self._bot_speaking = False
                await self._finish_turn(interrupted=False)

            # Fallback: End turn on BotStoppedSpeaking if no LLM (e.g., TTS-only responses)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                # Only end turn if we haven't already (LLMFullResponseEndFrame takes precedence)
                if self._turn_active and self._bot_speaking:
                    self._log_debug(f"  Ending turn via BotStoppedSpeakingFrame fallback")
                    self._bot_speaking = False
                    await self._finish_turn(interrupted=False)

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
            # If no turn is active yet, start one automatically
            # This ensures we capture initialization frames with proper context
            if self._turn_context_token is None:
                self._log_debug(
                    f"  No active turn - auto-starting turn for {service_type} initialization"
                )
                self._turn_context_token = await self._start_turn()

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
        self._log_debug(f">>> Creating {service_type} span")
        self._log_debug(f"  Context token type: {type(self._turn_context_token)}")
        self._log_debug(f"  Context token value: {self._turn_context_token}")

        span = self._tracer.start_span(
            name=f"pipecat.{service_type}",
            context=self._turn_context_token,
        )

        span_ctx = span.get_span_context()
        self._log_debug(
            f"  Created span - trace_id: {span_ctx.trace_id:032x}, span_id: {span_ctx.span_id:016x}"
        )
        if hasattr(span, "parent") and span.parent:
            self._log_debug(f"  Parent span_id: {span.parent.span_id:016x}")
        else:
            self._log_debug(f"  No parent span")
        # Extract metadata
        metadata = self._detector.extract_service_metadata(service)

        if service_type == "llm":
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.LLM.value,
            )
            span.set_attribute(SpanAttributes.LLM_MODEL_NAME, metadata.get("model", "unknown"))
            span.set_attribute(SpanAttributes.LLM_PROVIDER, metadata.get("provider", "unknown"))
        elif service_type == "tts" or service_type == "stt":
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.CHAIN.value,
            )
            span.set_attribute("audio.voice", metadata.get("voice", "unknown"))
            span.set_attribute("audio.voice_id", metadata.get("voice_id", "unknown"))
        else:
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.CHAIN.value,
            )

        # Set service.name to the actual service class name for uniqueness
        span.set_attribute("service.name", service.__class__.__name__)

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
        return

    async def _start_turn(self):
        """Start a new conversation turn and set it as parent context."""
        self._turn_number += 1

        self._log_debug(f"\n{'=' * 60}")
        self._log_debug(f">>> STARTING TURN #{self._turn_number}")
        self._log_debug(f"  Conversation ID: {self._conversation_id}")

        self._turn_span = self._tracer.start_span(
            name="pipecat.conversation.turn",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                "conversation.turn_number": self._turn_number,
            },
        )

        span_ctx = self._turn_span.get_span_context()
        self._log_debug(
            f"  Turn span created - trace_id: {span_ctx.trace_id:032x}, span_id: {span_ctx.span_id:016x}"
        )

        if self._conversation_id:
            self._turn_span.set_attribute(SpanAttributes.SESSION_ID, self._conversation_id)
            self._log_debug(f"  Set session.id attribute: {self._conversation_id}")

        self._turn_context_token = trace_api.set_span_in_context(self._turn_span)
        self._log_debug(f"  Context token created: {type(self._turn_context_token)}")

        self._turn_active = True
        self._turn_user_text = []
        self._turn_bot_text = []

        self._log_debug(f"{'=' * 60}\n")
        return self._turn_context_token

    async def _finish_turn(self, interrupted: bool = False):
        """
        Finish the current conversation turn and detach context.

        Args:
            interrupted: Whether the turn was interrupted
        """
        if not self._turn_active or not self._turn_span:
            self._log_debug("  Skipping finish_turn - no active turn")
            return

        self._log_debug(f"\n{'=' * 60}")
        self._log_debug(f">>> FINISHING TURN #{self._turn_number} (interrupted={interrupted})")
        self._log_debug(f"  Active service spans: {len(self._active_spans)}")

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

        service_ids_to_finish = list(self._active_spans.keys())
        for service_id in service_ids_to_finish:
            self._finish_span(service_id)

        # Clear turn context (no need to detach since we're not using attach)
        self._log_debug(f"  Clearing context token")
        self._turn_context_token = None

        self._log_debug(
            f"  Turn finished - input: {len(self._turn_user_text)} chunks, "
            f"output: {len(self._turn_bot_text)} chunks"
        )
        self._log_debug(f"{'=' * 60}\n")

        # Reset turn state
        self._turn_active = False
        self._turn_span = None
