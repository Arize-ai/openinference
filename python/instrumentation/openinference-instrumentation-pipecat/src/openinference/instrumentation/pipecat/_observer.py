"""OpenInference observer for Pipecat pipelines."""

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set
from contextvars import Token

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span
from opentelemetry.context import Context
from opentelemetry.context import attach as context_api_attach
from opentelemetry.context import detach as context_api_detach
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
    CancelFrame,
    EndFrame,
    Frame,
    LLMTextFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.services.image_service import ImageGenService
from pipecat.services.vision_service import VisionService
from pipecat.services.websocket_service import WebsocketService

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
        max_frames: int = 100,
        turn_end_timeout_secs: float = 2.5,
    ):
        """
        Initialize the observer.

        Args:
            tracer: OpenInference tracer
            config: Trace configuration
            conversation_id: Optional conversation/session ID to link all spans
            debug_log_filename: Optional filename for debug logging
            max_frames: Maximum number of frame IDs to keep in history for
                duplicate detection. Defaults to 100.
            turn_end_timeout_secs: Timeout in seconds after bot stops speaking
                before automatically ending the turn. Defaults to 2.5.
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

        # Track processed frames to avoid duplicates
        self._processed_frames: Set[int] = set()
        self._frame_history: Deque[int] = deque(maxlen=max_frames)

        # Track active spans per service instance
        # Key: id(service), Value: {"span": span, "frame_count": int}
        self._active_spans: Dict[int, Dict[str, Any]] = {}

        # Track the last frame seen from each service to detect completion
        self._last_frames: Dict[int, Frame] = {}

        # Turn tracking state (based on TurnTrackingObserver pattern)
        self._turn_active = False
        self._turn_span: Optional[Span] = None
        self._turn_context_token: Optional[Token[Context]] = None
        self._turn_number: int = 0
        self._turn_start_time: int = 0
        self._turn_user_text: List[str] = []
        self._turn_bot_text: List[str] = []
        self._bot_speaking: bool = False
        self._has_bot_spoken: bool = False
        self._turn_end_timeout_secs: float = turn_end_timeout_secs
        self._end_turn_timer: Optional[asyncio.TimerHandle] = None

    def _log_debug(self, message: str) -> None:
        """Log debug message to file and logger."""
        if self._debug_log_file:
            timestamp = datetime.now().isoformat()
            log_line = f"[{timestamp}] {message}\n"
            self._debug_log_file.write(log_line)
            self._debug_log_file.flush()
        logger.debug(message)

    def __del__(self) -> None:
        """Clean up debug log file."""
        if self._debug_log_file:
            try:
                self._log_debug("=== Observer destroyed ===")
                self._debug_log_file.close()
            except Exception as e:
                logger.error(f"Error closing debug log file: {e}")
                pass

    def _schedule_turn_end(self, data: FramePushed) -> None:
        """Schedule turn end with a timeout."""
        # Cancel any existing timer
        self._cancel_turn_end_timer()

        # Create a new timer
        loop = asyncio.get_event_loop()
        self._end_turn_timer = loop.call_later(
            self._turn_end_timeout_secs,
            lambda: asyncio.create_task(self._end_turn_after_timeout(data)),
        )
        self._log_debug(f"  Scheduled turn end timer ({self._turn_end_timeout_secs}s)")

    def _cancel_turn_end_timer(self) -> None:
        """Cancel the turn end timer if it exists."""
        if self._end_turn_timer:
            self._end_turn_timer.cancel()
            self._end_turn_timer = None
            self._log_debug("  Cancelled turn end timer")

    async def _end_turn_after_timeout(self, data: FramePushed) -> None:
        """End turn after timeout has expired."""
        if self._turn_active and not self._bot_speaking:
            self._log_debug(f"  Turn {self._turn_number} ending due to timeout")
            await self._finish_turn(interrupted=False)
            self._end_turn_timer = None

    async def on_push_frame(self, data: FramePushed) -> None:
        """
        Called when a frame is pushed between processors.

        Args:
            data: FramePushed event data with source, destination, frame, direction
        """
        try:
            frame = data.frame
            frame_type = frame.__class__.__name__
            source_name = data.source.__class__.__name__ if data.source else "Unknown"

            # Skip already processed frames to avoid duplicates from propagation
            if frame.id in self._processed_frames:
                self._log_debug(f"FRAME (DUPLICATE SKIPPED): {frame_type} from {source_name}")
                return

            # Mark frame as processed
            self._processed_frames.add(int(frame.id))
            self._frame_history.append(frame.id)

            # If we've exceeded our history size, rebuild the set from deque
            if len(self._processed_frames) > len(self._frame_history):
                self._processed_frames = set(self._frame_history)

            # Log every frame
            self._log_debug(f"FRAME: {frame_type} from {source_name}")

            # Turn tracking based on TurnTrackingObserver pattern
            # Use generic speaking frames for turn boundaries
            if isinstance(frame, StartFrame):
                # Start the first turn immediately when pipeline starts
                if self._turn_number == 0:
                    self._log_debug("  Starting first turn via StartFrame")
                    await self._start_turn(data)

            elif isinstance(frame, UserStartedSpeakingFrame):
                await self._handle_user_started_speaking(data)

            elif isinstance(frame, BotStartedSpeakingFrame):
                await self._handle_bot_started_speaking(data)

            elif isinstance(frame, BotStoppedSpeakingFrame) and self._bot_speaking:
                await self._handle_bot_stopped_speaking(data)

            elif isinstance(frame, (EndFrame, CancelFrame)):
                await self._handle_pipeline_end(data)

            # Collect conversation text (separate concern from turn boundaries)
            if isinstance(frame, TranscriptionFrame):
                # Collect user text
                if self._turn_active and frame.text:
                    self._turn_user_text.append(frame.text)
                    self._log_debug(f"  Collected user text: {frame.text[:50]}...")

            elif isinstance(frame, (LLMTextFrame, TextFrame)):
                # Collect bot text
                if self._turn_active and frame.text:
                    self._turn_bot_text.append(frame.text)

            # Handle service frames for creating service spans
            service_type = self._detector.detect_service_type(data.source)
            if service_type:
                await self._handle_service_frame(data, service_type)

        except Exception as e:
            logger.debug(f"Error in observer: {e}")

    async def _handle_user_started_speaking(self, data: FramePushed) -> None:
        """Handle user speaking events, including interruptions."""
        if self._bot_speaking:
            # Handle interruption - end current turn and start a new one
            self._log_debug("  User interruption detected - ending current turn")
            self._cancel_turn_end_timer()
            await self._finish_turn(interrupted=True)
            self._bot_speaking = False  # Bot is considered interrupted
            self._log_debug("  Starting new turn after interruption")
            await self._start_turn(data)
        elif self._turn_active and self._has_bot_spoken:
            # User started speaking during the turn_end_timeout_secs period after bot speech
            self._log_debug("  User speaking after bot - ending turn and starting new one")
            self._cancel_turn_end_timer()
            await self._finish_turn(interrupted=False)
            await self._start_turn(data)
        elif not self._turn_active:
            # Start a new turn after previous one ended
            self._log_debug("  Starting new turn (user speaking)")
            await self._start_turn(data)
        else:
            # User is speaking within the same turn (before bot has responded)
            self._log_debug(f"  User is already speaking in Turn {self._turn_number}")

    async def _handle_bot_started_speaking(self, data: FramePushed) -> None:
        """Handle bot speaking events."""
        self._bot_speaking = True
        self._has_bot_spoken = True
        # Cancel any pending turn end timer when bot starts speaking again
        self._cancel_turn_end_timer()
        self._log_debug("  Bot started speaking")

    async def _handle_bot_stopped_speaking(self, data: FramePushed) -> None:
        """Handle bot stopped speaking events."""
        self._bot_speaking = False
        self._log_debug("  Bot stopped speaking")
        # Schedule turn end with timeout
        # This is needed to handle cases where the bot's speech ends and then resumes
        # This can happen with HTTP TTS services or function calls
        self._schedule_turn_end(data)

    async def _handle_pipeline_end(self, data: FramePushed) -> None:
        """Handle pipeline end or cancellation by flushing any active turn."""
        if self._turn_active:
            self._log_debug("  Pipeline ending - finishing active turn")
            # Cancel any pending turn end timer
            self._cancel_turn_end_timer()
            # End the current turn
            await self._finish_turn(interrupted=True)

    async def _handle_service_frame(self, data: FramePushed, service_type: str) -> None:
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
                self._turn_context_token = await self._start_turn(data)

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

    def _create_service_span(self, service: FrameProcessor, service_type: str) -> Span:
        """
        Create a span for a service with type-specific attributes.

        Args:
            service: The service instance (FrameProcessor)
            service_type: Service type (llm, tts, stt, image_gen, vision, mcp, websocket)

        Returns:
            The created span
        """
        self._log_debug(f">>> Creating {service_type} span")
        self._log_debug(f"  Context token type: {type(self._turn_context_token)}")
        self._log_debug(f"  Context token value: {self._turn_context_token}")

        span = self._tracer.start_span(
            name=f"pipecat.{service_type}",
        )

        span_ctx = span.get_span_context()
        self._log_debug(
            f"  Created span - trace_id: {span_ctx.trace_id:032x}, span_id: {span_ctx.span_id:016x}"
        )
        if hasattr(span, "parent") and span.parent:
            self._log_debug(f"  Parent span_id: {span.parent.span_id:016x}")
        else:
            self._log_debug("  No parent span")

        # Extract metadata from service
        metadata = self._detector.extract_service_metadata(service)

        # Set service.name to the actual service class name for uniqueness
        span.set_attribute("service.name", service.__class__.__name__)

        # Set common attributes if available
        if metadata.get("provider"):
            span.set_attribute("service.provider", metadata["provider"])
        if metadata.get("model"):
            span.set_attribute("service.model", metadata["model"])

        # Set type-specific attributes based on service type
        if service_type == "llm" and isinstance(service, LLMService):
            self._set_llm_attributes(span, service, metadata)
        elif service_type == "stt" and isinstance(service, STTService):
            self._set_stt_attributes(span, service, metadata)
        elif service_type == "tts" and isinstance(service, TTSService):
            self._set_tts_attributes(span, service, metadata)
        elif service_type == "image_gen" and isinstance(service, ImageGenService):
            self._set_image_gen_attributes(span, service, metadata)
        elif service_type == "vision" and isinstance(service, VisionService):
            self._set_vision_attributes(span, service, metadata)
        elif service_type == "mcp" and isinstance(service, FrameProcessor):
            self._set_mcp_attributes(span, service, metadata)
        elif service_type == "websocket" and isinstance(service, WebsocketService):
            self._set_websocket_attributes(span, service, metadata)
        else:
            # Default for unknown service types
            span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND,
                OpenInferenceSpanKindValues.CHAIN.value,
            )

        return span

    def _set_llm_attributes(
        self, span: Span, service: LLMService, metadata: Dict[str, Any]
    ) -> None:
        """Set LLM-specific span attributes."""
        span.set_attribute(  #
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.LLM.value,
        )
        span.set_attribute(  #
            SpanAttributes.LLM_MODEL_NAME, metadata.get("model", "unknown")
        )
        span.set_attribute(  #
            SpanAttributes.LLM_PROVIDER, metadata.get("provider", "unknown")
        )

        # Additional LLM attributes from settings if available
        if hasattr(service, "_settings"):
            settings = service._settings
            if "temperature" in settings:
                span.set_attribute("llm.temperature", settings["temperature"])
            if "max_tokens" in settings:
                span.set_attribute("llm.max_tokens", settings["max_tokens"])
            if "top_p" in settings:
                span.set_attribute("llm.top_p", settings["top_p"])

    def _set_stt_attributes(
        self, span: Span, service: STTService, metadata: Dict[str, Any]
    ) -> None:
        """Set STT-specific span attributes."""
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.CHAIN.value,
        )

        # Audio attributes
        if metadata.get("sample_rate"):
            span.set_attribute("audio.sample_rate", metadata["sample_rate"])
        if metadata.get("is_muted") is not None:
            span.set_attribute("audio.is_muted", metadata["is_muted"])
        if metadata.get("user_id"):
            span.set_attribute("audio.user_id", metadata["user_id"])

    def _set_tts_attributes(
        self, span: Span, service: TTSService, metadata: Dict[str, Any]
    ) -> None:
        """Set TTS-specific span attributes."""
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.CHAIN.value,
        )

        # Audio and voice attributes
        if metadata.get("voice_id"):
            span.set_attribute("audio.voice_id", metadata["voice_id"])
            span.set_attribute(
                "audio.voice", metadata["voice_id"]
            )  # Also set as audio.voice for compatibility
        if metadata.get("sample_rate"):
            span.set_attribute("audio.sample_rate", metadata["sample_rate"])
        if service._text_aggregator and hasattr(service._text_aggregator, "text"):
            span.set_attribute(SpanAttributes.INPUT_VALUE, service._text_aggregator.text)

    def _set_image_gen_attributes(
        self, span: Span, service: ImageGenService, metadata: Dict[str, Any]
    ) -> None:
        """Set image generation-specific span attributes."""
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.CHAIN.value,
        )
        span.set_attribute("service.type", "image_generation")

    def _set_vision_attributes(
        self, span: Span, service: VisionService, metadata: Dict[str, Any]
    ) -> None:
        """Set vision-specific span attributes."""
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.CHAIN.value,
        )
        span.set_attribute("service.type", "vision")

    def _set_mcp_attributes(
        self, span: Span, service: FrameProcessor, metadata: Dict[str, Any]
    ) -> None:
        """Set MCP (Model Context Protocol) client-specific span attributes."""

        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.CHAIN.value,
        )
        span.set_attribute("service.type", "mcp_client")

        try:
            from pipecat.services.mcp_service import MCPClient

            if isinstance(service, MCPClient):
                # MCP-specific attributes
                if hasattr(service, "_server_params"):
                    server_params = service._server_params
                    span.set_attribute("mcp.server_type", type(server_params).__name__)
        except Exception as e:
            logger.error(f"Error setting MCP attributes: {e}")
            pass

    def _set_websocket_attributes(
        self, span: Span, service: WebsocketService, metadata: Dict[str, Any]
    ) -> None:
        """Set websocket service-specific span attributes."""
        span.set_attribute(  #
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.CHAIN.value,
        )
        span.set_attribute("service.type", "websocket")  #

        # Websocket-specific attributes
        if hasattr(service, "_reconnect_on_error"):
            span.set_attribute(  #
                "websocket.reconnect_on_error", service._reconnect_on_error
            )

    def _finish_span(self, service_id: int) -> None:
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
        span.set_status(trace_api.Status(trace_api.StatusCode.OK))  #
        span.end()
        return

    async def _start_turn(self, data: FramePushed) -> Token[Context]:
        """Start a new conversation turn and set it as parent context."""
        self._turn_active = True
        self._has_bot_spoken = False
        self._turn_number += 1
        self._turn_start_time = data.timestamp

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
            f"Turn span created - trace_id: {span_ctx.trace_id:032x},"
            f"span_id: {span_ctx.span_id:016x}"
        )

        if self._conversation_id:
            self._turn_span.set_attribute(  #
                SpanAttributes.SESSION_ID, self._conversation_id
            )
            self._log_debug(f"  Set session.id attribute: {self._conversation_id}")

        context = trace_api.set_span_in_context(self._turn_span)
        self._turn_context_token = context_api_attach(context)  #
        self._log_debug(f"  Context token created: {type(self._turn_context_token)}")

        self._turn_user_text = []
        self._turn_bot_text = []

        self._log_debug(f"{'=' * 60}\n")
        return self._turn_context_token

    async def _finish_turn(self, interrupted: bool = False) -> None:
        """
        Finish the current conversation turn and detach context.

        Args:
            interrupted: Whether the turn was interrupted
        """
        if not self._turn_active or not self._turn_span:
            self._log_debug("  Skipping finish_turn - no active turn")
            return

        # Calculate turn duration
        duration = 0.0
        if self._turn_start_time > 0:
            import time

            current_time = time.time_ns()
            duration = (current_time - self._turn_start_time) / 1_000_000_000  # Convert to seconds

        self._log_debug(f"\n{'=' * 60}")
        self._log_debug(
            f">>> FINISHING TURN #{self._turn_number} (interrupted={interrupted}, duration={duration:.2f}s)"
        )
        self._log_debug(f"  Active service spans: {len(self._active_spans)}")

        # Set input/output attributes
        if self._turn_user_text:
            user_input = " ".join(self._turn_user_text)
            self._turn_span.set_attribute(SpanAttributes.INPUT_VALUE, user_input)  #

        if self._turn_bot_text:
            bot_output = " ".join(self._turn_bot_text)
            self._turn_span.set_attribute(SpanAttributes.OUTPUT_VALUE, bot_output)  #

        # Set turn metadata
        end_reason = "interrupted" if interrupted else "completed"
        self._turn_span.set_attribute("conversation.end_reason", end_reason)  #
        self._turn_span.set_attribute("conversation.turn_duration_seconds", duration)
        self._turn_span.set_attribute("conversation.was_interrupted", interrupted)  #

        # Finish span
        self._turn_span.set_status(trace_api.Status(trace_api.StatusCode.OK))  #
        self._turn_span.end()  #

        service_ids_to_finish = list(self._active_spans.keys())
        for service_id in service_ids_to_finish:
            self._finish_span(service_id)

        # Clear turn context (no need to detach since we're not using attach)
        self._log_debug("  Clearing context token")
        if self._turn_context_token:
            context_api_detach(self._turn_context_token)
            self._turn_context_token = None
        self._log_debug(
            f"  Turn finished - input: {len(self._turn_user_text)} chunks, "
            f"output: {len(self._turn_bot_text)} chunks"
        )
        self._log_debug(f"{'=' * 60}\n")

        # Reset turn state
        self._turn_active = False
        self._turn_span = None
