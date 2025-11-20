"""OpenInference observer for Pipecat pipelines."""

import asyncio
import logging
import time
from collections import deque
from contextvars import Token
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set

from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.trace import Span

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.pipecat._attributes import (
    detect_service_type,
    extract_attributes_from_frame,
    extract_service_attributes,
)
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
    LLMContextFrame,
    StartFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_output import BaseOutputTransport

# Suppress OpenTelemetry context detach errors - these are expected in async code
# where contexts may be created and detached in different async contexts
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

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
        verbose: bool = False,
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

        # Session management
        self._conversation_id = conversation_id

        # Debug logging to file
        self._debug_log_file = None
        self._verbose = verbose
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
        if self._verbose:
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
        loop = asyncio.get_running_loop()
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
            # Only collect from final/complete frames to avoid duplication
            if isinstance(frame, TranscriptionFrame):
                # Collect user text from STT output
                if self._turn_active and frame.text:
                    self._turn_user_text.append(frame.text)
                    self._log_debug(f"  Collected user text: {frame.text[:50]}...")

            elif isinstance(frame, TTSTextFrame):
                # Collect bot text from TTS input (final complete sentences)
                # Only collect if the frame comes from an actual TTS service, not transport
                # This prevents duplication when frames propagate through the pipeline
                service_type = detect_service_type(data.source)
                if self._turn_active and frame.text and service_type == "tts":
                    self._turn_bot_text.append(frame.text)
                    self._log_debug(f"  Collected bot text from TTS: {frame.text[:50]}...")

            # Handle service frames for creating service spans
            # Check both source (frames emitted BY service)
            # and destination (frames received BY service)
            source_service_type = detect_service_type(data.source)
            dest_service_type = detect_service_type(data.destination)

            # Handle frames emitted by a service (outputs)
            if source_service_type:
                await self._handle_service_frame(data, is_input=False)

            # Handle frames received by a service (inputs)
            # Only process if destination is different from source to avoid double-counting
            if dest_service_type and data.destination != data.source:
                await self._handle_service_frame(data, is_input=True)

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

    async def _handle_service_frame(self, data: FramePushed, is_input: bool = False) -> None:
        """
        Handle frame from an LLM, TTS, or STT service.
        Detects nested LLM calls within TTS/STT services.

        Args:
            data: FramePushed event data
            is_input: True if this frame is being received by the service (input),
                     False if being emitted by the service (output)
        """
        from pipecat.frames.frames import (
            EndFrame,
            ErrorFrame,
        )

        # Use destination for input frames, source for output frames
        service = data.destination if is_input else data.source
        service_id = id(service)
        frame = data.frame
        service_type = detect_service_type(service)

        if service_type != "unknown":
            # Check if we need to create a new span
            # For LLM services, LLMContextFrame signals a new invocation
            # finish previous span if exists
            if isinstance(frame, LLMContextFrame) and service_id in self._active_spans:
                self._log_debug(
                    f"  New LLM invocation detected"
                    f"  Finishing previous span for service {service_id}"
                )
                self._finish_span(service_id)

            # Check if we already have a span for this service
            if service_id not in self._active_spans:
                # If no turn is active yet, start one automatically
                # This ensures we capture initialization frames with proper context
                if not self._turn_active or self._turn_span is None:
                    self._log_debug(
                        f"  No active turn - auto-starting turn for {service_id} initialization"
                    )
                    await self._start_turn(data)

                # Create new span directly under turn (no nesting logic)
                # All service spans are siblings under the turn span
                span = self._create_service_span(service, service_type)
                self._active_spans[service_id] = {
                    "span": span,
                    "service_type": service_type,  # Track service type for later use
                    "frame_count": 0,
                    "accumulated_input": "",  # Deduplicated accumulated input text
                    "accumulated_output": "",  # Deduplicated accumulated output text
                    "start_time_ns": time.time_ns(),  # Store start time in nanoseconds (Unix epoch)
                    "processing_time_seconds": None,  # Will be set from metrics
                }

            # Check if span still exists (it might have been ended by a previous call)
            if service_id not in self._active_spans:
                self._log_debug(f"  Span for service {service_id} already ended, skipping frame")
                return

            # Increment frame count for this service
            span_info = self._active_spans[service_id]
            span_info["frame_count"] += 1

            # Extract and add attributes from this frame to the span
            span = span_info["span"]
            frame_attrs = extract_attributes_from_frame(frame)

            # Log frame direction for debugging
            direction = "INPUT" if is_input else "OUTPUT"
            self._log_debug(
                f"  Processing {direction} frame: {frame.__class__.__name__} for {service_type}"
            )
            if frame_attrs:
                self._log_debug(
                    f"    Extracted {len(frame_attrs)} attributes: {list(frame_attrs.keys())}"
                )
            else:
                self._log_debug("    No attributes extracted from this frame")

            # Handle text chunk accumulation with deduplication
            # IMPORTANT: Only collect INPUT chunks when frame is received by service (is_input=True)
            # and only collect OUTPUT chunks when frame is emitted by service (is_input=False)

            # Check for streaming text chunks
            text_chunk: str = frame_attrs.get("text.chunk", "")
            accumulated: str = ""
            if text_chunk:
                # For TTS input frames, only accumulate if going to output transport
                # This ensures we only capture complete sentences being sent to the user
                if is_input and service_type == "tts":
                    # Check if destination is the final output transport
                    if not isinstance(data.destination, BaseOutputTransport):
                        self._log_debug("    Skipping TTS chunk (not going to output transport)")
                        text_chunk = ""  # Skip this chunk

                if text_chunk and is_input:
                    # Input chunk - check if this extends our accumulated text
                    accumulated = span_info["accumulated_input"]
                    if not accumulated:
                        # First chunk
                        span_info["accumulated_input"] = text_chunk
                        self._log_debug(f"    Accumulated INPUT chunk (first): {text_chunk}...")
                    elif text_chunk.startswith(accumulated):
                        # New chunk contains all previous text plus more (redundant pattern)
                        # Extract only the new part
                        new_part = text_chunk[len(accumulated) :]
                        if new_part:
                            span_info["accumulated_input"] = text_chunk
                            self._log_debug(f"    Accumulated INPUT (new part): {new_part}...")
                        else:
                            self._log_debug("    Skipped fully redundant INPUT chunk")
                    elif accumulated and accumulated in text_chunk:
                        # Current accumulated text is contained in new chunk
                        # This means we're getting the full text again with more added
                        span_info["accumulated_input"] = text_chunk if text_chunk else ""
                        new_part = text_chunk.replace(accumulated, "", 1) if text_chunk else ""
                        self._log_debug(f"    Accumulated INPUT (replaced): {new_part}...")
                    else:
                        # Non-overlapping chunk - just append
                        span_info["accumulated_input"] = accumulated + text_chunk
                        self._log_debug(f"    Accumulated INPUT chunk (append): {text_chunk}...")
                else:
                    # Output chunk - same logic
                    accumulated = span_info["accumulated_output"]
                    if not accumulated:
                        span_info["accumulated_output"] = text_chunk
                        self._log_debug(f"    Accumulated OUTPUT chunk (first): {text_chunk}...")
                    elif text_chunk.startswith(accumulated):
                        new_part = text_chunk[len(accumulated) :]
                        if new_part:
                            span_info["accumulated_output"] = text_chunk
                            self._log_debug(f"    Accumulated OUTPUT (new part): {new_part}...")
                        else:
                            self._log_debug("    Skipped fully redundant OUTPUT chunk")
                    elif accumulated in text_chunk:
                        span_info["accumulated_output"] = text_chunk
                        new_part = text_chunk.replace(accumulated, "", 1)
                        self._log_debug(f"    Accumulated OUTPUT (replaced): {new_part}...")
                    elif accumulated and text_chunk:
                        span_info["accumulated_output"] = accumulated + text_chunk
                        self._log_debug(f"    Accumulated OUTPUT chunk (append): {text_chunk}...")
                    else:
                        self._log_debug("    Skipped OUTPUT chunk (no accumulated text)")

            # Process all other attributes
            for key, value in frame_attrs.items():
                # Skip text.chunk since we handled it above
                if key == "text.chunk":
                    continue

                # Skip input-related attributes if this is an output frame
                if not is_input and (
                    key in (SpanAttributes.INPUT_VALUE, SpanAttributes.LLM_INPUT_MESSAGES)
                    or key.startswith("llm.input_messages.")
                ):
                    self._log_debug(
                        f"    Skipping INPUT attribute {key} (frame is OUTPUT from service)"
                    )
                    continue

                # Skip output-related attributes if this is an input frame
                if is_input and (
                    key in (SpanAttributes.OUTPUT_VALUE, SpanAttributes.LLM_OUTPUT_MESSAGES)
                    or key.startswith("llm.output_messages.")
                ):
                    self._log_debug(
                        f"    Skipping OUTPUT attribute {key} (frame is INPUT to service)"
                    )
                    continue

                # Handle complete (non-streaming) INPUT_VALUE (e.g., from TranscriptionFrame)
                # Special case for STT: TranscriptionFrame is OUTPUT from STT but represents the
                # transcribed text which should be recorded as INPUT to the span for observability
                if key == SpanAttributes.INPUT_VALUE and value:
                    if is_input or service_type == "stt":
                        # This is a complete input, not streaming - set immediately
                        # For STT, we capture output transcriptions as input values
                        span.set_attribute(SpanAttributes.INPUT_VALUE, value)
                        self._log_debug(f"    Set complete INPUT_VALUE: {str(value)[:100]}...")

                # Handle complete (non-streaming) OUTPUT_VALUE
                elif key == SpanAttributes.OUTPUT_VALUE and value and not is_input:
                    # This is a complete output, not streaming - set immediately
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, value)
                    self._log_debug(f"    Set complete OUTPUT_VALUE: {str(value)}...")

                elif key == "service.processing_time_seconds":
                    # Store processing time for use in _finish_span to calculate proper end_time
                    span_info["processing_time_seconds"] = value
                    span.set_attribute("service.processing_time_seconds", value)
                else:
                    # For all other attributes, just set them (may overwrite)
                    span.set_attribute(key, value)

            # Store this as the last frame from this service
            self._last_frames[service_id] = frame

        # Finish span only on completion frames (EndFrame or ErrorFrame)
        if isinstance(frame, (EndFrame, ErrorFrame)):
            self._finish_span(service_id)

    def _create_service_span(
        self,
        service: FrameProcessor,
        service_type: str,
    ) -> Span:
        """
        Create a span for a service with type-specific attributes.
        All service spans are created as children of the turn span.

        Args:
            service: The service instance (FrameProcessor)
            service_type: Service type (llm, tts, stt, image_gen, vision, mcp, websocket)

        Returns:
            The created span
        """
        span_name = f"pipecat.{service_type}"
        self._log_debug(f">>> Creating {service_type} span")

        # Create span under the turn context
        # Explicitly set the turn span as parent to avoid context issues in async code
        if self._turn_span and self._turn_active:
            turn_context = trace_api.set_span_in_context(self._turn_span)
            span = self._tracer.start_span(
                name=span_name,
                context=turn_context,
            )
            self._log_debug(f"  Created service span under turn #{self._turn_number}")
        else:
            # No active turn, create as root span (will be in new trace)
            self._log_debug(f"  WARNING: No active turn! Creating root span for {service_type}")
            span = self._tracer.start_span(
                name=span_name,
            )

        # Set service.name to the actual service class name for uniqueness
        span.set_attribute("service.name", service.__class__.__name__)

        # Extract and apply service-specific attributes
        service_attrs = extract_service_attributes(service)
        span.set_attributes(service_attrs)
        self._log_debug(f"  Set attributes: {service_attrs}")

        return span

    def _finish_span(self, service_id: int) -> None:
        """
        Finish a span for a service.

        Args:
            service_id: The id() of the service instance
        """
        if service_id not in self._active_spans:
            return

        span_info = self._active_spans.pop(service_id)
        span: Span = span_info["span"]
        start_time_ns = span_info["start_time_ns"]

        # Calculate end time (use processing time if available, otherwise use current time)
        processing_time_seconds = span_info.get("processing_time_seconds")
        if processing_time_seconds is not None:
            end_time_ns = start_time_ns + int(processing_time_seconds * 1_000_000_000)
        else:
            end_time_ns = time.time_ns()

        # Set accumulated input/output text values from streaming chunks
        # These were deduplicated during accumulation
        accumulated_input = span_info.get("accumulated_input", "")
        accumulated_output = span_info.get("accumulated_output", "")

        if accumulated_input:
            span.set_attribute(SpanAttributes.INPUT_VALUE, accumulated_input)
            self._log_debug(
                f"  Set input.value from accumulated chunks: {len(accumulated_input)} chars"
            )

        if accumulated_output:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, accumulated_output)
            self._log_debug(
                f"  Set output.value from accumulated chunks: {len(accumulated_output)} chars"
            )

            # For LLM spans, also set flattened output messages format
            service_type = span_info.get("service_type")
            if service_type == "llm":
                span.set_attribute("llm.output_messages.0.message.role", "assistant")
                span.set_attribute("llm.output_messages.0.message.content", accumulated_output)

        span.set_status(trace_api.Status(trace_api.StatusCode.OK))  #
        span.end(end_time=int(end_time_ns))
        return

    async def _start_turn(self, data: FramePushed) -> None:
        """Start a new conversation turn and set it as parent context."""
        self._turn_active = True
        self._has_bot_spoken = False
        self._turn_number += 1
        self._turn_start_time = time.time_ns()  # Use our own clock for consistency

        self._log_debug(f"\n{'=' * 60}")
        self._log_debug(f">>> STARTING TURN #{self._turn_number}")
        self._log_debug(f"  Conversation ID: {self._conversation_id}")

        # Create turn span as root (no parent)
        # Each turn will be a separate trace automatically
        # Use an empty context to ensure no ambient parent span is picked up
        self._turn_span = self._tracer.start_span(
            name="pipecat.conversation.turn",
            context=Context(),  # Empty context ensures this is a true root span
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                "conversation.turn_number": self._turn_number,
            },
        )

        if self._conversation_id:
            self._turn_span.set_attribute(  #
                SpanAttributes.SESSION_ID, self._conversation_id
            )
            self._log_debug(f"  Set session.id attribute: {self._conversation_id}")

        self._turn_user_text = []
        self._turn_bot_text = []
        return

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
        current_time_ns = time.time_ns()
        duration = (current_time_ns - self._turn_start_time) / 1_000_000_000  # Convert to seconds

        self._log_debug(f"\n{'=' * 60}")
        self._log_debug(
            f">>> FINISHING TURN #{self._turn_number}"
            + f" (interrupted={interrupted}, duration={duration:.2f}s)"
        )
        self._log_debug(f"  Active service spans: {len(self._active_spans)}")

        # Set input/output attributes
        if self._turn_user_text:
            user_input = " ".join(self._turn_user_text)
            self._turn_span.set_attribute(SpanAttributes.INPUT_VALUE, user_input)  #

        if self._turn_bot_text:
            bot_output = " ".join(self._turn_bot_text)
            self._turn_span.set_attribute(SpanAttributes.OUTPUT_VALUE, bot_output)  #

        # Finish all active service spans BEFORE ending the turn span
        # This ensures child spans are ended before the parent
        service_ids_to_finish = list(self._active_spans.keys())
        for service_id in service_ids_to_finish:
            self._finish_span(service_id)

        # Set turn metadata
        end_reason = "interrupted" if interrupted else "completed"
        self._turn_span.set_attribute("conversation.end_reason", end_reason)  #
        self._turn_span.set_attribute("conversation.turn_duration_seconds", duration)
        self._turn_span.set_attribute("conversation.was_interrupted", interrupted)

        # Finish turn span (parent) last
        self._turn_span.set_status(trace_api.Status(trace_api.StatusCode.OK))  #
        self._turn_span.end(end_time=int(current_time_ns))  #

        # Clear turn state
        self._log_debug("  Clearing turn state")
        self._turn_active = False
        self._turn_span = None
        self._turn_context_token = None
