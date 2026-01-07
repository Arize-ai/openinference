"""OpenInference observer for Pipecat pipelines."""

import logging
import time
from contextvars import Token
from datetime import datetime
from typing import Any, Dict, List, Optional, TextIO

from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.trace import Span

from openinference.instrumentation import OITracer, TraceConfig
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.loggers.user_bot_latency_log_observer import (
    UserBotLatencyLogObserver,
)
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.ai_service import AIService
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_output import BaseOutputTransport

from ._attributes import (  # noqa: F401
    detect_service_type,
    detect_service_type_from_class_string,
    extract_attributes_from_frame,
    extract_service_attributes,
)

# Suppress OpenTelemetry context detach errors - these are expected in async code
# where contexts may be created and detached in different async contexts
logging.getLogger("opentelemetry.context").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


class OpenInferenceObserver(TurnTrackingObserver):
    """
    Observer that creates OpenInference spans for Pipecat frame processing.

    Observes frame flow through pipeline and creates spans for LLM, TTS, and STT services.
    Implements proper span hierarchy with session ID propagation.
    """

    def __init__(
        self,
        tracer: OITracer,
        config: TraceConfig,
        additional_span_attributes: Optional[Dict[str, Any]] = None,
        conversation_id: Optional[str] = None,
        debug_log_filename: Optional[str] = None,
        max_frames: int = 100,
        turn_end_timeout_secs: float = 2.5,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the observer.

        Args:
            tracer: OpenInference tracer
            config: Trace configuration
            additional_span_attributes: Optional additional span attributes to add to all spans
            conversation_id: Optional conversation/session ID to link all spans
            debug_log_filename: Optional filename for debug logging
            max_frames: Maximum number of frame IDs to keep in history for duplicate detection
            turn_end_timeout_secs: Timeout in seconds after bot stops speaking before automatically
                ending the turn
            verbose: Optional verbose logging
            kwargs: Additional keyword arguments to pass to the base class
        """
        super().__init__(  # type: ignore[no-untyped-call]
            max_frames=max_frames,
            turn_end_timeout_secs=turn_end_timeout_secs,
            **kwargs,
        )
        self._latency_observer: UserBotLatencyLogObserver = UserBotLatencyLogObserver()  # type: ignore
        self._tracer: OITracer = tracer
        self._config: TraceConfig = config
        self._additional_span_attributes: Dict[str, str] = {}
        if additional_span_attributes and isinstance(additional_span_attributes, dict):
            for k, v in additional_span_attributes.items():
                self._additional_span_attributes[str(k)] = str(v)
        # Session management
        self._conversation_id: Optional[str] = conversation_id

        # Debug logging to file
        self._debug_log_file: Optional[TextIO] = None
        self._verbose: bool = verbose
        if debug_log_filename:
            # Write log to current working directory (where the script is running)
            try:
                self._debug_log_file = open(debug_log_filename, "w")
                self._log_debug(f"=== Observer initialized for conversation {conversation_id} ===")
                self._log_debug(f"=== Log file: {debug_log_filename} ===")
            except Exception as e:
                logger.error(f"Could not open debug log file: {e}")

        self._active_spans: Dict[int, Dict[str, Any]] = {}

        # Track the last frame seen from each service to detect completion
        self._last_frames: Dict[int, Frame] = {}

        # Turn tracking state
        self._turn_span: Optional[Span] = None
        self._turn_context_token: Optional[Token[Context]] = None
        self._turn_user_text: List[str] = []
        self._turn_bot_text: List[str] = []

        self._stt_includes_inter_frame_spaces: bool = False
        self._llm_includes_inter_frame_spaces: bool = False
        self._tts_includes_inter_frame_spaces: bool = False
        self._seen_vad_user_stopped_speaking_frame: bool = False

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

    async def on_push_frame(self, data: FramePushed) -> None:
        """
        Called when a frame is pushed between processors.

        Args:
            data: FramePushed event data with source, destination, frame, direction
        """
        await super().on_push_frame(data)
        # ensure UserBotLatencyLogObserver is using self._user_bot_latency_processed_frames !
        await self._latency_observer.on_push_frame(data)

        try:
            src = data.source
            dst = data.destination
            frame = data.frame
            frame_type = frame.__class__.__name__
            source_name = data.source.__class__.__name__ if data.source else "Unknown"

            if self._verbose:
                # Log every frame if verbose
                self._log_debug(f"FRAME: {frame_type} from {source_name}")

            if isinstance(frame, StartFrame):
                if self._turn_count == 0:
                    self._log_debug("  Starting first turn via StartFrame")

            # _end_turn is called by super(); no need to handle
            # elif isinstance(frame, (EndFrame, CancelFrame)):
            #     await self._handle_service_frame(data)

            # STT
            elif isinstance(src, STTService):
                if isinstance(
                    frame,
                    (
                        TranscriptionFrame,
                        VADUserStartedSpeakingFrame,
                        VADUserStoppedSpeakingFrame,
                        MetricsFrame,
                    ),
                ):
                    await self._handle_service_frame(data)

            # LLM (source)
            elif isinstance(src, LLMService):
                if isinstance(
                    frame,
                    (
                        LLMFullResponseStartFrame,
                        LLMFullResponseEndFrame,
                        LLMTextFrame,
                        MetricsFrame,
                    ),
                ):
                    await self._handle_service_frame(data)

            # LLM (destination)
            elif isinstance(dst, LLMService):
                if isinstance(frame, LLMContextFrame):
                    await self._handle_service_frame(data, override_service=dst)

            # TTS
            elif isinstance(src, TTSService):
                if isinstance(frame, (TTSTextFrame, TTSStartedFrame, MetricsFrame)):
                    await self._handle_service_frame(data)

            elif isinstance(src, BaseOutputTransport):
                if isinstance(frame, (TTSTextFrame)):
                    await self._handle_service_frame(data)

        except Exception as e:
            logger.debug(f"Error in observer on_push_frame: {e}")

    async def _handle_service_frame(
        self, data: FramePushed, override_service: Optional[AIService] = None
    ) -> None:
        """
        Handle frame from an LLM, TTS, or STT service.

        Args:
            data: FramePushed event data
            override_service: Don't use src service (default); explicitly use the passed in service
        """
        service = override_service if override_service else data.source
        service_type = detect_service_type(service)
        service_id = id(service)
        frame = data.frame

        self._log_debug(f"FRAME: {frame}, service_type: {service_type}")

        if service_type in ("llm", "stt", "tts"):
            # only these frame types will start a span:
            ## VADUserStartedSpeakingFrame (STT)
            ## LLMFullResponseStartFrame (LLM)
            ## TTSStartedFrame (TTS)

            # New Span (or add to self._turn_bot_text)
            if service_id not in self._active_spans:
                if isinstance(
                    frame,
                    (VADUserStartedSpeakingFrame, LLMContextFrame, TTSStartedFrame),
                ):
                    self._log_debug(f"  {service_type.upper()} response STARTED. ({frame})")
                    self._log_debug(f"   >>>>  {service_type.upper()} response STARTED. ({frame})")

                    # If no turn is active yet, start one automatically
                    # This ensures we capture initialization frames with proper context
                    if not self._is_turn_active or self._turn_span is None:
                        self._log_debug(
                            f"  No active turn - auto-starting turn for {service_id} initialization"
                        )
                        await self._start_turn(data)

                    # Create new span directly under turn (no nesting logic)
                    # All service spans are siblings under the turn span
                    self._log_debug(f"  CREATING new SPAN for {service_type}: {service_id}")
                    span = self._create_service_span(service, service_type)
                    self._active_spans[service_id] = {
                        "span": span,
                        "service_type": service_type,  # Track service type for later use
                        "frame_count": 0,
                        "accumulated_input": "",  # Deduplicated accumulated input text
                        "accumulated_output": "",  # Deduplicated accumulated output text
                        "start_time_ns": time.time_ns(),  # Store start time in nanoseconds
                        "processing_time_seconds": None,  # Will be set from metrics
                    }
                    # Increment frame count for this service
                    span_info = self._active_spans[service_id]
                    span_info["frame_count"] += 1

                    # Set input context for LLM Span
                    if isinstance(frame, LLMContextFrame):
                        index = 0
                        for ctx in frame.context.messages:
                            for key, value in ctx.items():  # type: ignore[union-attr]
                                span.set_attribute(
                                    f"llm.input_messages.{index}.message.{key}",
                                    value,  # type: ignore[arg-type]
                                )
                            index += 1

                # BaseOutputTransport's service_id isn't in self._active_spans but that's ok;
                # just collect text for the turn, not the span
                elif isinstance(frame, TTSTextFrame) and isinstance(
                    data.source, BaseOutputTransport
                ):
                    # Collect bot text from transport output for conversation turn
                    if frame.text and not frame.skip_tts:
                        self._turn_bot_text.append(frame.text)

            # Update Existing Span
            else:
                # Extract and add attributes from this frame to the span
                frame_attrs = extract_attributes_from_frame(frame)
                active_span = self._active_spans[service_id]["span"]
                span_info = self._active_spans[service_id]

                # STT
                if isinstance(frame, VADUserStoppedSpeakingFrame):
                    self._seen_vad_user_stopped_speaking_frame = True

                elif isinstance(frame, TranscriptionFrame):
                    self._stt_includes_inter_frame_spaces = frame.includes_inter_frame_spaces

                    # Collect user text from STT output for conversation turn
                    if self._is_turn_active and frame.text:
                        self._turn_user_text.append(frame.text)
                        self._log_debug(f"  Collected user text: {frame.text[:50]}...")

                    # Collect user text from STT output for STT span
                    span_info["accumulated_input"] += frame.text
                    if not self._stt_includes_inter_frame_spaces:
                        span_info["accumulated_input"] += " "

                    if self._seen_vad_user_stopped_speaking_frame:
                        duration = 0.0
                        current_time_ns = time.time_ns()
                        duration = (current_time_ns - span_info["start_time_ns"]) / 1_000_000_000
                        active_span.set_attribute(
                            "stt.time_to_last_transcription_seconds", duration
                        )
                        ### let _end_turn finish STT Span
                        # # Finish STT Span
                        # self._finish_span(service_id)

                # LLM
                elif isinstance(frame, LLMFullResponseEndFrame):
                    self._log_debug(f"  LLM response ended  Finish span for service {service_id}")
                    # Finish LLM Span
                    self._finish_span(service_id)

                elif isinstance(frame, LLMTextFrame):
                    self._llm_includes_inter_frame_spaces = frame.includes_inter_frame_spaces

                    # Collect user text from LLM output for LLM span
                    span_info["accumulated_output"] += frame.text
                    if not self._llm_includes_inter_frame_spaces:
                        span_info["accumulated_output"] += " "

                # TTS
                ### let _end_turn finish TTS Span
                # elif isinstance(frame, BotStoppedSpeakingFrame):
                #     # Finish TTS Span
                #     self._finish_span(service_id)

                elif isinstance(frame, TTSTextFrame):
                    self._tts_includes_inter_frame_spaces = frame.includes_inter_frame_spaces

                    if isinstance(data.source, TTSService):
                        # Collect full bot text from TTS output for TTS span
                        span_info["accumulated_output"] += frame.text
                        if not self._tts_includes_inter_frame_spaces:
                            span_info["accumulated_output"] += " "

                # Metrics
                elif isinstance(frame, MetricsFrame):
                    for key, value in frame_attrs.items():
                        # ensure this metrics frame is in reference to this service
                        if (
                            "metrics.processor" == key
                            and detect_service_type_from_class_string(value) != service_type
                        ):
                            return

                    for key, value in frame_attrs.items():
                        self._log_debug(f"setting span attribute - {key} : {value}")

                        if key == "service.processing_time_seconds":
                            # Store processing time for use in _finish_span to calculate end_time
                            span_info["processing_time_seconds"] = value
                        active_span.set_attribute(key, value)

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
        if self._turn_span and self._is_turn_active:
            turn_context = trace_api.set_span_in_context(self._turn_span)
            span = self._tracer.start_span(
                name=span_name,
                context=turn_context,
            )
            self._log_debug(f"  Created service span under turn #{self._turn_count}")
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

        self._log_debug(f"finishing {service_id} in active_span/s: {self._active_spans}")

        span_info = self._active_spans.pop(service_id)
        span: Span = span_info["span"]
        start_time_ns = span_info["start_time_ns"]
        service_type = span_info.get("service_type")

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
            if service_type == "llm":
                span.set_attribute("llm.output_messages.0.message.role", "assistant")
                span.set_attribute("llm.output_messages.0.message.content", accumulated_output)

        # For STT spans, minutes used is entirety of the turn, because it is always "listening"
        if service_type == "stt":
            duration = 0.0
            current_time_ns = time.time_ns()
            duration = (
                (current_time_ns - self._turn_start_time) / 1_000_000_000 / 60
            )  # Convert to minutes
            span.set_attribute("stt.minutes", duration)

        span.set_status(trace_api.Status(trace_api.StatusCode.OK))  #
        span.end(end_time=int(end_time_ns))
        return

    async def _start_turn(self, data: FramePushed) -> None:
        """Start a new conversation turn and set it as parent context."""

        await super()._start_turn(data)

        self._turn_start_time = time.time_ns()  # Use our own clock for consistency

        self._log_debug(f"\n{'=' * 60}")
        self._log_debug(f">>> STARTING TURN #{self._turn_count}")
        self._log_debug(f"  Conversation ID: {self._conversation_id}")

        # Create turn span as root (no parent)
        # Each turn will be a separate trace automatically
        # Use an empty context to ensure no ambient parent span is picked up
        span_attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "conversation.turn_number": self._turn_count,
        }

        if self._additional_span_attributes:
            span_attributes.update(self._additional_span_attributes)
        self._turn_span = self._tracer.start_span(
            name="pipecat.conversation.turn",
            context=Context(),  # Empty context ensures this is a true root span
            attributes=span_attributes,  # type: ignore
        )

        if self._conversation_id:
            self._turn_span.set_attribute(  #
                SpanAttributes.SESSION_ID, self._conversation_id
            )
            self._log_debug(f"  Set session.id attribute: {self._conversation_id}")

        self._turn_user_text = []
        self._turn_bot_text = []
        return

    async def _end_turn(self, data: FramePushed, was_interrupted: bool = False) -> None:
        """
        Finish the current conversation turn and detach context.

        Args:
            was_interrupted: Whether the turn was interrupted
        """
        await super()._end_turn(data, was_interrupted)

        if not self._turn_span:
            self._log_debug("  Skipping finish_turn - no active turn")
            return

        # Calculate turn duration
        duration = 0.0
        current_time_ns = time.time_ns()
        duration = (current_time_ns - self._turn_start_time) / 1_000_000_000  # Convert to seconds
        # reset _seen_vad_user_stopped_speaking_frame
        self._seen_vad_user_stopped_speaking_frame = False

        self._log_debug(
            f"\n{'=' * 60}\n"
            + f">>> FINISHING TURN #{self._turn_count}\n"
            + f" (interrupted={was_interrupted}, duration={duration:.2f}s)\n"
            + f"  Active service spans: {len(self._active_spans)}\n"
        )

        # Set input/output attributes
        if self._turn_user_text:
            user_input = " ".join(self._turn_user_text)
            self._turn_span.set_attribute(SpanAttributes.INPUT_VALUE, user_input)
        if self._turn_bot_text:
            if self._tts_includes_inter_frame_spaces:
                join_space = ""
            else:
                join_space = " "
            bot_output = join_space.join(self._turn_bot_text)
            self._turn_span.set_attribute(SpanAttributes.OUTPUT_VALUE, bot_output)

        if len(self._latency_observer._latencies):  # from UserBotLatencyLogObserver
            self._turn_span.set_attribute(
                "conversation.user_to_bot_latency",
                self._latency_observer._latencies[-1],
            )
            self._log_debug(
                f"  Set user_to_bot_latency attribute: {self._latency_observer._latencies[-1]}"
            )
            self._log_debug(
                f"  UserBotLatencyLogObserver latencies: {self._latency_observer._latencies}"
            )
        # Finish all active service spans BEFORE ending the turn span
        # This ensures child spans are ended before the parent
        service_ids_to_finish = list(self._active_spans.keys())
        if len(service_ids_to_finish):
            self._log_debug(f"__ service_ids_to_finish: {service_ids_to_finish}")
        for service_id in service_ids_to_finish:
            self._finish_span(service_id)

        # Set turn metadata
        end_reason = "interrupted" if was_interrupted else "completed"
        self._turn_span.set_attribute("conversation.end_reason", end_reason)  #
        self._turn_span.set_attribute("conversation.turn_duration_seconds", duration)
        self._turn_span.set_attribute("conversation.was_interrupted", was_interrupted)

        # Finish turn span (parent) last
        self._turn_span.set_status(trace_api.Status(trace_api.StatusCode.OK))  #
        self._turn_span.end(end_time=int(current_time_ns))  #

        # Clear turn state
        self._log_debug("  Clearing turn state")
        # self._is_turn_active = False # let super handle this
        self._turn_span = None
        self._turn_context_token = None
