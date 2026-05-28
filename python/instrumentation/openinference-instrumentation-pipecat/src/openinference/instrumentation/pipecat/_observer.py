"""OpenInference observer for Pipecat pipelines."""

import asyncio
import collections
import logging
import time
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Set, TextIO

from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.trace import Span

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.helpers import safe_json_dumps
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
    FunctionCallResultFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    StartFrame,
    STTMuteFrame,
    TranscriptionFrame,
    TTSStartedFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.observers.turn_tracking_observer import TurnTrackingObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
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

    Implements an explicit bidirectional turn state machine: either party can
    initiate a turn, interruptions roll cleanly to a new turn, and the close
    timer fires even when the responding side never speaks.
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
        no_responder_timeout_secs: float = 10.0,
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
            turn_end_timeout_secs: Timeout in seconds after both parties have spoken and gone
                silent before automatically closing the turn.
            no_responder_timeout_secs: Timeout in seconds after only the initiator has spoken
                and gone silent (the responder never spoke) before closing the turn.
            verbose: Optional verbose logging
            kwargs: Additional keyword arguments to pass to the base class
        """
        super().__init__(  # type: ignore[no-untyped-call]
            max_frames=max_frames,
            turn_end_timeout_secs=turn_end_timeout_secs,
            **kwargs,
        )
        self._no_responder_timeout_secs: float = no_responder_timeout_secs

        self._latency_observer: UserBotLatencyObserver = UserBotLatencyObserver()  # type: ignore[no-untyped-call]
        self._last_user_to_bot_latency: Optional[float] = None

        @self._latency_observer.event_handler("on_latency_measured")  # type: ignore[misc]
        async def _on_latency_measured(_observer: Any, latency_seconds: float) -> None:
            self._last_user_to_bot_latency = latency_seconds

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
            try:
                self._debug_log_file = open(debug_log_filename, "w")
                self._log_debug(f"=== Observer initialized for conversation {conversation_id} ===")
                self._log_debug(f"=== Log file: {debug_log_filename} ===")
            except Exception as e:
                logger.error(f"Could not open debug log file: {e}")

        self._active_spans: Dict[int | str, Dict[str, Any]] = {}

        # Turn span + initiator
        self._turn_span: Optional[Span] = None
        self._turn_initiator: Optional[str] = None
        self._turn_start_time: int = 0
        self._turn_user_text: List[str] = []
        self._turn_bot_text: List[str] = []

        self._stt_includes_inter_frame_spaces: bool = False
        self._llm_includes_inter_frame_spaces: bool = False
        self._tts_includes_inter_frame_spaces: bool = False

        # Bidirectional state machine — current activity (survives turn boundaries)
        self._user_speaking: bool = False
        self._bot_speaking: bool = False

        # Bidirectional state machine — per-turn flags (reset on _open_turn)
        self._user_spoken_this_turn: bool = False
        self._bot_spoken_this_turn: bool = False
        self._bot_response_pending: bool = False
        self._last_speaker: Optional[str] = None
        # VAD-stop edge. Non-streaming STT (e.g. Whisper) emits the final
        # TranscriptionFrame AFTER VADUserStoppedSpeakingFrame, so we cannot
        # finish the STT span on the VAD-stop edge or we lose the transcript.
        # Instead, mark that we've seen the stop and finish on the next
        # TranscriptionFrame (which has the final transcript).
        self._seen_vad_user_stopped_speaking_frame: bool = False

        # Close timer (replaces base class's bot-gated timer)
        self._end_turn_timer: Optional[asyncio.TimerHandle] = None

        # Frame dedup (mirror base class behaviour because we no longer
        # delegate to ``super().on_push_frame``)
        self._processed_frames: Set[int] = set()
        self._frame_history: Deque[int] = collections.deque(maxlen=max_frames)

        # Track completed tool calls to avoid duplicate spans
        self._completed_tool_calls: Set[str] = set()

        # Track the LLM span that triggered tool calls (for parent-child relationship)
        # This stores (service_id, span_info) for LLM spans awaiting tool call completion
        self._llm_span_awaiting_tool_calls: Optional[Dict[str, Any]] = None

        # Track active TTS / STT span service_id for efficient lookup
        self._active_tts_service_id: Optional[int] = None
        self._active_stt_service_id: Optional[int] = None

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
        Dispatch frames through (a) the bidirectional turn state machine and
        (b) the service-span builders. We deliberately do NOT call
        ``super().on_push_frame``: the base class's turn logic gates close on
        bot speech, which fails any user-initiated session where the bot never
        responds.
        """
        # Frame dedup applies ONLY to Stage 1 (turn-state events that must
        # fire once per logical frame). The same frame is observed on every
        # push between processors; Stage 1.5 (lazy bot-turn open) and Stage 2
        # (service span dispatch) need to see every push because they branch
        # on ``dst`` / ``src`` being a specific service. Returning early here
        # would drop those observations entirely in any pipeline where the
        # frame originates above the relevant service.
        frame_already_seen = data.frame.id in self._processed_frames
        if not frame_already_seen:
            self._processed_frames.add(data.frame.id)
            self._frame_history.append(data.frame.id)
            if len(self._processed_frames) > len(self._frame_history):
                self._processed_frames = set(self._frame_history)

        await self._latency_observer.on_push_frame(data)

        try:
            src = data.source
            dst = data.destination
            frame = data.frame
            frame_type = frame.__class__.__name__
            source_name = src.__class__.__name__ if src else "Unknown"

            if self._verbose:
                self._log_debug(f"FRAME: {frame_type} from {source_name}")

            # === Stage 1: turn state machine === (dedup'd: once per frame.id)
            if frame_already_seen:
                pass
            elif isinstance(frame, StartFrame):
                if self._turn_count == 0:
                    self._log_debug("  StartFrame seen — first turn opens lazily on first speech")
            elif isinstance(frame, (UserStartedSpeakingFrame, VADUserStartedSpeakingFrame)):
                await self._on_user_started(data)
            elif isinstance(frame, UserStoppedSpeakingFrame):
                await self._on_user_stopped(data)
            elif isinstance(frame, VADUserStoppedSpeakingFrame):
                await self._on_user_stopped(data)
                # STT close strategy depends on STT style:
                #  - Streaming STT (e.g. Deepgram): transcripts have already
                #    arrived during speech, so the active STT span has
                #    accumulated_input. Close it now — no more transcripts
                #    expected for this utterance.
                #  - Non-streaming STT (e.g. OpenAI Whisper): the final
                #    TranscriptionFrame arrives AFTER this VAD-stop frame.
                #    Closing now would drop the transcript both from the
                #    STT span's accumulated_input AND from _turn_user_text
                #    (collection is gated on an active STT span). Defer by
                #    flagging; the next TranscriptionFrame finishes the
                #    span after collecting its text.
                # Safety nets (non-STT span starting, turn close) still
                # apply in either path.
                self._seen_vad_user_stopped_speaking_frame = True
                if self._active_stt_service_id is not None:
                    span_info = self._active_spans.get(self._active_stt_service_id)
                    has_transcript = bool(
                        span_info and span_info.get("accumulated_input", "").strip()
                    )
                    if has_transcript:
                        self._seen_vad_user_stopped_speaking_frame = False
                        self._finish_stt_span(self._active_stt_service_id)
            elif isinstance(frame, BotStartedSpeakingFrame):
                await self._on_bot_started(data)
            elif isinstance(frame, BotStoppedSpeakingFrame):
                await self._on_bot_stopped(data)
            elif isinstance(frame, (EndFrame, CancelFrame)):
                if self._turn_span is not None:
                    await self._close_turn(end_reason="interrupted", interrupted=True)
                self._cancel_close_timer()

            # === Stage 1.5: lazy bot-turn open for bot-initiated service frames ===
            # In a bot-initiated session (e.g. an LLM-driven greeting on session
            # start), the LLM/TTS pipeline emits service frames before any
            # BotStartedSpeakingFrame. Open a turn here so those spans are not
            # dropped. Runs BEFORE Stage 1b so ``_bot_response_pending`` set on
            # LLMContextFrame isn't immediately wiped by ``_open_turn``.
            if self._turn_span is None and (
                (isinstance(frame, LLMContextFrame) and isinstance(dst, LLMService))
                or (isinstance(frame, TTSStartedFrame) and isinstance(src, TTSService))
                or (isinstance(frame, TTSTextFrame) and isinstance(src, BaseOutputTransport))
            ):
                await self._open_turn(initiator="bot")

            # === Stage 1b: bot-response-pending tracking ===
            # LLMContextFrame heading into an LLMService means a response is on
            # the way; LLMFullResponseEndFrame from an LLMService means the
            # response stream finished. These do NOT count as bot speech for
            # turn semantics (only BotStartedSpeakingFrame or transport-source
            # TTSTextFrame do); they just gate the close timer.
            if isinstance(frame, LLMContextFrame) and isinstance(dst, LLMService):
                self._bot_response_pending = True
                self._cancel_close_timer()
            elif isinstance(frame, LLMFullResponseEndFrame) and isinstance(src, LLMService):
                self._bot_response_pending = False
                if not self._user_speaking and not self._bot_speaking:
                    self._schedule_close_timer()

            # === Stage 2: service span dispatch ===
            if isinstance(src, STTService):
                if isinstance(frame, STTMuteFrame):
                    if frame.mute and self._active_stt_service_id is not None:
                        self._log_debug(
                            f"  STTMuteFrame (mute=True) - "
                            f"finishing active STT span {self._active_stt_service_id}"
                        )
                        self._finish_span(self._active_stt_service_id)
                        self._active_stt_service_id = None
                    elif not frame.mute:
                        self._log_debug("  STTMuteFrame (mute=False) - starting new STT span")
                        await self._handle_service_frame(data)
                elif isinstance(
                    frame,
                    (
                        TranscriptionFrame,
                        VADUserStartedSpeakingFrame,
                        MetricsFrame,
                    ),
                ):
                    await self._handle_service_frame(data)
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
                elif isinstance(frame, FunctionCallResultFrame):
                    await self._handle_tool_frame(data)
            elif isinstance(dst, LLMService):
                if isinstance(frame, LLMContextFrame):
                    await self._handle_service_frame(data, override_service=dst)
            elif isinstance(src, TTSService):
                if isinstance(frame, (TTSTextFrame, TTSStartedFrame, MetricsFrame)):
                    await self._handle_service_frame(data)
            elif isinstance(src, BaseOutputTransport):
                if isinstance(frame, TTSTextFrame):
                    await self._handle_service_frame(data)

        except Exception as e:
            logger.debug(f"Error in observer on_push_frame: {e}")

    # ------------------------------------------------------------------
    # State machine — speaker edges
    # ------------------------------------------------------------------

    async def _on_user_started(self, data: FramePushed) -> None:
        """Handle a user-started-speaking edge."""
        if self._turn_span is None:
            await self._open_turn(initiator="user")
        elif self._bot_speaking:
            # Interruption: user starts while bot is still speaking.
            await self._close_turn(end_reason="interrupted", interrupted=True)
            await self._open_turn(initiator="user")
        elif (
            self._user_spoken_this_turn
            and self._bot_spoken_this_turn
            and not self._user_speaking
            and not self._bot_speaking
        ):
            # Roll: both have spoken in the current turn and neither is
            # currently speaking — start a fresh turn for the new utterance.
            await self._close_turn(end_reason="completed")
            await self._open_turn(initiator="user")
        # else: continue the current turn

        if not self._user_speaking:
            self._user_speaking = True
            self._user_spoken_this_turn = True
        self._last_speaker = "user"
        self._cancel_close_timer()

    async def _on_user_stopped(self, data: FramePushed) -> None:
        """Handle a user-stopped-speaking edge."""
        if not self._user_speaking:
            return
        self._user_speaking = False
        self._schedule_close_timer()

    async def _on_bot_started(self, data: FramePushed) -> None:
        """Handle a bot-started-speaking edge."""
        if self._turn_span is None:
            await self._open_turn(initiator="bot")
        elif self._user_speaking:
            await self._close_turn(end_reason="interrupted", interrupted=True)
            await self._open_turn(initiator="bot")
        elif (
            self._user_spoken_this_turn
            and self._bot_spoken_this_turn
            and not self._user_speaking
            and not self._bot_speaking
        ):
            await self._close_turn(end_reason="completed")
            await self._open_turn(initiator="bot")
        # else: continue the current turn

        if not self._bot_speaking:
            self._bot_speaking = True
            self._bot_spoken_this_turn = True
        self._last_speaker = "bot"
        self._cancel_close_timer()

    async def _on_bot_stopped(self, data: FramePushed) -> None:
        """Handle a bot-stopped-speaking edge."""
        if not self._bot_speaking:
            return
        self._bot_speaking = False
        self._schedule_close_timer()

    # ------------------------------------------------------------------
    # Turn lifecycle
    # ------------------------------------------------------------------

    async def _open_turn(self, initiator: str) -> None:
        """Open a new conversation turn span with the given initiator."""
        self._turn_count += 1
        self._turn_start_time = time.time_ns()

        self._log_debug(f"\n{'=' * 60}")
        self._log_debug(f">>> STARTING TURN #{self._turn_count} (initiator={initiator})")
        self._log_debug(f"  Conversation ID: {self._conversation_id}")

        # Reset per-turn state BEFORE creating the span so that any hooks
        # fired by start_span observe a consistent (cleared) view.
        self._turn_initiator = initiator
        self._is_turn_active = True
        self._user_spoken_this_turn = False
        self._bot_spoken_this_turn = False
        self._last_speaker = None
        self._bot_response_pending = False
        self._seen_vad_user_stopped_speaking_frame = False
        self._turn_user_text = []
        self._turn_bot_text = []
        self._active_spans = {}
        self._active_stt_service_id = None
        self._active_tts_service_id = None
        self._completed_tool_calls.clear()
        self._llm_span_awaiting_tool_calls = None
        # Reset latency so each turn only carries its own measurement.
        self._last_user_to_bot_latency = None

        span_attributes: Dict[str, Any] = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "conversation.turn_number": self._turn_count,
            "conversation.initiator": initiator,
        }
        if self._additional_span_attributes:
            span_attributes.update(self._additional_span_attributes)

        self._turn_span = self._tracer.start_span(
            name="pipecat.conversation.turn",
            context=Context(),  # Root span — each turn is its own trace
            attributes=span_attributes,
        )

        if self._conversation_id:
            self._turn_span.set_attribute(SpanAttributes.SESSION_ID, self._conversation_id)
            self._log_debug(f"  Set session.id attribute: {self._conversation_id}")

        await self._call_event_handler("on_turn_started", self._turn_count)

    async def _close_turn(self, end_reason: str, *, interrupted: bool = False) -> None:
        """Close the active conversation turn span."""
        if self._turn_span is None:
            return

        self._cancel_close_timer()

        # Finalise any in-flight service spans before ending the parent span.
        for service_id in list(self._active_spans.keys()):
            self._finish_span(service_id)

        current_time_ns = time.time_ns()
        duration_secs = (current_time_ns - self._turn_start_time) / 1_000_000_000

        self._log_debug(
            f"\n{'=' * 60}\n"
            f">>> FINISHING TURN #{self._turn_count} "
            f"(end_reason={end_reason}, interrupted={interrupted}, "
            f"duration={duration_secs:.2f}s)"
        )

        if self._turn_user_text:
            user_input = " ".join(self._turn_user_text)
            self._turn_span.set_attribute(SpanAttributes.INPUT_VALUE, user_input)
        if self._turn_bot_text:
            join_space = "" if self._tts_includes_inter_frame_spaces else " "
            bot_output = join_space.join(self._turn_bot_text)
            self._turn_span.set_attribute(SpanAttributes.OUTPUT_VALUE, bot_output)

        self._turn_span.set_attribute("conversation.end_reason", end_reason)
        self._turn_span.set_attribute("conversation.was_interrupted", interrupted)
        self._turn_span.set_attribute("conversation.turn_duration_seconds", duration_secs)

        if self._last_user_to_bot_latency is not None:
            self._turn_span.set_attribute(
                "conversation.user_to_bot_latency",
                self._last_user_to_bot_latency,
            )

        self._turn_span.set_status(trace_api.Status(trace_api.StatusCode.OK))
        self._turn_span.end(end_time=int(current_time_ns))

        finished_turn = self._turn_count

        # Clear turn state. Do NOT clear ``_user_speaking`` / ``_bot_speaking``
        # — those reflect current physical activity and survive turn
        # boundaries (e.g. during an interruption the bot may still be
        # producing audio for a brief overlap window).
        self._turn_span = None
        self._turn_initiator = None
        self._is_turn_active = False
        self._last_speaker = None
        self._user_spoken_this_turn = False
        self._bot_spoken_this_turn = False
        self._bot_response_pending = False
        self._seen_vad_user_stopped_speaking_frame = False
        self._turn_user_text = []
        self._turn_bot_text = []
        self._active_spans = {}
        self._active_stt_service_id = None
        self._active_tts_service_id = None
        self._completed_tool_calls.clear()
        self._llm_span_awaiting_tool_calls = None

        await self._call_event_handler("on_turn_ended", finished_turn, duration_secs, interrupted)

    # ------------------------------------------------------------------
    # Close timer
    # ------------------------------------------------------------------

    def _schedule_close_timer(self) -> None:
        """(Re-)arm the close timer based on current responder state."""
        if self._bot_response_pending:
            return
        self._cancel_close_timer()

        if self._user_spoken_this_turn != self._bot_spoken_this_turn:
            # Only the initiator has spoken — wait the no-responder window.
            timeout = self._no_responder_timeout_secs
            pending_end_reason = "no_responder_timeout"
        else:
            # Both have spoken (or neither has — degenerate, but harmless).
            timeout = self._turn_end_timeout_secs
            pending_end_reason = "completed"

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — cannot schedule. Caller has no recourse.
            return

        # Hold the to-be-assigned handle in a one-slot list so the callback
        # closure can read whichever handle was ultimately stored on ``self``.
        # Default-arg binding (``reason=pending_end_reason``) avoids stale
        # closure capture if this method's locals are mutated later.
        handle_ref: List[Optional[asyncio.TimerHandle]] = [None]

        def _fire(reason: str = pending_end_reason) -> None:
            asyncio.create_task(self._on_close_timer_fired(reason, handle_ref[0]))

        handle = loop.call_later(timeout, _fire)
        handle_ref[0] = handle
        self._end_turn_timer = handle

    def _cancel_close_timer(self) -> None:
        if self._end_turn_timer is not None:
            self._end_turn_timer.cancel()
            self._end_turn_timer = None

    async def _on_close_timer_fired(
        self,
        end_reason: str,
        fired_handle: Optional[asyncio.TimerHandle] = None,
    ) -> None:
        """Close timer callback — close the turn or re-arm if state changed."""
        # Only clear ``_end_turn_timer`` if it's still the handle we were fired
        # for — between OS-level fire and this task running, another path may
        # have scheduled a new timer that we must not clobber.
        if fired_handle is None or self._end_turn_timer is fired_handle:
            self._end_turn_timer = None
        if self._turn_span is None:
            return
        if not self._user_speaking and not self._bot_speaking and not self._bot_response_pending:
            await self._close_turn(end_reason=end_reason)
        else:
            self._schedule_close_timer()

    # ------------------------------------------------------------------
    # Service / tool span handling
    # ------------------------------------------------------------------

    def _finish_stt_span(self, service_id: int) -> None:
        """
        Finish an STT span, stamping the time-to-last-transcription duration.

        Called when one of:
          - the final TranscriptionFrame arrives after a VAD-stop edge
            (normal path; the duration covers span-start to final transcript);
          - a non-STT service span starts while an STT span is still active
            (safety net for missing/late transcripts);
          - the turn closes (terminal fallback).
        """
        span_info = self._active_spans.get(service_id)
        if span_info is not None:
            duration = (time.time_ns() - span_info["start_time_ns"]) / 1_000_000_000
            span_info["span"].set_attribute("stt.time_to_last_transcription_seconds", duration)
        self._finish_span(service_id)
        # _finish_span already clears _active_stt_service_id when applicable,
        # but be explicit in case the span was never registered.
        if self._active_stt_service_id == service_id:
            self._active_stt_service_id = None

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
            ## LLMContextFrame (LLM)
            ## STTMuteFrame [if mute:false] (STT)
            ## TTSStartedFrame (TTS)
            ## VADUserStartedSpeakingFrame (STT)

            if service_id not in self._active_spans:
                if isinstance(
                    frame,
                    (LLMContextFrame, STTMuteFrame, TTSStartedFrame, VADUserStartedSpeakingFrame),
                ):
                    # For TTS: consecutive TTS operations are merged into one span.
                    # When a NEW non-TTS span starts (LLM/STT), finish any active TTS span first.
                    if service_type != "tts" and self._active_tts_service_id is not None:
                        self._log_debug(
                            f"  Non-TTS span ({service_type}) starting - "
                            f"finishing active TTS span {self._active_tts_service_id}"
                        )
                        self._finish_span(self._active_tts_service_id)
                        self._active_tts_service_id = None

                    # Safety net: a non-STT span starting finishes any lingering
                    # STT span. Normally the explicit VADUserStoppedSpeakingFrame
                    # handler already closed it.
                    if service_type != "stt" and self._active_stt_service_id is not None:
                        self._log_debug(
                            f"  Non-STT span ({service_type}) starting - "
                            f"finishing active STT span {self._active_stt_service_id}"
                        )
                        self._finish_stt_span(self._active_stt_service_id)

                    self._log_debug(f"  {service_type.upper()} response STARTED. ({frame})")

                    # Service frames must arrive inside an active turn. The
                    # state machine opens a turn on the first speech edge —
                    # if we ever see a service frame outside that window, log
                    # and skip rather than silently creating an orphan span.
                    if not self._is_turn_active or self._turn_span is None:
                        self._log_debug(
                            f"  No active turn - skipping {service_type} span "
                            f"creation for service {service_id}"
                        )
                        return

                    self._log_debug(f"  CREATING new SPAN for {service_type}: {service_id}")
                    span = self._create_service_span(service, service_type)
                    self._active_spans[service_id] = {
                        "span": span,
                        "service_type": service_type,
                        "frame_count": 0,
                        "accumulated_input": "",
                        "accumulated_output": "",
                        "start_time_ns": time.time_ns(),
                        "processing_time_seconds": None,
                    }
                    span_info = self._active_spans[service_id]
                    span_info["frame_count"] += 1

                    if service_type == "tts":
                        self._active_tts_service_id = service_id
                    elif service_type == "stt":
                        self._active_stt_service_id = service_id

                    if isinstance(frame, LLMContextFrame):
                        span.set_attributes(extract_attributes_from_frame(frame))

                # Transport-source TTSTextFrame: bot text without its own span.
                elif isinstance(frame, TTSTextFrame) and isinstance(
                    data.source, BaseOutputTransport
                ):
                    if frame.text and not frame.skip_tts:
                        self._turn_bot_text.append(frame.text)
                        # Transport-source TTS counts as bot speech for the
                        # state machine even without a BotStartedSpeakingFrame.
                        self._bot_spoken_this_turn = True

                # Defensive: TranscriptionFrame whose STT span has already
                # been closed (e.g. by an early non-STT span starting).
                # Still capture the text in the turn so the user message is
                # never dropped from input.value. Also clear the VAD-stop
                # flag so it doesn't leak into a later turn.
                elif isinstance(frame, TranscriptionFrame):
                    if self._is_turn_active and frame.text:
                        self._turn_user_text.append(frame.text)
                        self._log_debug(
                            f"  Collected user text (STT span closed): {frame.text[:50]}..."
                        )
                    self._seen_vad_user_stopped_speaking_frame = False

            else:
                # Update existing span
                frame_attrs = extract_attributes_from_frame(frame)
                active_span = self._active_spans[service_id]["span"]
                span_info = self._active_spans[service_id]

                # STT
                if isinstance(frame, TranscriptionFrame):
                    self._stt_includes_inter_frame_spaces = frame.includes_inter_frame_spaces

                    # Collect user text from STT output for conversation turn
                    if self._is_turn_active and frame.text:
                        self._turn_user_text.append(frame.text)
                        self._log_debug(f"  Collected user text: {frame.text[:50]}...")

                    # Collect user text from STT output for STT span
                    span_info["accumulated_input"] += frame.text
                    if not self._stt_includes_inter_frame_spaces:
                        span_info["accumulated_input"] += " "

                    # If this transcript completes a VAD-stop edge, close the
                    # STT span now that we have the final text.
                    if self._seen_vad_user_stopped_speaking_frame:
                        self._seen_vad_user_stopped_speaking_frame = False
                        self._finish_stt_span(service_id)

                # LLM
                elif isinstance(frame, LLMFullResponseEndFrame):
                    accumulated_output = span_info.get("accumulated_output", "").strip()
                    if not accumulated_output:
                        # No text output means this LLM call likely triggered tool calls.
                        # Store a reference so tool spans can be parented under it.
                        self._log_debug(
                            f"  LLM response ended with no text output - "
                            f"storing span {service_id} as potential tool call parent"
                        )
                        self._llm_span_awaiting_tool_calls = {
                            "service_id": service_id,
                            "span": active_span,
                        }
                    else:
                        # LLM produced text output, clear any awaiting tool call parent
                        self._llm_span_awaiting_tool_calls = None
                        self._log_debug(
                            "  LLM response ended with output - clearing tool call parent"
                        )
                    self._log_debug(f"  LLM response ended  Finish span for service {service_id}")
                    self._finish_span(service_id)

                elif isinstance(frame, LLMTextFrame):
                    self._llm_includes_inter_frame_spaces = frame.includes_inter_frame_spaces
                    span_info["accumulated_output"] += frame.text
                    if not self._llm_includes_inter_frame_spaces:
                        span_info["accumulated_output"] += " "

                # TTS
                elif isinstance(frame, TTSTextFrame):
                    self._tts_includes_inter_frame_spaces = frame.includes_inter_frame_spaces

                    if isinstance(data.source, TTSService):
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
                            span_info["processing_time_seconds"] = value
                        active_span.set_attribute(key, value)

    async def _handle_tool_frame(self, data: FramePushed) -> None:
        """
        Handle tool/function call frames.

        Creates TOOL spans only from FunctionCallResultFrame to avoid duplicate spans
        from multiple FunctionCallInProgressFrame events.
        """
        frame = data.frame
        tool_call_id = getattr(frame, "tool_call_id", None)

        if not tool_call_id:
            self._log_debug(f"  Tool frame without tool_call_id: {frame}")
            return

        self._log_debug(f"TOOL FRAME: {frame.__class__.__name__}, tool_call_id: {tool_call_id}")

        if isinstance(frame, FunctionCallResultFrame):
            if tool_call_id in self._completed_tool_calls:
                self._log_debug(f"  Tool call {tool_call_id} already processed, skipping")
                return

            if tool_call_id in self._active_spans:
                self._log_debug(f"  Tool span already exists for {tool_call_id}, skipping")
                return

            if not self._is_turn_active or self._turn_span is None:
                self._log_debug(
                    f"  No active turn - skipping tool span creation for {tool_call_id}"
                )
                return

            self._log_debug(f"  CREATING new TOOL SPAN for: {tool_call_id}")
            span = self._create_tool_span(frame)

            frame_attrs = extract_attributes_from_frame(frame)

            arguments = getattr(frame, "arguments", None)
            result = getattr(frame, "result", None)

            self._log_debug(f"  tool_call frame.result = {result}")
            self._log_debug(
                f"  tool_call safe_json_dumps(arguments) = {safe_json_dumps(arguments)}"
            )

            self._active_spans[tool_call_id] = {
                "span": span,
                "service_type": "tool",
                "frame_count": 1,
                "accumulated_input": safe_json_dumps(arguments) if arguments else "",
                "accumulated_output": (safe_json_dumps(result) if result is not None else ""),
                "start_time_ns": time.time_ns(),
                "processing_time_seconds": None,
            }

            for key, value in frame_attrs.items():
                span.set_attribute(key, value)

            self._log_debug(f"  Tool call completed, finishing span for {tool_call_id}")
            self._finish_tool_span(tool_call_id)

            self._completed_tool_calls.add(tool_call_id)

    def _create_tool_span(self, frame: Frame) -> Span:
        """
        Create a TOOL span for a function call.

        Tool spans are created as children of the LLM span that triggered them
        (if available), otherwise as children of the turn span.
        """
        function_name = getattr(frame, "function_name", "unknown_tool")
        span_name = f"pipecat.tool.{function_name}"
        self._log_debug(f">>> Creating tool span: {span_name}")

        if self._llm_span_awaiting_tool_calls:
            parent_span = self._llm_span_awaiting_tool_calls["span"]
            parent_context = trace_api.set_span_in_context(parent_span)
            span = self._tracer.start_span(
                name=span_name,
                context=parent_context,
            )
            self._log_debug(
                f"  Created tool span under LLM span "
                f"{self._llm_span_awaiting_tool_calls['service_id']}"
            )
        elif self._turn_span and self._is_turn_active:
            turn_context = trace_api.set_span_in_context(self._turn_span)
            span = self._tracer.start_span(
                name=span_name,
                context=turn_context,
            )
            self._log_debug(f"  Created tool span under turn #{self._turn_count}")
        else:
            self._log_debug("  WARNING: No active turn! Creating root span for tool")
            span = self._tracer.start_span(name=span_name)

        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.TOOL.value,
        )
        span.set_attribute(SpanAttributes.TOOL_NAME, function_name)

        return span

    def _finish_tool_span(self, tool_call_id: str) -> None:
        """Finish a tool span."""
        if tool_call_id not in self._active_spans:
            return

        self._log_debug(f"Finishing tool span for: {tool_call_id}")

        span_info = self._active_spans.pop(tool_call_id)
        span: Span = span_info["span"]
        start_time_ns = span_info["start_time_ns"]

        processing_time_seconds = span_info.get("processing_time_seconds")
        if processing_time_seconds is not None:
            end_time_ns = start_time_ns + int(processing_time_seconds * 1_000_000_000)
        else:
            end_time_ns = time.time_ns()

        accumulated_input = span_info.get("accumulated_input", "")
        accumulated_output = span_info.get("accumulated_output", "")
        service_type = span_info.get("service_type", "")

        if accumulated_input:
            span.set_attribute(SpanAttributes.INPUT_VALUE, accumulated_input)
            if service_type == "tool":
                span.set_attribute(SpanAttributes.TOOL_PARAMETERS, accumulated_input)

        if accumulated_output:
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, accumulated_output)

        span.set_status(trace_api.Status(trace_api.StatusCode.OK))
        span.end(end_time=int(end_time_ns))

    def _create_service_span(
        self,
        service: FrameProcessor,
        service_type: str,
    ) -> Span:
        """
        Create a span for a service with type-specific attributes.
        All service spans are created as children of the turn span.
        """
        span_name = f"pipecat.{service_type}"
        self._log_debug(f">>> Creating {service_type} span")

        if self._turn_span and self._is_turn_active:
            turn_context = trace_api.set_span_in_context(self._turn_span)
            span = self._tracer.start_span(
                name=span_name,
                context=turn_context,
            )
            self._log_debug(f"  Created service span under turn #{self._turn_count}")
        else:
            self._log_debug(f"  WARNING: No active turn! Creating root span for {service_type}")
            span = self._tracer.start_span(
                name=span_name,
            )

        span.set_attribute("service.name", service.__class__.__name__)

        service_attrs = extract_service_attributes(service)
        span.set_attributes(service_attrs)
        self._log_debug(f"  Set attributes: {service_attrs}")

        return span

    def _finish_span(self, service_id: int | str) -> None:
        """Finish a span for a service."""
        if service_id not in self._active_spans:
            return

        self._log_debug(f"finishing {service_id} in active_span/s: {self._active_spans}")

        span_info = self._active_spans.pop(service_id)
        span: Span = span_info["span"]
        start_time_ns = span_info["start_time_ns"]
        service_type = span_info.get("service_type")

        # Clear TTS/STT tracking if finishing their span
        if service_type == "tts" and self._active_tts_service_id == service_id:
            self._active_tts_service_id = None
        elif service_type == "stt" and self._active_stt_service_id == service_id:
            self._active_stt_service_id = None

        processing_time_seconds = span_info.get("processing_time_seconds")
        if processing_time_seconds is not None:
            end_time_ns = start_time_ns + int(processing_time_seconds * 1_000_000_000)
        else:
            end_time_ns = time.time_ns()

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

            if service_type == "llm":
                span.set_attribute("llm.output_messages.0.message.role", "assistant")
                span.set_attribute("llm.output_messages.0.message.content", accumulated_output)

        # For STT spans, minutes used is entirety of the turn (always listening)
        if service_type == "stt":
            current_time_ns = time.time_ns()
            duration_minutes = (current_time_ns - self._turn_start_time) / 1_000_000_000 / 60
            span.set_attribute("stt.minutes", duration_minutes)

        span.set_status(trace_api.Status(trace_api.StatusCode.OK))
        span.end(end_time=int(end_time_ns))
