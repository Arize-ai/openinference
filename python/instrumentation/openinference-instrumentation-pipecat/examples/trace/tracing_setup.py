"""
Arize AX Tracing Setup for Pipecat Voice Agent

This module configures OpenTelemetry tracing to send telemetry data to Arize AX
for comprehensive observability of the voice agent pipeline.

Pure OpenInference Conventions for GenAI Use Cases:
- CHAIN: Used for ALL manual operations (pipeline, session, LLM service setup, etc.)
- Auto-instrumented spans: Keep their appropriate kinds (ChatCompletion=LLM, etc.)
- Attributes: Only OpenInference semantic conventions (SpanAttributes.*)
- Custom data: Stored in SpanAttributes.METADATA for proper categorization
"""

import os
import logging
import atexit
import asyncio
import json
import threading
import time
from typing import Optional, Callable, Any, Dict
from functools import wraps
from opentelemetry import trace as trace_api
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import SpanProcessor, ReadableSpan
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
from arize.otel import register

# For overriding Pipecat's internal tracing
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global tracer provider and tracer
_tracer_provider = None
_tracer = None


# Turn-based tracing state management
class TurnTracker:
    """Manages conversation turns for separate trace creation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._current_turn_span = None
        self._turn_counter = 0
        self._user_speaking = False
        self._bot_speaking = False
        self._turn_start_time = None
        self._session_id = None
        self._conversation_input = ""
        self._conversation_output = ""
        self._context_token = None
        self._tts_parent_span = None
        self._stt_parent_span = None
        self._llm_parent_span = None
        self._stt_full_output = ""
        self._tts_full_input = ""

    def set_session_id(self, session_id: str):
        """Set the session ID for all subsequent turns."""
        with self._lock:
            self._session_id = session_id
            logger.debug(f"📍 Set session ID: {session_id}")

    def add_conversation_input(self, text: str):
        """Add user input to the current conversation."""
        with self._lock:
            if self._conversation_input:
                self._conversation_input += " " + text
            else:
                self._conversation_input = text

    def add_conversation_output(self, text: str):
        """Add bot output to the current conversation."""
        with self._lock:
            if self._conversation_output:
                self._conversation_output += " " + text
            else:
                self._conversation_output = text

    def start_user_turn(self) -> trace_api.Span:
        """Start a new root trace when user begins speaking."""
        with self._lock:
            # Check if there's an active turn and what phase we're in
            if self._current_turn_span and self._current_turn_span.is_recording():
                if self._bot_speaking:
                    # User is interrupting while bot is speaking (TTS phase)
                    # End the current trace and start a new one
                    logger.info(
                        f"🔄 User interrupting bot speech - ending turn {self._turn_counter}, starting new turn"
                    )
                    self._end_current_turn("User interrupted bot speech")
                    # Continue to create new turn below
                else:
                    # User is speaking during STT/LLM phase - continue existing turn
                    logger.debug(
                        f"⚠️ User continuing to speak during turn {self._turn_counter} - same trace"
                    )
                    return self._current_turn_span

            self._turn_counter += 1
            self._user_speaking = True
            self._bot_speaking = False
            self._turn_start_time = time.time()
            # Reset conversation input/output for new turn
            self._conversation_input = ""
            self._conversation_output = ""
            self._tts_parent_span = None
            self._stt_parent_span = None
            self._llm_parent_span = None
            self._stt_full_output = ""
            self._tts_full_input = ""

            tracer = get_tracer()
            if not tracer:
                return None

            # Create span attributes with session ID if available
            attributes = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                "conversation.turn_number": self._turn_counter,
                "conversation.speaker": "user",
                "conversation.turn_type": "user_initiated",
                "conversation.start_time": self._turn_start_time,
            }

            # Add session ID if available
            if self._session_id:
                attributes["session.id"] = self._session_id

            # Create a new ROOT trace for this turn (no parent context)
            # Use a fresh context to ensure this is a root span
            fresh_context = context_api.Context()
            self._current_turn_span = tracer.start_span(
                "Interaction", context=fresh_context, attributes=attributes
            )

            # Set this as the active span for all subsequent operations
            self._activate_turn_span()

            logger.debug(
                f"🎙️ Started ROOT trace {self._turn_counter} - Interaction (session: {self._session_id or 'unknown'})"
            )
            return self._current_turn_span

    def mark_user_finished_speaking(self):
        """Mark that user has finished speaking (but turn continues with bot response)."""
        with self._lock:
            if self._current_turn_span and self._user_speaking:
                self._user_speaking = False
                # Add event to mark user finished speaking
                self._current_turn_span.add_event(
                    "user_finished_speaking",
                    attributes={"event.timestamp": time.time()},
                )
                logger.debug(f"👤 User finished speaking in turn {self._turn_counter}")

    def mark_bot_started_speaking(self):
        """Mark that bot has started speaking (still within the same turn)."""
        with self._lock:
            if self._current_turn_span and not self._user_speaking:
                self._bot_speaking = True

                # Add event to mark bot started speaking
                self._current_turn_span.add_event(
                    "bot_started_speaking", attributes={"event.timestamp": time.time()}
                )
                logger.debug(f"🤖 Bot started speaking in turn {self._turn_counter}")

    def end_bot_turn(self):
        """End the current turn when bot finishes speaking."""
        with self._lock:
            if self._current_turn_span and self._bot_speaking:
                self._bot_speaking = False
                self._end_current_turn("Turn completed - Bot finished speaking")
                logger.debug(
                    f"✅ Completed turn {self._turn_counter} - Bot finished speaking"
                )

    def _end_current_turn(self, reason: str):
        """Internal method to end the current turn span."""
        if self._current_turn_span:
            duration = (
                time.time() - self._turn_start_time if self._turn_start_time else 0
            )

            # Add full conversation input/output to the root span
            if self._conversation_input:
                self._current_turn_span.set_attribute(
                    SpanAttributes.INPUT_VALUE, self._conversation_input[:1000]
                )  # Truncate for readability
            if self._conversation_output:
                self._current_turn_span.set_attribute(
                    SpanAttributes.OUTPUT_VALUE, self._conversation_output[:1000]
                )  # Truncate for readability

            self._current_turn_span.set_attribute("conversation.end_reason", reason)
            self._current_turn_span.set_attribute(
                "conversation.duration_seconds", duration
            )
            self._current_turn_span.set_status(
                trace_api.Status(trace_api.StatusCode.OK)
            )
            self._current_turn_span.end()

            # Close any remaining parent spans
            if self._llm_parent_span:
                self._llm_parent_span.set_status(
                    trace_api.Status(trace_api.StatusCode.OK)
                )
                self._llm_parent_span.end()
                self._llm_parent_span = None
                logger.debug("🧠 Closed LLM parent span at interaction end (fallback)")

            if self._tts_parent_span:
                self._tts_parent_span.set_status(
                    trace_api.Status(trace_api.StatusCode.OK)
                )
                self._tts_parent_span.end()
                self._tts_parent_span = None
                logger.debug("🔊 Closed TTS parent span at interaction end")

            if self._stt_parent_span:
                self._stt_parent_span.set_status(
                    trace_api.Status(trace_api.StatusCode.OK)
                )
                self._stt_parent_span.end()
                self._stt_parent_span = None
                logger.debug("🎤 Closed STT parent span at interaction end")

            self._current_turn_span = None
            self._turn_start_time = None
            # Reset conversation data
            self._conversation_input = ""
            self._conversation_output = ""
            self._stt_full_output = ""
            self._tts_full_input = ""

            # Force flush after each turn to ensure traces are sent
            force_flush_traces()

    def _activate_turn_span(self):
        """Set the current turn span as active in the context, overriding any previous context."""
        if self._current_turn_span:
            # Create a completely fresh context with only our turn span
            # This ensures that LLM and TTS spans will be children of the interaction, not setup spans
            turn_context = trace_api.set_span_in_context(
                self._current_turn_span, context_api.Context()
            )
            token = context_api.attach(turn_context)
            # Store the token so we can detach it later if needed
            self._context_token = token
            logger.debug(
                f"🔄 Activated turn span context - all subsequent spans will be children of Interaction"
            )

    def get_current_turn_span(self) -> Optional[trace_api.Span]:
        """Get the current active turn span."""
        return self._current_turn_span

    def is_in_turn(self) -> bool:
        """Check if we're currently in an active turn."""
        return self._current_turn_span is not None

    def get_turn_number(self) -> int:
        """Get the current turn number."""
        return self._turn_counter

    def cleanup(self):
        """Clean up any active turn span."""
        with self._lock:
            if self._current_turn_span:
                self._end_current_turn("Session ended")


# Global turn tracker instance
_turn_tracker = TurnTracker()

# OpenInferenceOnlyProcessor removed - no longer needed since we disable
# competing auto-instrumentations at the source using OTEL_PYTHON_DISABLED_INSTRUMENTATIONS


def accept_current_state():
    """
    Set up manual span creation for TTS and STT operations.

    The strategy is:
    1. Our manual spans use proper OpenInference conventions (CHAIN)
    2. ChatCompletion spans use proper OpenInference conventions (LLM)
    3. TTS/STT spans are manually created by monkey patching service methods
    4. All spans get exported to Arize
    """
    logger.info("🚀 Setting up manual span creation for TTS/STT operations")
    logger.info("📊 Strategy:")
    logger.info("   • Manual spans: OpenInference CHAIN ✅")
    logger.info("   • ChatCompletion spans: OpenInference LLM ✅")
    logger.info("   • TTS/STT spans: Manual creation via monkey patching ✅")
    logger.info("   • Arize export: All spans sent as-is ✅")


class _NoOpSpan:
    """No-op span that doesn't create any traces"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, *args):
        pass

    def set_attributes(self, *args):
        pass

    def record_exception(self, *args):
        pass

    def set_status(self, *args):
        pass

    def add_event(self, *args):
        pass


# Removed problematic GenAISpanKindProcessor - it was causing issues


def get_turn_tracker() -> TurnTracker:
    """Get the global turn tracker instance."""
    return _turn_tracker


def set_session_id(session_id: str):
    """Set the session ID for all subsequent turns."""
    _turn_tracker.set_session_id(session_id)


def add_conversation_input(text: str):
    """Add user input to the current conversation."""
    _turn_tracker.add_conversation_input(text)


def add_conversation_output(text: str):
    """Add bot output to the current conversation."""
    _turn_tracker.add_conversation_output(text)


def start_conversation_turn():
    """Start a new conversation turn when user begins speaking."""
    return _turn_tracker.start_user_turn()


def mark_user_finished():
    """Mark that user has finished speaking."""
    _turn_tracker.mark_user_finished_speaking()


def mark_bot_started():
    """Mark that bot has started speaking."""
    _turn_tracker.mark_bot_started_speaking()


def end_conversation_turn():
    """End the current conversation turn when bot finishes speaking."""
    _turn_tracker.end_bot_turn()


def get_current_turn_span():
    """Get the current active turn span."""
    return _turn_tracker.get_current_turn_span()


def is_in_conversation_turn():
    """Check if we're currently in an active conversation turn."""
    return _turn_tracker.is_in_turn()


def cleanup_turn_tracking():
    """Clean up turn tracking on shutdown."""
    _turn_tracker.cleanup()


def patch_pipecat_span_creation():
    """
    Monkey patch OpenAI TTS, STT, and LLM service methods to create manual spans for every operation.
    Also integrate turn-based tracing triggers.
    """
    logger.info(
        "🔧 Patching OpenAI TTS, STT, and LLM services for manual spans and turn-based tracing"
    )

    try:
        # Import the service classes
        from pipecat.services.openai.llm import OpenAILLMService
        from pipecat.services.openai.stt import OpenAISTTService
        from pipecat.services.openai.tts import OpenAITTSService
        import asyncio
        import functools
        from opentelemetry import context as context_api

        # Store original methods
        original_openai_llm_process_frame = OpenAILLMService.process_frame
        original_openai_stt_transcribe = OpenAISTTService._transcribe
        original_openai_tts_run_tts = OpenAITTSService.run_tts

        @functools.wraps(original_openai_llm_process_frame)
        async def traced_openai_llm_process_frame(self, frame, direction):
            """Wrapped OpenAI LLM process_frame method with manual span creation"""
            tracer = get_tracer()
            if not tracer:
                # Fallback to original if no tracer
                return await original_openai_llm_process_frame(self, frame, direction)

            # Check if we have an active turn, if not, create one for LLM processing
            current_span = get_current_turn_span()
            if not current_span or not current_span.is_recording():
                # LLM is being called without an active interaction - start one
                logger.info(
                    "🤖 LLM called without active interaction - starting new interaction"
                )
                turn_span = start_conversation_turn()
                current_span = get_current_turn_span()

            if current_span and current_span.is_recording():
                # Ensure the interaction context is active for OpenAI instrumentation
                with trace_api.use_span(current_span):
                    # Get or create persistent LLM parent span
                    turn_tracker = get_turn_tracker()
                    if not turn_tracker._llm_parent_span:
                        # Create LLM parent span - we'll add input/output as we process
                        turn_tracker._llm_parent_span = tracer.start_span(
                            "LLM",
                            attributes={
                                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                                "service.name": "openai",
                                "model": getattr(self, "_model", "gpt-3.5-turbo"),
                                "conversation.turn_number": get_turn_tracker().get_turn_number(),
                            },
                        )
                        logger.debug(
                            "🧠 Created persistent LLM parent span for interaction"
                        )

                    # Extract input from the frame if it has messages
                    llm_input = None
                    if hasattr(frame, "messages") and frame.messages:
                        # Get the last user message as LLM input
                        for msg in reversed(frame.messages):
                            if msg.get("role") == "user":
                                llm_input = msg.get("content", "")
                                break
                        if llm_input:
                            turn_tracker._llm_parent_span.set_attribute(
                                SpanAttributes.INPUT_VALUE, llm_input[:1000]
                            )
                            logger.debug(f"🧠 Added LLM input: '{llm_input[:50]}...'")

                    # If no messages in frame, use conversation input as fallback
                    elif turn_tracker._conversation_input:
                        turn_tracker._llm_parent_span.set_attribute(
                            SpanAttributes.INPUT_VALUE,
                            turn_tracker._conversation_input[:1000],
                        )
                        logger.debug(
                            f"🧠 Added LLM input (fallback): '{turn_tracker._conversation_input[:50]}...'"
                        )
                        llm_input = turn_tracker._conversation_input

                    # Use the persistent LLM parent span for all LLM calls
                    with trace_api.use_span(turn_tracker._llm_parent_span):
                        try:
                            # The OpenAI instrumentation will create child ChatCompletion spans under the LLM parent
                            result = await original_openai_llm_process_frame(
                                self, frame, direction
                            )

                            # Try to extract LLM output from the result
                            if hasattr(result, "text") and result.text:
                                # Update LLM parent span with output
                                turn_tracker._llm_parent_span.set_attribute(
                                    SpanAttributes.OUTPUT_VALUE, result.text[:1000]
                                )
                                logger.debug(
                                    f"🧠 Added LLM output: '{result.text[:50]}...'"
                                )
                            elif hasattr(result, "content") and result.content:
                                turn_tracker._llm_parent_span.set_attribute(
                                    SpanAttributes.OUTPUT_VALUE, result.content[:1000]
                                )
                                logger.debug(
                                    f"🧠 Added LLM output: '{result.content[:50]}...'"
                                )

                            return result

                        except Exception as e:
                            turn_tracker._llm_parent_span.record_exception(e)
                            turn_tracker._llm_parent_span.set_status(
                                trace_api.Status(trace_api.StatusCode.ERROR, str(e))
                            )
                            raise
            else:
                # Fallback if no current turn span can be created
                logger.warning("⚠️ LLM processing without interaction context")
                return await original_openai_llm_process_frame(self, frame, direction)

        @functools.wraps(original_openai_tts_run_tts)
        async def traced_openai_tts_run_tts(self, text: str):
            """Wrapped OpenAI TTS method with manual span creation and turn-based tracing"""
            tracer = get_tracer()
            if not tracer:
                # Fallback to original if no tracer
                async for frame in original_openai_tts_run_tts(self, text):
                    yield frame
                return

            # TURN-BASED TRACING: Mark bot started speaking
            if is_in_conversation_turn():
                mark_bot_started()
                # Capture conversation output
                add_conversation_output(text)
                logger.info(
                    f"🤖 Bot started speaking: '{text[:50]}...' - Turn {get_turn_tracker().get_turn_number()}"
                )

            # Get the current turn span
            current_span = get_current_turn_span()
            if not current_span or not current_span.is_recording():
                # TTS is being called without an active interaction - start one
                logger.info(
                    "🔊 OpenAI TTS called without active interaction - starting new interaction"
                )
                turn_span = start_conversation_turn()
                current_span = get_current_turn_span()

            if current_span and current_span.is_recording():
                # Ensure the interaction context is active
                with trace_api.use_span(current_span):
                    # Get or create TTS parent span
                    turn_tracker = get_turn_tracker()
                    if not turn_tracker._tts_parent_span:
                        # Close LLM parent span when TTS starts
                        if turn_tracker._llm_parent_span:
                            if not turn_tracker._llm_parent_span.attributes.get(
                                SpanAttributes.OUTPUT_VALUE
                            ):
                                turn_tracker._llm_parent_span.set_attribute(
                                    SpanAttributes.OUTPUT_VALUE, text[:1000]
                                )
                                logger.debug(
                                    f"🧠 Added LLM output from TTS text: '{text[:50]}...'"
                                )

                            turn_tracker._llm_parent_span.set_status(
                                trace_api.Status(trace_api.StatusCode.OK)
                            )
                            turn_tracker._llm_parent_span.end()
                            turn_tracker._llm_parent_span = None
                            logger.debug("🧠 Closed LLM parent span - starting TTS")

                        turn_tracker._tts_parent_span = tracer.start_span(
                            "TTS",
                            attributes={
                                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                                "service.name": "openai",
                                "voice": getattr(self, "_voice", "unknown"),
                                "model": getattr(self, "_model", "tts-1"),
                                "conversation.turn_number": get_turn_tracker().get_turn_number(),
                            },
                        )
                        logger.debug(
                            "🔊 Created OpenAI TTS parent span for interaction"
                        )

                    # Add this TTS text to the full input
                    turn_tracker._tts_full_input += text + " "

                    # Update TTS parent span with accumulated input
                    turn_tracker._tts_parent_span.set_attribute(
                        SpanAttributes.INPUT_VALUE,
                        turn_tracker._tts_full_input.strip()[:1000],
                    )

                    # Use the persistent TTS parent span
                    with trace_api.use_span(turn_tracker._tts_parent_span):
                        try:
                            # Call original method and yield frames
                            frame_count = 0
                            async for frame in original_openai_tts_run_tts(self, text):
                                frame_count += 1
                                yield frame

                            # Add frame count to parent span
                            turn_tracker._tts_parent_span.set_attribute(
                                "total_frames", frame_count
                            )

                            # TURN-BASED TRACING: End the conversation turn when TTS finishes
                            if is_in_conversation_turn():
                                end_conversation_turn()
                                logger.info(
                                    f"✅ Bot finished speaking - Ended turn {get_turn_tracker().get_turn_number()}"
                                )

                        except Exception as e:
                            if turn_tracker._tts_parent_span:
                                turn_tracker._tts_parent_span.record_exception(e)
                                turn_tracker._tts_parent_span.set_status(
                                    trace_api.Status(trace_api.StatusCode.ERROR, str(e))
                                )
                            if is_in_conversation_turn():
                                end_conversation_turn()
                            raise
            else:
                # Fallback - standalone span
                with tracer.start_as_current_span(
                    "tts",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                        SpanAttributes.INPUT_VALUE: text[:500],
                        "service.name": "openai",
                        "voice": getattr(self, "_voice", "unknown"),
                        "model": getattr(self, "_model", "tts-1"),
                    },
                ) as span:
                    try:
                        frame_count = 0
                        async for frame in original_openai_tts_run_tts(self, text):
                            frame_count += 1
                            yield frame
                        span.set_attribute("frame_count", frame_count)
                        span.set_status(trace_api.Status(trace_api.StatusCode.OK))
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(
                            trace_api.Status(trace_api.StatusCode.ERROR, str(e))
                        )
                        raise

        @functools.wraps(original_openai_stt_transcribe)
        async def traced_openai_stt_transcribe(self, audio: bytes):
            """Wrapped OpenAI STT _transcribe method with manual span creation and turn-based tracing"""
            tracer = get_tracer()
            if not tracer:
                # Fallback to original if no tracer
                return await original_openai_stt_transcribe(self, audio)

            # TURN-BASED TRACING: Start a new conversation turn when user speaks (BEFORE transcription)
            start_conversation_turn()
            logger.info(
                f"🎙️ User started speaking - Starting turn {get_turn_tracker().get_turn_number()}"
            )

            # Get the current turn span
            current_span = get_current_turn_span()
            if not current_span or not current_span.is_recording():
                # No turn span - just call original
                logger.warning("⚠️ STT called without turn span")
                return await original_openai_stt_transcribe(self, audio)

            # Ensure the interaction context is active for OpenAI instrumentation
            with trace_api.use_span(current_span):
                # Get or create STT parent span
                turn_tracker = get_turn_tracker()
                if not turn_tracker._stt_parent_span:
                    turn_tracker._stt_parent_span = tracer.start_span(
                        "STT",
                        attributes={
                            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                            "service.name": "openai",
                            "model": getattr(self, "_model", "whisper-1"),
                            "conversation.turn_number": get_turn_tracker().get_turn_number(),
                        },
                    )
                    logger.debug("🎤 Created OpenAI STT parent span for interaction")

                # Use the persistent STT parent span and call transcribe within it
                with trace_api.use_span(turn_tracker._stt_parent_span):
                    try:
                        # Call original transcribe method - OpenAI instrumentation will create child spans
                        result = await original_openai_stt_transcribe(self, audio)

                        if result and result.text and result.text.strip():
                            transcript = result.text

                            # Capture conversation input
                            add_conversation_input(transcript)

                            # Add to STT full output
                            turn_tracker._stt_full_output += transcript + " "

                            # Update STT parent span with accumulated output
                            turn_tracker._stt_parent_span.set_attribute(
                                SpanAttributes.OUTPUT_VALUE,
                                turn_tracker._stt_full_output.strip()[:1000],
                            )

                            # TURN-BASED TRACING: Mark user finished speaking
                            mark_user_finished()
                            logger.debug(
                                f"👤 User finished speaking: '{transcript[:50]}...' in turn {get_turn_tracker().get_turn_number()}"
                            )

                        return result

                    except Exception as e:
                        if turn_tracker._stt_parent_span:
                            turn_tracker._stt_parent_span.record_exception(e)
                            turn_tracker._stt_parent_span.set_status(
                                trace_api.Status(trace_api.StatusCode.ERROR, str(e))
                            )
                        raise

        # Apply the patches
        OpenAILLMService.process_frame = traced_openai_llm_process_frame
        OpenAISTTService._transcribe = traced_openai_stt_transcribe
        OpenAITTSService.run_tts = traced_openai_tts_run_tts

        logger.info(
            "✅ Successfully patched OpenAI TTS, STT, and LLM services for manual span creation"
        )

    except Exception as e:
        logger.warning(f"Failed to patch TTS/STT/LLM services: {e}")
        raise


def setup_arize_tracing():
    """
    Set up Arize AX tracing with proper configuration for development and production.
    """
    global _tracer_provider, _tracer

    try:
        # STEP 1: Set up enhanced tracing strategy
        accept_current_state()

        # STEP 2: Minimal instrumentation disabling - only disable truly competing ones
        disabled_instrumentations = [
            "traceloop-sdk"  # Only disable traceloop which can conflict
        ]

        # Let Pipecat's native tracing work by not disabling its instrumentations
        existing_disabled = os.getenv("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS", "")
        if existing_disabled:
            all_disabled = f"{existing_disabled},{','.join(disabled_instrumentations)}"
        else:
            all_disabled = ",".join(disabled_instrumentations)

        os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = all_disabled
        logger.info(f"🚫 Minimal disabled instrumentations: {all_disabled}")
        logger.info("🔧 Allowing Pipecat's native TTS/STT instrumentation to work")

        # Get configuration from environment
        space_id = os.getenv("ARIZE_SPACE_ID")
        api_key = os.getenv("ARIZE_API_KEY")
        project_name = os.getenv("ARIZE_PROJECT_NAME", "pipecat-voice-agent")
        is_development = (
            os.getenv("DEVELOPMENT", "false").lower() == "true"
            or os.getenv("LOCAL_RUN", "false").lower() == "true"
        )

        if not space_id or not api_key:
            logger.warning(
                "Arize credentials not found in environment. Tracing will be disabled."
            )
            return None

        logger.info(f"🔭 Initializing Arize AX Tracing (Native Mode) 🔭")
        logger.info(f"|  Project: {project_name}")
        logger.info(f"|  Development Mode: {is_development}")
        logger.info(f"|  Mode: OpenInference + Native Pipecat spans")

        # STEP 3: Register with Arize using their helper function
        _tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name=project_name,
            # Use immediate export in development for better debugging
            batch=not is_development,
            log_to_console=is_development,
        )

        # Set as global tracer provider
        trace_api.set_tracer_provider(_tracer_provider)

        # Get tracer
        _tracer = trace_api.get_tracer(__name__)
        # STEP 5: Create manual spans for TTS, STT, and LLM operations
        try:
            patch_pipecat_span_creation()
            logger.info("🔧 Manual TTS/STT/LLM span creation enabled")

        except Exception as e:
            logger.warning(f"Failed to set up manual span creation: {e}")

        logger.info(
            "🎯 Manual span creation mode: Create spans for every TTS/STT/LLM operation"
        )
        logger.info("📝 Manual spans: OpenInference CHAIN kind ✅")
        logger.info("🤖 ChatCompletion spans: OpenInference LLM kind ✅")
        logger.info("🔧 TTS/STT/LLM spans: Manual span creation ✅")

        logger.info("✅ Arize AX tracing initialized successfully")

        # Register cleanup on exit
        atexit.register(shutdown_tracing)

        return _tracer_provider

    except Exception as e:
        logger.error(f"Failed to initialize Arize AX tracing: {e}")
        return None


def get_tracer():
    """Get the configured tracer instance."""
    return _tracer or trace_api.get_tracer(__name__)


def force_flush_traces():
    """Force flush all pending traces to Arize AX."""
    try:
        if _tracer_provider and hasattr(_tracer_provider, "force_flush"):
            _tracer_provider.force_flush(timeout_millis=5000)
            logger.debug("✅ Traces flushed to Arize AX")
    except Exception as e:
        logger.debug(f"Trace flush failed (this is normal on shutdown): {e}")


def shutdown_tracing():
    """Gracefully shutdown tracing infrastructure."""
    try:
        # Clean up turn tracking first
        cleanup_turn_tracking()

        if _tracer_provider and hasattr(_tracer_provider, "shutdown"):
            _tracer_provider.shutdown()
            logger.debug("✅ Tracing infrastructure shut down")
    except Exception as e:
        logger.debug(f"Tracing shutdown failed (this is normal): {e}")


def capture_current_context():
    """Capture the current OpenTelemetry context for async propagation."""
    return context_api.get_current()


def with_context_propagation(func: Callable) -> Callable:
    """
    Decorator that ensures proper context propagation for async functions.
    Based on Arize documentation for async context propagation.
    """
    if asyncio.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Capture the current context before the async call
            current_context = capture_current_context()

            # Attach the context in this async function
            token = context_api.attach(current_context)
            try:
                return await func(*args, **kwargs)
            finally:
                context_api.detach(token)

        return async_wrapper
    else:

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return sync_wrapper


def trace_voice_agent_operation(operation_name: str, span_kind: str = "CHAIN"):
    """
    Decorator for tracing voice agent operations with proper async context propagation.

    Args:
        operation_name: Name of the operation being traced
        span_kind: OpenInference span kind. Use "CHAIN" for general operations, "LLM" for LLM calls
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()

            # Determine span kind
            span_kind_value = getattr(
                OpenInferenceSpanKindValues,
                span_kind.upper(),
                OpenInferenceSpanKindValues.CHAIN,
            ).value

            with tracer.start_as_current_span(
                operation_name,
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: span_kind_value,
                },
            ) as span:
                # Add function metadata using OpenInference conventions
                metadata = {
                    "function_name": func.__name__,
                    "operation_type": operation_name,
                }
                span.set_attribute(SpanAttributes.METADATA, json.dumps(metadata))

                try:
                    if asyncio.iscoroutinefunction(func):
                        # For async functions, we need to run them with proper context propagation
                        current_context = context_api.get_current()

                        async def async_wrapper():
                            token = context_api.attach(current_context)
                            try:
                                return await func(*args, **kwargs)
                            finally:
                                context_api.detach(token)

                        # Return the coroutine
                        return async_wrapper()
                    else:
                        # For sync functions, run directly
                        result = func(*args, **kwargs)
                        span.set_attribute(
                            SpanAttributes.OUTPUT_VALUE, str(result)[:500]
                        )  # Truncate large outputs
                        return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(
                        trace_api.Status(trace_api.StatusCode.ERROR, str(e))
                    )
                    raise

        return wrapper

    return decorator


def create_session_span(
    session_id: str, session_type: str = "voice_agent"
) -> trace_api.Span:
    """
    Create a main session span that will be the parent for all operations.
    This ensures all traces are connected under one main trace.
    """
    tracer = get_tracer()

    session_span = tracer.start_span(
        f"pipecat_session_{session_type}",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "session.id": session_id,
            "session.type": session_type,
            "agent.name": "pipecat-voice-agent",
            "agent.version": "1.0.0",
        },
    )

    # Set this span as the current span in context
    context_with_span = trace_api.set_span_in_context(session_span)
    context_api.attach(context_with_span)

    return session_span


def end_session_span(
    session_span: trace_api.Span, session_summary: str = "Session completed"
):
    """
    End the session span and ensure all traces are flushed.
    """
    try:
        session_span.set_attribute(SpanAttributes.OUTPUT_VALUE, session_summary)
        session_span.set_status(trace_api.Status(trace_api.StatusCode.OK))
        session_span.end()

        # Force flush on session end to ensure all data is sent
        force_flush_traces()

    except Exception as e:
        logger.error(f"Error ending session span: {e}")


def add_session_metadata(**metadata):
    """Add metadata to the current span context."""
    current_span = trace_api.get_current_span()
    if current_span and current_span.is_recording():
        for key, value in metadata.items():
            if value is not None:
                current_span.set_attribute(f"session.{key}", str(value))


def trace_llm_interaction(prompt: str, response: str, model: str = "unknown"):
    """Add LLM interaction tracing to current span using OpenInference conventions."""
    current_span = trace_api.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(
            "llm_interaction",
            attributes={
                SpanAttributes.LLM_MODEL_NAME: model,
                SpanAttributes.INPUT_VALUE: prompt[:500],  # Truncate for readability
                SpanAttributes.OUTPUT_VALUE: response[:500],
            },
        )


def trace_audio_processing(operation: str, details: dict = None):
    """Add audio processing events to current span using OpenInference conventions."""
    current_span = trace_api.get_current_span()
    if current_span and current_span.is_recording():
        # Use metadata for custom audio processing attributes
        metadata = {"audio_operation": operation}
        if details:
            for key, value in details.items():
                metadata[f"audio_{key}"] = str(value)

        current_span.add_event(
            "audio_processing",
            attributes={SpanAttributes.METADATA: json.dumps(metadata)},
        )


def trace_pipeline_event(event_name: str, **attributes):
    """Add pipeline events to current span using OpenInference conventions."""
    current_span = trace_api.get_current_span()
    if current_span and current_span.is_recording():
        # Use metadata for pipeline-specific attributes
        metadata = {}
        for key, value in attributes.items():
            metadata[f"pipeline_{key}"] = str(value) if value is not None else "None"

        current_span.add_event(
            event_name, attributes={SpanAttributes.METADATA: json.dumps(metadata)}
        )


def create_llm_operation_span(operation_name: str, model: str, input_text: str = None):
    """Create a CHAIN span for LLM operations using pure OpenInference conventions."""
    tracer = get_tracer()
    if not tracer:
        return None

    current_context = context_api.get_current()

    span = tracer.start_span(
        operation_name,
        context=current_context,
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            SpanAttributes.LLM_MODEL_NAME: model,
        },
    )

    if input_text:
        span.set_attribute(SpanAttributes.INPUT_VALUE, input_text[:500])  # Truncate

    return span


def create_tts_operation_span(
    operation_name: str, text: str, voice_id: str = None, model: str = None
):
    """Create a CHAIN span for TTS operations using pure OpenInference conventions."""
    tracer = get_tracer()
    if not tracer:
        return None

    current_context = context_api.get_current()

    attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
        SpanAttributes.INPUT_VALUE: text[:500],  # Truncate for readability
    }

    # Add TTS-specific metadata
    metadata = {"operation_type": "text_to_speech"}
    if voice_id:
        metadata["voice_id"] = voice_id
    if model:
        metadata["model"] = model

    attributes[SpanAttributes.METADATA] = json.dumps(metadata)

    span = tracer.start_span(
        operation_name, context=current_context, attributes=attributes
    )

    return span


def finish_llm_span(span, output_text: str = None, token_usage: dict = None):
    """Finish an LLM span with output and token usage information."""
    if not span or not span.is_recording():
        return

    if output_text:
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_text[:500])  # Truncate

    if token_usage:
        if "prompt_tokens" in token_usage:
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT, token_usage["prompt_tokens"]
            )
        if "completion_tokens" in token_usage:
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,
                token_usage["completion_tokens"],
            )
        if "total_tokens" in token_usage:
            span.set_attribute(
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL, token_usage["total_tokens"]
            )

    span.set_status(trace_api.Status(trace_api.StatusCode.OK))
    span.end()


def finish_tts_span(span, duration: float = None, character_count: int = None):
    """Finish a TTS span with duration and character count information."""
    if not span or not span.is_recording():
        return

    metadata = {}
    if duration:
        metadata["duration_seconds"] = duration
    if character_count:
        metadata["character_count"] = character_count

    if metadata:
        span.set_attribute(SpanAttributes.METADATA, json.dumps(metadata))

    span.set_status(trace_api.Status(trace_api.StatusCode.OK))
    span.end()


# Context manager for session-level tracing (minimal for turn-based tracing)
class SessionTracer:
    def __init__(self, session_id: str, session_type: str = "voice_agent"):
        self.session_id = session_id
        self.session_type = session_type
        # No session span creation - each user turn will be independent

    def __enter__(self):
        # Just log the session start, but don't create any spans
        logger.info(
            f"📍 Session started: {self.session_id} (type: {self.session_type})"
        )
        logger.info(
            "🔄 Turn-based tracing: Each user utterance creates independent traces"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Just log the session end
        if exc_type:
            logger.info(f"❌ Session ended with error: {self.session_id} - {exc_val}")
        else:
            logger.info(f"✅ Session completed: {self.session_id}")

        # Force flush traces at session end to ensure all turn traces are sent
        force_flush_traces()


def create_child_span_with_context(name: str, span_kind: str = "CHAIN", **attributes):
    """
    Create a child span that properly inherits from the current context.
    Useful for manual span creation in async operations.

    Args:
        name: Name of the span
        span_kind: OpenInference span kind ("CHAIN" for general ops, "LLM" for LLM calls)
        **attributes: Additional span attributes
    """
    tracer = get_tracer()

    # Get current context to ensure proper parent-child relationship
    current_context = context_api.get_current()

    span_kind_value = getattr(
        OpenInferenceSpanKindValues,
        span_kind.upper(),
        OpenInferenceSpanKindValues.CHAIN,
    ).value

    # Create span with current context as parent
    span = tracer.start_span(
        name,
        context=current_context,
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: span_kind_value,
            **attributes,
        },
    )

    return span
