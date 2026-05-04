"""End-to-end smoke test exercising the observer through a real PipelineTask.

Unlike test_observer.py (which calls observer methods directly with synthetic
frames), this test wires the observer into pipecat's own test harness so we
catch integration drift — frame ordering, lifecycle hooks, the
TurnTrackingObserver parent class, and the PipelineTask wrapper.
"""

import json

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.tests.utils import SleepFrame, run_test

from openinference.instrumentation.pipecat._observer import OpenInferenceObserver


class StubLLMService(LLMService):
    """Minimal LLMService that emits a canned response when it sees a context frame.

    Real LLM providers subclass LLMService and override process_frame to call
    out to a model. For the smoke test we just need the observer to see the
    same frame shapes that flow in production: an LLMContextFrame in,
    bracketed Start/Text/End frames out.
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        # Forward every frame downstream so lifecycle frames (StartFrame, EndFrame)
        # reach the sink and terminate the pipeline.
        await self.push_frame(frame, direction)
        if isinstance(frame, LLMContextFrame):
            await self.push_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
            await self.push_frame(LLMTextFrame("Hi there"), FrameDirection.DOWNSTREAM)
            await self.push_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)


async def test_observer_creates_spans_through_real_pipeline(
    observer: OpenInferenceObserver,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Drive a real PipelineTask through pipecat's run_test harness.

    Verifies that, given a normal LLM request/response frame flow, the observer
    produces a turn span with an LLM child span carrying the response text.
    """
    service = StubLLMService()
    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])

    # SleepFrame between the request and the implicit EndFrame gives our stub
    # time to emit its synthesized response frames before TurnTrackingObserver
    # ends the turn on EndFrame.
    await run_test(
        service,
        frames_to_send=[LLMContextFrame(context=context), SleepFrame(sleep=0.1)],
        observers=[observer],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    span_names = [s.name for s in spans]

    assert "pipecat.conversation.turn" in span_names, span_names
    assert "pipecat.llm" in span_names, span_names

    llm_span = next(s for s in spans if s.name == "pipecat.llm")
    turn_span = next(s for s in spans if s.name == "pipecat.conversation.turn")

    assert llm_span.parent is not None
    assert llm_span.parent.span_id == turn_span.context.span_id

    assert llm_span.attributes is not None
    assert "Hi there" in str(llm_span.attributes.get("output.value", ""))


async def test_observer_handles_dict_valued_message_content(
    observer: OpenInferenceObserver,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Multimodal/dict message content must be normalized before reaching OTel.

    OpenTelemetry rejects dict attribute values, so the observer's LLM-context
    path must JSON-serialize structured content. Without normalization the
    `llm.input_messages.0.message.content` attribute is dropped entirely.
    """
    service = StubLLMService()
    structured_content = {"type": "text", "text": "Hello"}
    context = LLMContext(messages=[{"role": "user", "content": structured_content}])

    await run_test(
        service,
        frames_to_send=[LLMContextFrame(context=context), SleepFrame(sleep=0.1)],
        observers=[observer],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.name == "pipecat.llm")
    assert llm_span.attributes is not None

    content_attr = llm_span.attributes.get("llm.input_messages.0.message.content")
    assert content_attr == json.dumps(structured_content)


async def test_observer_handles_llm_specific_message(
    observer: OpenInferenceObserver,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """LLMSpecificMessage entries in the context must be unwrapped, not skipped.

    Provider-specific message wrappers don't expose `.items()`, so iterating
    them like dicts raises and the input attrs are lost. The observer must
    unwrap to the inner payload before setting span attributes.
    """
    service = StubLLMService()
    context = LLMContext(
        messages=[
            LLMSpecificMessage(
                llm="openai",
                message={"role": "user", "content": "Hello from provider-specific"},
            ),
        ]
    )

    await run_test(
        service,
        frames_to_send=[LLMContextFrame(context=context), SleepFrame(sleep=0.1)],
        observers=[observer],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.name == "pipecat.llm")
    assert llm_span.attributes is not None

    role = llm_span.attributes.get("llm.input_messages.0.message.role")
    content = llm_span.attributes.get("llm.input_messages.0.message.content")
    assert role == "user", f"expected role 'user', got {role!r} (LLMSpecificMessage skipped)"
    assert content == "Hello from provider-specific"
