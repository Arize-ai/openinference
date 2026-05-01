"""End-to-end smoke test exercising the observer through a real PipelineTask.

Unlike test_observer.py (which calls observer methods directly with synthetic
frames), this test wires the observer into pipecat's own test harness so we
catch integration drift — frame ordering, lifecycle hooks, the
TurnTrackingObserver parent class, and the PipelineTask wrapper.
"""

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
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
