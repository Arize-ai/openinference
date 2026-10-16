"""LlamaIndex spans must be current so sibling instrumentors nest under them.

Without attaching the span in new_span, OpenAIInstrumentor (and any user span
opened with start_as_current_span during a LlamaIndex call) starts a new root
trace. See https://github.com/Arize-ai/openinference/issues/4053.
"""

from typing import Iterator

import httpx
import pytest
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

dispatcher = get_dispatcher(__name__)


@pytest.fixture
def setup_instrumentation(
    tracer_provider: TracerProvider,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LlamaIndexInstrumentor().uninstrument()


def test_downstream_span_nests_under_llama_index_span(
    setup_instrumentation: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """A span started during a decorated call should parent under the LlamaIndex span."""

    @dispatcher.span
    def outer() -> None:
        assert trace_api.get_current_span().get_span_context().is_valid
        with tracer_provider.get_tracer(__name__).start_as_current_span("downstream"):
            pass

    outer()

    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    assert "downstream" in spans
    outer_spans = [s for s in in_memory_span_exporter.get_finished_spans() if s.name.endswith("outer")]
    assert len(outer_spans) == 1
    outer_span = outer_spans[0]
    child = spans["downstream"]
    assert child.parent is not None
    assert child.parent.span_id == outer_span.context.span_id
    assert child.context.trace_id == outer_span.context.trace_id
    # After the call returns, the LlamaIndex span must not remain current
    # (streaming tests rely on this for unrelated root spans).
    assert not trace_api.get_current_span().get_span_context().is_valid


def test_openai_chat_span_shares_trace_with_llama_index_llm(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """With OpenAIInstrumentor enabled, ChatCompletion nests under OpenAI.chat."""
    from openinference.instrumentation.openai import OpenAIInstrumentor

    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    try:

        def answer(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "Paris."},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 19,
                        "completion_tokens": 2,
                        "total_tokens": 21,
                    },
                },
            )

        llm = OpenAI(
            model="gpt-4o-mini",
            api_key="sk-test",
            http_client=httpx.Client(transport=httpx.MockTransport(answer)),
        )
        llm.chat([ChatMessage(role="user", content="Capital of France?")])
    finally:
        OpenAIInstrumentor().uninstrument()
        LlamaIndexInstrumentor().uninstrument()

    spans = list(in_memory_span_exporter.get_finished_spans())
    assert spans, "expected finished spans"
    roots = [s for s in spans if s.parent is None]
    children = [s for s in spans if s.parent is not None]
    assert children, (
        f"expected nested spans under LlamaIndex; got roots={[s.name for s in roots]} "
        f"all={[s.name for s in spans]}"
    )
    parent_ids = {s.parent.span_id for s in children if s.parent is not None}
    span_ids = {s.context.span_id for s in spans}
    assert parent_ids & span_ids
    # Bug was two independent roots (OpenAI.chat + ChatCompletion).
    assert len(roots) == 1, f"expected one root trace, got {[s.name for s in roots]}"
