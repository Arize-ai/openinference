import contextlib
from functools import partial
from typing import Any, Callable, ContextManager, Iterator, NamedTuple, Optional

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    TracerProvider,
    _spans,
    get_llm_attributes,
    suppress_tracing,
)
from openinference.instrumentation._genai_attributes import GenAIAttributes
from openinference.instrumentation.logging import logger as openinference_logger
from openinference.semconv.trace import SpanAttributes

_GENAI_CONFIG = TraceConfig(enable_genai_semconv=True)
_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
_PARENT_SPAN_ID = "b7ad6b7169203331"


def _remote_parent(*, sampled: bool) -> context_api.Context:
    trace_flags = "01" if sampled else "00"
    return TraceContextTextMapPropagator().extract(
        {"traceparent": f"00-{_TRACE_ID}-{_PARENT_SPAN_ID}-{trace_flags}"}
    )


@contextlib.contextmanager
def _attached(ctx: context_api.Context) -> Iterator[None]:
    token = context_api.attach(ctx)
    try:
        yield
    finally:
        context_api.detach(token)


def _provider(exporter: InMemorySpanExporter, **kwargs: Any) -> TracerProvider:
    tracer_provider = TracerProvider(config=_GENAI_CONFIG, **kwargs)
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return tracer_provider


class _NonRecordingScenario(NamedTuple):
    tracer: OITracer
    exporter: InMemorySpanExporter
    surrounding: Callable[[], ContextManager[Any]]


@pytest.fixture(
    params=[
        "always_off_sampler",
        "unsampled_remote_parent",
        "suppress_tracing",
        "otel_sdk_disabled",
        "no_op_tracer_provider",
    ]
)
def scenario(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> _NonRecordingScenario:
    exporter = InMemorySpanExporter()
    surrounding: Callable[[], ContextManager[Any]] = contextlib.nullcontext
    if request.param == "always_off_sampler":
        tracer = _provider(exporter, sampler=ALWAYS_OFF).get_tracer(__name__)
    elif request.param == "unsampled_remote_parent":
        tracer = _provider(exporter).get_tracer(__name__)
        surrounding = partial(_attached, _remote_parent(sampled=False))
    elif request.param == "suppress_tracing":
        tracer = _provider(exporter).get_tracer(__name__)
        surrounding = suppress_tracing
    elif request.param == "otel_sdk_disabled":
        monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
        tracer = _provider(exporter).get_tracer(__name__)
    else:
        tracer = OITracer(trace_api.NoOpTracerProvider().get_tracer(__name__), _GENAI_CONFIG)
    return _NonRecordingScenario(tracer, exporter, surrounding)


def test_start_as_current_span(scenario: _NonRecordingScenario) -> None:
    with scenario.surrounding():
        with scenario.tracer.start_as_current_span(
            "llm-span", openinference_span_kind="llm"
        ) as span:
            span.set_attributes(
                get_llm_attributes(
                    model_name="gpt-4o-mini",
                    input_messages=[{"role": "user", "content": "hello"}],
                )
            )
            span.set_input("hello")
            span.set_output("hi")

    assert not span.is_recording()
    assert not scenario.exporter.get_finished_spans()


def test_start_span_and_end(scenario: _NonRecordingScenario) -> None:
    with scenario.surrounding():
        span = scenario.tracer.start_span("llm-span", openinference_span_kind="llm")
        span.set_attributes(get_llm_attributes(model_name="gpt-4o-mini"))
        span.end()

    assert not scenario.exporter.get_finished_spans()


def test_llm_decorator(scenario: _NonRecordingScenario) -> None:
    @scenario.tracer.llm
    def call_llm(prompt: str) -> str:
        return "hi"

    with scenario.surrounding():
        assert call_llm("hello") == "hi"

    assert not scenario.exporter.get_finished_spans()


async def test_async_llm_decorator(scenario: _NonRecordingScenario) -> None:
    @scenario.tracer.llm
    async def call_llm(prompt: str) -> str:
        return "hi"

    with scenario.surrounding():
        assert await call_llm("hello") == "hi"

    assert not scenario.exporter.get_finished_spans()


def test_chain_decorator(scenario: _NonRecordingScenario) -> None:
    @scenario.tracer.chain
    def run_chain(question: str) -> str:
        return "answer"

    with scenario.surrounding():
        assert run_chain("question") == "answer"

    assert not scenario.exporter.get_finished_spans()


def test_agent_decorator(scenario: _NonRecordingScenario) -> None:
    @scenario.tracer.agent
    def run_agent(task: str) -> str:
        return "done"

    with scenario.surrounding():
        assert run_agent("task") == "done"

    assert not scenario.exporter.get_finished_spans()


def test_tool_decorator(scenario: _NonRecordingScenario) -> None:
    @scenario.tracer.tool
    def get_weather(city: str) -> str:
        return "sunny"

    with scenario.surrounding():
        assert get_weather("Paris") == "sunny"

    assert not scenario.exporter.get_finished_spans()


def test_sampled_remote_parent_still_emits_genai_attributes() -> None:
    exporter = InMemorySpanExporter()
    tracer = _provider(exporter).get_tracer(__name__)

    with _attached(_remote_parent(sampled=True)):
        with tracer.start_as_current_span("llm-span", openinference_span_kind="llm") as span:
            span.set_attributes(get_llm_attributes(model_name="gpt-4o-mini"))

    (exported_span,) = exporter.get_finished_spans()
    assert exported_span.context is not None
    assert format(exported_span.context.trace_id, "032x") == _TRACE_ID
    attributes = dict(exported_span.attributes or {})
    assert attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"


class _RecordingSpanWithoutReadableAttributes(trace_api.NonRecordingSpan):
    def __init__(self) -> None:
        super().__init__(trace_api.INVALID_SPAN_CONTEXT)
        self.ended = False

    def is_recording(self) -> bool:
        return True

    def end(self, end_time: Optional[int] = None) -> None:
        self.ended = True


def test_recording_span_without_readable_attributes_still_ends() -> None:
    opaque_span = _RecordingSpanWithoutReadableAttributes()

    class _OpaqueTracer(trace_api.NoOpTracer):
        def start_span(self, *args: Any, **kwargs: Any) -> trace_api.Span:
            return opaque_span

    tracer = OITracer(_OpaqueTracer(), _GENAI_CONFIG)

    with tracer.start_as_current_span("llm-span", openinference_span_kind="llm") as span:
        span.set_attributes(get_llm_attributes(model_name="gpt-4o-mini"))

    assert opaque_span.ended


@pytest.fixture
def failing_genai_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(attributes: Any) -> Any:
        raise ValueError("unexpected attribute shape")

    monkeypatch.setattr(_spans, "get_genai_attributes", _raise)


@pytest.fixture
def openinference_logs(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    openinference_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        openinference_logger.removeHandler(caplog.handler)


@pytest.mark.usefixtures("failing_genai_conversion")
def test_genai_conversion_failure_still_ends_span(
    openinference_logs: pytest.LogCaptureFixture,
) -> None:
    exporter = InMemorySpanExporter()
    tracer = _provider(exporter).get_tracer(__name__)

    @tracer.llm
    def call_llm(prompt: str) -> str:
        return "hi"

    with tracer.start_as_current_span("llm-span", openinference_span_kind="llm") as span:
        span.set_attributes(get_llm_attributes(model_name="gpt-4o-mini"))
    assert call_llm("hello") == "hi"

    exported_spans = exporter.get_finished_spans()
    assert len(exported_spans) == 2
    for exported_span in exported_spans:
        attributes = dict(exported_span.attributes or {})
        assert attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "LLM"
        assert not any(key.startswith("gen_ai.") for key in attributes)
    assert (
        openinference_logs.text.count("Failed to derive GenAI semantic convention attributes.") == 2
    )
