"""Tests for the run's real input and output on the trace root span.

None of the span data types describing an agent operation carry input or output, so these
values come from the runner call and from ``RunHooks.on_agent_end`` instead. That depends
on patching a runner whose internals have been reorganised more than once, so the tests
that matter here are the end-to-end ones: they drive a real ``Runner`` through a fake
model and assert on the exported spans.

The cases worth reading first are the ones that defeated inferring these values from
child LLM spans: ``test_llm_calling_input_guardrail_does_not_supply_the_root_output`` and
``test_structured_output_type_is_recorded_as_json``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import pytest

try:
    from agents import (
        Agent,
        GuardrailFunctionOutput,
        RunHooks,
        Runner,
        function_tool,
        input_guardrail,
        trace,
    )
    from agents.exceptions import MaxTurnsExceeded
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import (
        Response,
        ResponseCompletedEvent,
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )
except ImportError:
    # Handle compatibility issue with OpenAI SDK >=1.103.0 where WebSearchToolFilters was removed
    # Introduced in: https://github.com/openai/openai-python/commit/3d3d16a
    pytest.skip(
        "agents package incompatible with current OpenAI SDK version", allow_module_level=True
    )

from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from openinference.instrumentation import TraceConfig, suppress_tracing
from openinference.instrumentation.config import REDACTED_VALUE
from openinference.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
    _patch_agent_runner,
)
from openinference.instrumentation.openai_agents._run_io import (
    RunIO,
    _run_io,
    find_agent_runner_bindings,
    input_attributes,
    make_sync_run_wrapper,
    output_attributes,
    run_hooks_class,
)

_ANSWER = "It is 21C and sunny."


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="m1",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )


class _FakeModel(Model):
    """Answers with ``text``, first calling ``tool_name`` once when one is given."""

    def __init__(self, text: str = _ANSWER, tool_name: Optional[str] = None) -> None:
        self.text = text
        self.tool_name = tool_name
        self.turns = 0

    def _output(self) -> list[Any]:
        self.turns += 1
        if self.tool_name is not None and self.turns == 1:
            return [
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call-1",
                    name=self.tool_name,
                    arguments="{}",
                )
            ]
        return [_message(self.text)]

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        return ModelResponse(output=self._output(), usage=Usage(), response_id=None)

    async def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        response = Response(
            id="resp-1",
            created_at=0.0,
            model="fake",
            object="response",
            output=self._output(),
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        yield ResponseCompletedEvent(
            type="response.completed", response=response, sequence_number=0
        )


def _instrument(config: Optional[TraceConfig] = None) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    kwargs: dict[str, Any] = {"tracer_provider": provider}
    if config is not None:
        kwargs["config"] = config
    OpenAIAgentsInstrumentor().instrument(**kwargs)
    return exporter


@pytest.fixture
def exporter() -> Any:
    instance = _instrument()
    yield instance
    OpenAIAgentsInstrumentor().uninstrument()


@pytest.fixture
def masked_exporter() -> Any:
    instance = _instrument(TraceConfig(hide_inputs=True, hide_outputs=True))
    yield instance
    OpenAIAgentsInstrumentor().uninstrument()


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _root(spans: list[ReadableSpan]) -> dict[str, Any]:
    """The trace root: the AGENT span the processor opens for the trace itself.

    Identified by having no parent rather than by name, because the SDK also opens a
    task span carrying the same workflow name.
    """
    matching = [_attrs(s) for s in spans if s.parent is None]
    assert len(matching) == 1, f"expected exactly one root span, got {len(matching)}"
    return matching[0]


def _roots(spans: list[ReadableSpan]) -> list[dict[str, Any]]:
    return [_attrs(s) for s in spans if s.parent is None]


# --- end to end through the real SDK ------------------------------------------------


async def test_root_span_records_the_runs_input_and_output(
    exporter: InMemorySpanExporter,
) -> None:
    agent = Agent(name="WeatherAgent", model=_FakeModel())

    result = await Runner.run(agent, "What's the weather in London?")
    assert result.final_output == _ANSWER

    attrs = _root(list(exporter.get_finished_spans()))
    assert attrs["openinference.span.kind"] == "AGENT"
    assert attrs["input.value"] == "What's the weather in London?"
    assert attrs["output.value"] == _ANSWER
    assert attrs["input.mime_type"] == "text/plain"
    assert attrs["output.mime_type"] == "text/plain"


async def test_a_list_input_is_recorded_as_json(exporter: InMemorySpanExporter) -> None:
    agent = Agent(name="WeatherAgent", model=_FakeModel())
    messages: list[Any] = [{"role": "user", "content": "What's the weather in London?"}]

    await Runner.run(agent, list(messages))

    attrs = _root(list(exporter.get_finished_spans()))
    assert attrs["input.mime_type"] == "application/json"
    assert json.loads(str(attrs["input.value"])) == messages


async def test_streaming_run_records_input_and_output(exporter: InMemorySpanExporter) -> None:
    """``run_streamed`` returns before the run finishes, so the value is recorded as the
    stream is consumed rather than when the call returns."""
    agent = Agent(name="WeatherAgent", model=_FakeModel())

    streamed = Runner.run_streamed(agent, "What's the weather in London?")
    async for _ in streamed.stream_events():
        pass
    assert streamed.final_output == _ANSWER

    attrs = _root(list(exporter.get_finished_spans()))
    assert attrs["input.value"] == "What's the weather in London?"
    assert attrs["output.value"] == _ANSWER


def test_run_sync_records_input_and_output(exporter: InMemorySpanExporter) -> None:
    agent = Agent(name="WeatherAgent", model=_FakeModel())

    # run_sync drives the loop itself and asks the policy for one, which a synchronous
    # test does not otherwise have.
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        Runner.run_sync(agent, "What's the weather in London?")
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    attrs = _root(list(exporter.get_finished_spans()))
    assert attrs["input.value"] == "What's the weather in London?"
    assert attrs["output.value"] == _ANSWER


class _Weather(BaseModel):
    city: str
    celsius: int


async def test_structured_output_type_is_recorded_as_json(
    exporter: InMemorySpanExporter,
) -> None:
    """A run with an ``output_type`` finishes with a model, not the raw LLM text.

    This is one of the cases that makes inferring the value from a child LLM span wrong:
    the text below is JSON only because the agent asked for it, and the real result is
    the parsed object.
    """
    agent = Agent(
        name="WeatherAgent",
        model=_FakeModel('{"city":"London","celsius":21}'),
        output_type=_Weather,
    )

    result = await Runner.run(agent, "London?")
    assert isinstance(result.final_output, _Weather)

    attrs = _root(list(exporter.get_finished_spans()))
    assert attrs["output.mime_type"] == "application/json"
    assert json.loads(str(attrs["output.value"])) == {"city": "London", "celsius": 21}


async def test_llm_calling_input_guardrail_does_not_supply_the_root_output(
    exporter: InMemorySpanExporter,
) -> None:
    """The case that defeated inferring output from child LLM spans.

    An input guardrail runs concurrently with the agent's own turn and in the same trace,
    so whichever LLM span happened to finish last was a race. Here the guardrail runs an
    agent of its own; the root output must still be the agent's answer.
    """

    @input_guardrail
    async def guardrail(*args: Any, **kwargs: Any) -> GuardrailFunctionOutput:
        await Runner.run(Agent(name="Guard", model=_FakeModel("GUARDRAIL VERDICT")), "safe?")
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    agent = Agent(name="WeatherAgent", model=_FakeModel(), input_guardrails=[guardrail])

    result = await Runner.run(agent, "What's the weather in London?")
    assert result.final_output == _ANSWER

    attrs = _root(list(exporter.get_finished_spans()))
    assert attrs["output.value"] == _ANSWER
    assert "GUARDRAIL VERDICT" not in str(attrs["output.value"])


async def test_concurrent_runs_do_not_share_input_or_output(
    exporter: InMemorySpanExporter,
) -> None:
    """Each run's values live in its own context, so overlapping runs cannot mix."""
    first = Agent(name="First", model=_FakeModel("first answer"))
    second = Agent(name="Second", model=_FakeModel("second answer"))

    await asyncio.gather(
        Runner.run(first, "first question"),
        Runner.run(second, "second question"),
    )

    roots = _roots(list(exporter.get_finished_spans()))
    assert len(roots) == 2
    pairs = sorted((str(a["input.value"]), str(a["output.value"])) for a in roots)
    assert pairs == [
        ("first question", "first answer"),
        ("second question", "second answer"),
    ]


async def test_nested_agent_as_tool_run_does_not_overwrite_the_outer_output(
    exporter: InMemorySpanExporter,
) -> None:
    """An agent exposed as a tool starts its own run inside the outer one.

    Both runs reach ``on_agent_end``, and the inner one gets there first, so this is the
    case that would corrupt the root output if the value were shared rather than scoped
    to each run's own context.
    """
    inner = Agent(name="Inner", model=_FakeModel("inner answer"))
    outer = Agent(
        name="Outer",
        model=_FakeModel("outer answer", tool_name="inner_tool"),
        tools=[inner.as_tool(tool_name="inner_tool", tool_description="ask the inner agent")],
    )

    result = await Runner.run(outer, "outer question")
    assert result.final_output == "outer answer"

    attrs = _root(list(exporter.get_finished_spans()))
    assert attrs["input.value"] == "outer question"
    assert attrs["output.value"] == "outer answer"


async def test_a_user_managed_trace_over_several_runs_records_nothing(
    exporter: InMemorySpanExporter,
) -> None:
    """A trace the caller opened around several runs is not one run's boundary.

    The SDK lets a caller group runs under a trace of their own. That root span covers
    more than one run, so no single run's input or output describes it and none is
    reported -- which is the whole point of only recording what was actually observed.
    """
    first = Agent(name="First", model=_FakeModel("first answer"))
    second = Agent(name="Second", model=_FakeModel("second answer"))

    with trace("My workflow"):
        await Runner.run(first, "first question")
        await Runner.run(second, "second question")

    attrs = _root(list(exporter.get_finished_spans()))
    assert "input.value" not in attrs
    assert "output.value" not in attrs


async def test_caller_supplied_hooks_still_receive_every_callback(
    exporter: InMemorySpanExporter,
) -> None:
    """Composing our hook in must never swallow the caller's own callbacks."""
    seen: list[str] = []

    class _CallerHooks(RunHooks[Any]):
        async def on_agent_start(self, *args: Any, **kwargs: Any) -> None:
            seen.append("start")

        async def on_agent_end(self, *args: Any, **kwargs: Any) -> None:
            seen.append("end")

        async def on_tool_start(self, *args: Any, **kwargs: Any) -> None:
            seen.append("tool_start")

        async def on_tool_end(self, *args: Any, **kwargs: Any) -> None:
            seen.append("tool_end")

    @function_tool
    def ping() -> str:
        """Ping."""
        return "pong"

    agent = Agent(name="WeatherAgent", model=_FakeModel(tool_name="ping"), tools=[ping])
    await Runner.run(agent, "ping please", hooks=_CallerHooks())

    assert "start" in seen and "end" in seen
    assert "tool_start" in seen and "tool_end" in seen
    # And the output still reached the span.
    assert _root(list(exporter.get_finished_spans()))["output.value"] == _ANSWER


async def test_a_run_that_never_finishes_records_no_output(
    exporter: InMemorySpanExporter,
) -> None:
    """No final output means no output attribute, rather than an invented one."""

    @function_tool
    def ping() -> str:
        """Ping."""
        return "pong"

    agent = Agent(name="Looper", model=_FakeModel(tool_name="ping"), tools=[ping])
    with pytest.raises(MaxTurnsExceeded):
        await Runner.run(agent, "loop forever", max_turns=1)

    attrs = _root(list(exporter.get_finished_spans()))
    # The input is known before the run starts, so it is still reported.
    assert attrs["input.value"] == "loop forever"
    assert "output.value" not in attrs
    assert "output.mime_type" not in attrs


async def test_agent_spans_still_carry_no_inferred_input_or_output(
    exporter: InMemorySpanExporter,
) -> None:
    """Only the trace root gets these values; the guard against inference stands."""
    agent = Agent(name="WeatherAgent", model=_FakeModel())
    await Runner.run(agent, "What's the weather in London?")

    agent_spans = [
        _attrs(s)
        for s in exporter.get_finished_spans()
        if _attrs(s).get("agent.name") == "WeatherAgent"
    ]
    assert agent_spans, "expected an agent span"
    for attrs in agent_spans:
        assert "input.value" not in attrs
        assert "output.value" not in attrs


async def test_masking_redacts_the_recorded_values(
    masked_exporter: InMemorySpanExporter,
) -> None:
    """These go through the OITracer's spans, so TraceConfig applies as it does anywhere."""
    agent = Agent(name="WeatherAgent", model=_FakeModel())
    await Runner.run(agent, "What's the weather in London?")

    attrs = _root(list(masked_exporter.get_finished_spans()))
    assert attrs["input.value"] == REDACTED_VALUE
    assert attrs["output.value"] == REDACTED_VALUE


async def test_nothing_is_recorded_without_the_runner_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Proves the attributes come from the patch and not from somewhere else.

    Without this, a change that quietly stopped patching the runner would leave the other
    tests passing only if the values happened to arrive by some other route.
    """
    monkeypatch.setattr(
        "openinference.instrumentation.openai_agents._run_io.find_agent_runner_bindings",
        lambda: [],
    )
    exporter = _instrument()
    try:
        await Runner.run(Agent(name="WeatherAgent", model=_FakeModel()), "London?")
        attrs = _root(list(exporter.get_finished_spans()))
        assert "input.value" not in attrs
        assert "output.value" not in attrs
    finally:
        OpenAIAgentsInstrumentor().uninstrument()


async def test_no_spans_and_untouched_hooks_when_tracing_is_suppressed(
    exporter: InMemorySpanExporter,
) -> None:
    """Suppressed means inert, not just silent.

    No spans is the easy half. The half that matters here is that the caller's ``hooks``
    argument must reach the SDK exactly as they passed it: a user who suppressed tracing
    has asked the instrumentor to stay out of their call.
    """
    caller_hooks = RunHooks[Any]()
    agent = Agent(name="WeatherAgent", model=_FakeModel())

    with suppress_tracing():
        result = await Runner.run(agent, "What's the weather in London?", hooks=caller_hooks)

    assert result.final_output == _ANSWER
    assert not exporter.get_finished_spans()


@pytest.mark.parametrize("suppressed", [True, False], ids=["suppressed", "not-suppressed"])
def test_the_wrapper_only_substitutes_hooks_when_not_suppressed(suppressed: bool) -> None:
    """Directly checks what the SDK receives, which the span assertions cannot show."""
    seen: dict[str, Any] = {}

    def wrapped(*args: Any, **kwargs: Any) -> str:
        seen["hooks"] = kwargs.get("hooks")
        return "done"

    caller_hooks = RunHooks[Any]()
    wrapper = make_sync_run_wrapper()
    if suppressed:
        with suppress_tracing():
            wrapper(wrapped, None, (None, "input"), {"hooks": caller_hooks})
        assert seen["hooks"] is caller_hooks
    else:
        wrapper(wrapped, None, (None, "input"), {"hooks": caller_hooks})
        assert seen["hooks"] is not caller_hooks
        assert type(seen["hooks"]).__name__ == "_OpenInferenceRunHooks"


# --- value shapes -------------------------------------------------------------------


def test_a_string_input_is_recorded_as_text() -> None:
    assert input_attributes("hello") == {
        "input.value": "hello",
        "input.mime_type": "text/plain",
    }


def test_a_resumed_runs_state_object_is_not_recorded_as_input() -> None:
    """A resumed run is handed a RunState, which is bookkeeping rather than input.

    The gate is on the type, so this never reaches the shared serializer -- which would
    otherwise happily record ``str(run_state)``.
    """

    class _RunState:
        pass

    assert input_attributes(_RunState()) is None


def test_a_string_output_is_recorded_as_text() -> None:
    assert output_attributes("done") == {
        "output.value": "done",
        "output.mime_type": "text/plain",
    }


def test_a_pydantic_output_is_recorded_as_json() -> None:
    """Serialization is the shared helper's job; this pins the shape it produces."""
    recorded = output_attributes(_Weather(city="London", celsius=21))
    assert recorded is not None
    assert recorded["output.mime_type"] == "application/json"
    assert json.loads(str(recorded["output.value"])) == {"city": "London", "celsius": 21}


def test_an_output_of_an_unexpected_type_still_records_something() -> None:
    class _Opaque:
        def __repr__(self) -> str:
            return "<opaque>"

    recorded = output_attributes(_Opaque())
    assert recorded is not None
    assert "<opaque>" in str(recorded["output.value"])


def test_no_output_is_recorded_for_none() -> None:
    assert output_attributes(None) is None


def test_holder_distinguishes_unset_from_empty() -> None:
    assert RunIO().has_input is False
    assert RunIO().has_output is False
    assert RunIO(input="").has_input is True
    assert RunIO(output="").has_output is True


# --- hooks composition --------------------------------------------------------------


def test_every_hook_on_the_installed_sdk_is_delegated() -> None:
    """The SDK has grown hooks over releases; none may be left unforwarded."""
    from agents.lifecycle import RunHooksBase

    hooks_class = run_hooks_class()
    assert hooks_class is not None
    expected = {
        name
        for name, member in vars(RunHooksBase).items()
        if name.startswith("on_") and callable(member)
    }
    assert expected, "no hooks found on RunHooksBase"
    assert expected <= set(vars(hooks_class))


async def test_a_delegated_hook_reaches_the_callers_implementation() -> None:
    hooks_class = run_hooks_class()
    assert hooks_class is not None
    seen: list[Any] = []

    class _Caller(RunHooks[Any]):
        async def on_handoff(self, *args: Any, **kwargs: Any) -> None:
            seen.append(args)

    composed = hooks_class(_Caller())
    await composed.on_handoff("ctx", "from", "to")
    assert seen == [("ctx", "from", "to")]


async def test_recording_the_output_does_not_depend_on_the_context_argument() -> None:
    """Only the output's position is relied on; its first argument changed type between
    SDK releases and is never read."""
    hooks_class = run_hooks_class()
    assert hooks_class is not None
    holder = RunIO()

    token = _run_io.set(holder)
    try:
        await hooks_class().on_agent_end(object(), object(), "the answer")
    finally:
        _run_io.reset(token)
    assert holder.output == "the answer"


async def test_a_caller_hook_that_raises_is_not_swallowed() -> None:
    """Our wrapper must not change the SDK's error behaviour."""
    hooks_class = run_hooks_class()
    assert hooks_class is not None

    class _Boom(RunHooks[Any]):
        async def on_agent_end(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await hooks_class(_Boom()).on_agent_end(object(), object(), "x")


def test_hooks_of_an_unexpected_type_are_left_alone() -> None:
    """The SDK validates this argument and raises on a bad one. Wrapping it would hide
    that behind an instance that passes validation and then fails mid-run."""
    from openinference.instrumentation.openai_agents._run_io import _prepare

    sentinel = object()
    holder, kwargs = _prepare((None, "input"), {"hooks": sentinel})
    assert holder is not None
    assert kwargs["hooks"] is sentinel


def test_absent_hooks_are_composed_over_the_sdk_default() -> None:
    from openinference.instrumentation.openai_agents._run_io import _prepare

    holder, kwargs = _prepare((None, "input"), {})
    assert holder is not None
    assert holder.input == "input"
    assert type(kwargs["hooks"]).__name__ == "_OpenInferenceRunHooks"


# --- patching -----------------------------------------------------------------------


def test_patch_targets_the_real_runner_and_is_reversible() -> None:
    """Guards against the SDK moving or renaming these entry points."""
    patched = _patch_agent_runner()
    assert patched, "no runner entry point could be patched"
    assert {attribute for _, attribute, _ in patched} == {"run", "run_sync", "run_streamed"}
    try:
        for owner, attribute, original in patched:
            assert owner.__dict__[attribute] is not original
    finally:
        for owner, attribute, original in patched:
            setattr(owner, attribute, original)
    for owner, attribute, original in patched:
        assert owner.__dict__[attribute] is original


def test_uninstrument_restores_every_patched_entry_point() -> None:
    def snapshot() -> dict[str, Any]:
        return {
            f"{owner!r}.{attribute}": owner.__dict__[attribute]
            for owner, attribute in find_agent_runner_bindings()
        }

    before = snapshot()
    assert before, "no runner entry point is present in this SDK version"
    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument()
    during = snapshot()
    assert all(during[key] is not before[key] for key in before)
    instrumentor.uninstrument()
    assert snapshot() == before
