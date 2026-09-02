"""Span status on failure paths (regression tests for #3415).

On ADK >= 1.32 the ``execute_tool`` span is opened by ADK and enriched by our
``_TraceToolCall`` wrapper from a ``finally`` block *while the span is still
open*. Likewise ``call_llm`` is enriched by ``_TraceCallLlm`` for every streamed
response, before the stream may raise. OpenTelemetry treats ``OK`` as final, so
stamping ``OK`` in those wrappers locks out the ``ERROR`` that the span's own
exit handler would otherwise record. These tests drive both paths with a stub
model and a raising tool and assert the failed spans finish as ``ERROR``.
"""

from collections import defaultdict
from secrets import token_hex
from typing import Any, AsyncGenerator

import pytest
from google.adk import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode


def _tool_call_response(tool_name: str, **args: Any) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        id=f"call_{token_hex(4)}", name=tool_name, args=args
                    )
                )
            ],
        )
    )


def _text_response(text: str, partial: bool = False) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)]),
        partial=partial,
    )


async def _run(agent: Agent) -> str:
    """Runs one user turn and returns the app name (part of the root span name)."""
    app_name = f"app{token_hex(4)}"
    user_id, session_id = token_hex(4), token_hex(4)
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    await runner.session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text="hi")]),
    ):
        ...
    return app_name


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, list[ReadableSpan]]:
    spans: dict[str, list[ReadableSpan]] = defaultdict(list)
    for span in exporter.get_finished_spans():
        spans[span.name].append(span)
    return spans


def _has_exception_event(span: ReadableSpan) -> bool:
    return any(event.name == "exception" for event in span.events)


async def test_failing_tool_span_status_is_error(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def explode(city: str) -> dict[str, str]:
        """Always fails."""
        raise RuntimeError("tool exploded")

    class _StubLlm(BaseLlm):
        model: str = "stub"

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield _tool_call_response("explode", city="New York")

    agent = Agent(name=f"a{token_hex(4)}", model=_StubLlm(), tools=[explode])

    with pytest.raises(RuntimeError, match="tool exploded"):
        await _run(agent)

    spans = _spans_by_name(in_memory_span_exporter)
    (tool_span,) = spans["execute_tool explode"]
    assert tool_span.status.status_code is StatusCode.ERROR
    assert _has_exception_event(tool_span)
    # The successful LLM turn that requested the tool call stays OK. (Older ADK
    # versions do not close the `call_llm` span when a tool raises, so it may
    # not be exported at all there.)
    for call_llm_span in spans["call_llm"]:
        assert call_llm_span.status.status_code is StatusCode.OK
    # Enclosing spans also report the failure.
    (agent_span,) = spans[f"agent_run [{agent.name}]"]
    assert agent_span.status.status_code is StatusCode.ERROR
    (invocation_span,) = [
        s for name, ss in spans.items() if name.startswith("invocation") for s in ss
    ]
    assert invocation_span.status.status_code is StatusCode.ERROR


async def test_llm_stream_error_after_partial_response_status_is_error(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    class _StubLlm(BaseLlm):
        model: str = "stub"

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield _text_response("partial...", partial=True)
            raise RuntimeError("stream broke")

    agent = Agent(name=f"a{token_hex(4)}", model=_StubLlm())

    with pytest.raises(RuntimeError, match="stream broke"):
        await _run(agent)

    spans = _spans_by_name(in_memory_span_exporter)
    (call_llm_span,) = spans["call_llm"]
    assert call_llm_span.status.status_code is StatusCode.ERROR
    assert _has_exception_event(call_llm_span)
    (agent_span,) = spans[f"agent_run [{agent.name}]"]
    assert agent_span.status.status_code is StatusCode.ERROR
    (invocation_span,) = [
        s for name, ss in spans.items() if name.startswith("invocation") for s in ss
    ]
    assert invocation_span.status.status_code is StatusCode.ERROR


async def test_llm_response_with_error_code_status_is_error(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """ADK's structured `error_code` (e.g. a blocked response) marks the span ERROR."""

    class _StubLlm(BaseLlm):
        model: str = "stub"

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield LlmResponse(error_code="SAFETY", error_message="blocked by safety filter")

    agent = Agent(name=f"a{token_hex(4)}", model=_StubLlm())
    await _run(agent)

    spans = _spans_by_name(in_memory_span_exporter)
    (call_llm_span,) = spans["call_llm"]
    assert call_llm_span.status.status_code is StatusCode.ERROR
    assert call_llm_span.status.description == "blocked by safety filter"


async def test_successful_tool_and_llm_spans_status_is_ok(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Happy path keeps an explicit OK (not merely UNSET) on tool and LLM spans."""

    def get_weather(city: str) -> dict[str, str]:
        """Returns weather."""
        return {"status": "success", "report": f"Sunny in {city}."}

    class _StubLlm(BaseLlm):
        model: str = "stub"
        _index: int = 0

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            responses = [
                _tool_call_response("get_weather", city="New York"),
                _text_response("Sunny in New York."),
            ]
            response = responses[min(self._index, len(responses) - 1)]
            object.__setattr__(self, "_index", self._index + 1)
            yield response

    agent = Agent(name=f"a{token_hex(4)}", model=_StubLlm(), tools=[get_weather])
    app_name = await _run(agent)

    spans = _spans_by_name(in_memory_span_exporter)
    (tool_span,) = spans["execute_tool get_weather"]
    assert tool_span.status.status_code is StatusCode.OK
    assert not _has_exception_event(tool_span)
    call_llm_spans = spans["call_llm"]
    assert len(call_llm_spans) == 2
    for span in call_llm_spans:
        assert span.status.status_code is StatusCode.OK
    for name in (f"invocation [{app_name}]", f"agent_run [{agent.name}]"):
        (span,) = spans[name]
        assert span.status.status_code is StatusCode.OK, name
