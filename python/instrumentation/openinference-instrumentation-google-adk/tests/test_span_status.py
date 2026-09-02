"""Span status derived from ADK's structured response fields (follow-up to #3415).

The raised-exception paths (a tool that raises, a stream that raises after a
partial chunk) are covered in ``test_instrumentor.py``. These tests drive the
whole runner with a stub model to pin down what those tests do not:

- a response that ADK flags with an ``error_code`` ends ``call_llm`` as ERROR,
- SSE streaming with partial chunks still ends ``call_llm`` as OK once the final
  chunk arrives (partial chunks must not decide the status either way),
- the happy path keeps an explicit OK on every span, not merely UNSET, which
  ``Status.is_ok`` would also accept.
"""

from collections import defaultdict
from secrets import token_hex
from typing import Any, AsyncGenerator, Optional

from google.adk import Agent
from google.adk.agents.run_config import RunConfig, StreamingMode
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


async def _run(agent: Agent, run_config: Optional[RunConfig] = None) -> str:
    """Runs one user turn and returns the app name (part of the root span name)."""
    app_name = f"app{token_hex(4)}"
    user_id, session_id = token_hex(4), token_hex(4)
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    await runner.session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id
    )
    kwargs: dict[str, Any] = {"run_config": run_config} if run_config is not None else {}
    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text="hi")]),
        **kwargs,
    ):
        ...
    return app_name


def _spans_by_name(exporter: InMemorySpanExporter) -> dict[str, list[ReadableSpan]]:
    spans: dict[str, list[ReadableSpan]] = defaultdict(list)
    for span in exporter.get_finished_spans():
        spans[span.name].append(span)
    return spans


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


async def test_llm_response_with_error_code_and_no_message_uses_code_as_description(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    class _StubLlm(BaseLlm):
        model: str = "stub"

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            yield LlmResponse(error_code="RECITATION")

    agent = Agent(name=f"a{token_hex(4)}", model=_StubLlm())
    await _run(agent)

    spans = _spans_by_name(in_memory_span_exporter)
    (call_llm_span,) = spans["call_llm"]
    assert call_llm_span.status.status_code is StatusCode.ERROR
    assert call_llm_span.status.description == "RECITATION"


async def test_sse_stream_with_partial_chunks_status_is_ok(
    instrument: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Partial chunks must not set the status; the final chunk stamps OK."""

    class _StubLlm(BaseLlm):
        model: str = "stub"

        async def generate_content_async(
            self, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse, None]:
            assert stream, "expected SSE streaming to request a streamed response"
            yield _text_response("Sunny ", partial=True)
            yield _text_response("in New York.", partial=True)
            yield _text_response("Sunny in New York.")

    agent = Agent(name=f"a{token_hex(4)}", model=_StubLlm())
    app_name = await _run(agent, RunConfig(streaming_mode=StreamingMode.SSE))

    spans = _spans_by_name(in_memory_span_exporter)
    (call_llm_span,) = spans["call_llm"]
    assert call_llm_span.status.status_code is StatusCode.OK
    for name in (f"invocation [{app_name}]", f"agent_run [{agent.name}]"):
        (span,) = spans[name]
        assert span.status.status_code is StatusCode.OK, name


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
    assert not any(event.name == "exception" for event in tool_span.events)
    call_llm_spans = spans["call_llm"]
    assert len(call_llm_spans) == 2
    for span in call_llm_spans:
        assert span.status.status_code is StatusCode.OK
    for name in (f"invocation [{app_name}]", f"agent_run [{agent.name}]"):
        (span,) = spans[name]
        assert span.status.status_code is StatusCode.OK, name
