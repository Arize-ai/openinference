"""Trace the shared execution boundary of Agno external framework adapters."""

import asyncio
import logging
from contextlib import contextmanager
from time import time_ns
from typing import Any, AsyncIterator, Callable, Iterator, Mapping, Optional, Tuple

from openinference.semconv.trace import (
    MessageAttributes as Message,
)
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
)
from openinference.semconv.trace import (
    SpanAttributes as Span,
)
from openinference.semconv.trace import (
    ToolCallAttributes as ToolCall,
)
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.context import Context

from agno.run.agent import RunOutput
from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.agno.utils import (
    _AGNO_PARENT_NODE_CONTEXT_KEY,
    _generate_node_id,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# The sync stream creates its async iterator inside a worker thread. Carry only
# tracing context through its kwargs; remove it before invoking the adapter.
_CONTEXT_KEY = "_openinference_external_context"


@contextmanager
def _ignore_instrumentation_errors() -> Iterator[None]:
    try:
        yield
    except Exception:
        logger.exception("Failed to record external agent tracing data")


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _content(span: trace_api.Span, value: Any, *, output: bool) -> None:
    if value is None:
        return
    span.set_attribute(
        Span.OUTPUT_VALUE if output else Span.INPUT_VALUE,
        value if isinstance(value, str) else safe_json_dumps(value),
    )
    span.set_attribute(
        Span.OUTPUT_MIME_TYPE if output else Span.INPUT_MIME_TYPE,
        "text/plain" if isinstance(value, str) else "application/json",
    )


class _RunTrace:
    def __init__(
        self,
        tracer: trace_api.Tracer,
        agent: Any,
        input: Any,
        kwargs: Mapping[str, Any],
        capture_tools: bool,
    ) -> None:
        self.tracer = tracer
        self.capture_tools = capture_tools
        self.tools: dict[str, trace_api.Span] = {}
        # Keep completed parents available: buffered child events can arrive later.
        self.tool_contexts: dict[str, Context] = {}
        self.status: Optional[str] = None
        node_id = _generate_node_id()
        attrs: dict[str, Any] = {
            Span.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
            Span.GRAPH_NODE_ID: node_id,
            "agno.agent.id": agent.get_id(),
            "agno.agent.framework": agent.framework,
        }
        if agent.name:
            attrs[Span.AGENT_NAME] = agent.name
            attrs[Span.GRAPH_NODE_NAME] = agent.name
        if parent := context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY):
            attrs[Span.GRAPH_NODE_PARENT_ID] = parent
        for key in ("run_id", "session_id", "user_id"):
            if value := kwargs.get(key):
                attrs[f"agno.{key.replace('_', '.')}"] = value
                if key == "session_id":
                    attrs[Span.SESSION_ID] = value
                elif key == "user_id":
                    attrs[Span.USER_ID] = value
        if isinstance(model := getattr(agent, "model", None), str):
            attrs[Span.LLM_MODEL_NAME] = model
        self.span = tracer.start_span(f"{type(agent).__name__}.run", attributes=attrs)
        self.context = context_api.set_value(
            _AGNO_PARENT_NODE_CONTEXT_KEY, node_id, trace_api.set_span_in_context(self.span)
        )
        with _ignore_instrumentation_errors():
            _content(self.span, input, output=False)

    def observe(self, value: Any) -> None:
        with _ignore_instrumentation_errors():
            for field in ("run_id", "session_id", "user_id"):
                if identity := getattr(value, field, None):
                    self.span.set_attribute(f"agno.{field.replace('_', '.')}", identity)
                    if field == "session_id" and not context_api.get_value(Span.SESSION_ID):
                        self.span.set_attribute(Span.SESSION_ID, identity)
            event = _value(getattr(value, "event", ""))
            if event == "ToolCallStarted":
                self._start_tool(value.tool)
            elif event == "ToolCallCompleted":
                self._end_tool(value.tool)
            elif event == "RunError":
                self.set_status("error")
                _content(self.span, value.content, output=True)
            elif event == "RunCancelled":
                self.set_status("cancelled")
            elif event == "RunCompleted" or isinstance(value, RunOutput):
                self.set_status(_value(getattr(value, "status", "completed")))
                _content(self.span, getattr(value, "content", None), output=True)
                self._metrics(getattr(value, "metrics", None))
                self._tool_records(getattr(value, "tools", None))
            elif (getattr(value, "warning", None) or {}).get("type") == "retry":
                self._close_tools("incomplete")

    def set_status(self, status: str) -> None:
        status = status.lower()
        self.status = status
        self.span.set_attribute("agno.run.status", status)
        if status == "error":
            self.span.set_status(trace_api.StatusCode.ERROR, "External agent run failed")
        elif status == "completed":
            self.span.set_status(trace_api.StatusCode.OK)

    def _metrics(self, metrics: Any) -> None:
        for field, attribute in (
            ("input_tokens", Span.LLM_TOKEN_COUNT_PROMPT),
            ("output_tokens", Span.LLM_TOKEN_COUNT_COMPLETION),
            ("total_tokens", Span.LLM_TOKEN_COUNT_TOTAL),
            ("cache_read_tokens", Span.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ),
            ("cache_write_tokens", Span.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE),
            ("reasoning_tokens", Span.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING),
            ("cost", Span.LLM_COST_TOTAL),
        ):
            if (value := getattr(metrics, field, None)) is not None:
                self.span.set_attribute(attribute, value)

    def _tool_records(self, tools: Any) -> None:
        # Non-streamed runs expose completed records, not tool lifecycle times.
        # Preserve them as messages rather than inventing execution durations.
        for index, tool in enumerate(tools or []):
            call = f"{Span.LLM_OUTPUT_MESSAGES}.0.{Message.MESSAGE_TOOL_CALLS}.{index}"
            self.span.set_attribute(
                f"{Span.LLM_OUTPUT_MESSAGES}.0.{Message.MESSAGE_ROLE}", "assistant"
            )
            self.span.set_attribute(
                f"{call}.{ToolCall.TOOL_CALL_FUNCTION_NAME}", tool.tool_name or "tool"
            )
            if tool.tool_call_id:
                self.span.set_attribute(f"{call}.{ToolCall.TOOL_CALL_ID}", tool.tool_call_id)
            self.span.set_attribute(
                f"{call}.{ToolCall.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                safe_json_dumps(tool.tool_args),
            )
            result = f"{Span.LLM_OUTPUT_MESSAGES}.{index + 1}"
            self.span.set_attribute(f"{result}.{Message.MESSAGE_ROLE}", "tool")
            if tool.tool_call_id:
                self.span.set_attribute(
                    f"{result}.{Message.MESSAGE_TOOL_CALL_ID}", tool.tool_call_id
                )
            if tool.result is not None:
                self.span.set_attribute(f"{result}.{Message.MESSAGE_CONTENT}", str(tool.result))

    def _start_tool(self, tool: Any) -> None:
        if not self.capture_tools or tool is None or not tool.tool_call_id:
            return
        if tool.tool_call_id in self.tools:
            return
        parent_id = getattr(tool, "parent_tool_call_id", None)
        attributes = {
            Span.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
            Span.TOOL_NAME: tool.tool_name or "tool",
            ToolCall.TOOL_CALL_ID: tool.tool_call_id,
        }
        if parent_id:
            attributes["agno.tool.parent_call_id"] = parent_id
        span = self.tracer.start_span(
            tool.tool_name or "tool",
            context=self.tool_contexts.get(parent_id, self.context) if parent_id else self.context,
            attributes=attributes,
        )
        self.tools[tool.tool_call_id] = span
        self.tool_contexts[tool.tool_call_id] = trace_api.set_span_in_context(span, self.context)
        _content(span, tool.tool_args, output=False)

    def _end_tool(self, tool: Any) -> None:
        if tool is None:
            return
        span = self.tools.pop(tool.tool_call_id, None)
        if span is None:
            # A completion without a start does not establish a duration.
            return
        try:
            _content(span, tool.result, output=True)
            span.set_status(
                trace_api.StatusCode.ERROR if tool.tool_call_error else trace_api.StatusCode.OK
            )
        finally:
            span.end()

    def _close_tools(self, status: str) -> None:
        for span in self.tools.values():
            with _ignore_instrumentation_errors():
                span.set_attribute("agno.tool.status", status)
                if status == "error":
                    span.set_status(trace_api.StatusCode.ERROR)
                span.end()
        self.tools.clear()
        self.tool_contexts.clear()

    def finish(self) -> None:
        try:
            with _ignore_instrumentation_errors():
                if self.status is None:
                    self.set_status("cancelled")
                self._close_tools(self.status or "incomplete")
        finally:
            with _ignore_instrumentation_errors():
                self.span.end(end_time=time_ns())


class _ExternalRunWrapper:
    def __init__(self, tracer: trace_api.Tracer, capture_tools: bool = True) -> None:
        self.tracer = tracer
        self.capture_tools = capture_tools

    def _start(
        self, instance: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> Optional[_RunTrace]:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return None
        with _ignore_instrumentation_errors():
            return _RunTrace(
                self.tracer,
                instance,
                args[0] if args else kwargs.get("input"),
                kwargs,
                self.capture_tools,
            )
        return None

    def run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        return wrapped(*args, **{**kwargs, _CONTEXT_KEY: context_api.get_current()})

    def arun(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Capture before run_coroutine_sync can move this coroutine to a worker.
        parent = context_api.get_current()

        async def execute() -> Any:
            token = context_api.attach(parent)
            run = None
            try:
                run = self._start(instance, args, kwargs)
                if run is None:
                    return await wrapped(*args, **kwargs)
                with trace_api.use_span(
                    run.span,
                    end_on_exit=False,
                    record_exception=False,
                    set_status_on_exception=False,
                ):
                    node_token = context_api.attach(run.context)
                    try:
                        result = await wrapped(*args, **kwargs)
                        run.observe(result)
                        return result
                    finally:
                        context_api.detach(node_token)
            except BaseException as exc:
                if run:
                    with _ignore_instrumentation_errors():
                        run.set_status(
                            "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
                        )
                raise
            finally:
                if run:
                    run.finish()
                context_api.detach(token)

        return execute()

    def arun_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        options = dict(kwargs)
        parent: Context = options.pop(_CONTEXT_KEY, context_api.get_current())

        async def iterate() -> AsyncIterator[Any]:
            token = context_api.attach(parent)
            try:
                run = self._start(instance, args, options)
            finally:
                context_api.detach(token)
            # Request the final RunOutput for metadata, without changing what the
            # caller receives. Instrumentation must not force storage or streaming.
            include_output = options.get("yield_run_output", False)
            call_options = {**options, "yield_run_output": True} if run else options
            iterator = wrapped(*args, **call_options)
            active_context = run.context if run else parent
            try:
                while True:
                    token = context_api.attach(active_context)
                    try:
                        value = await iterator.__anext__()
                        if run:
                            run.observe(value)
                    except StopAsyncIteration:
                        break
                    finally:
                        # Never leave the run current while yielding to user code.
                        context_api.detach(token)
                    if run and not include_output and isinstance(value, RunOutput):
                        continue
                    yield value
            except BaseException as exc:
                if run and run.status is None:
                    with _ignore_instrumentation_errors():
                        run.set_status(
                            "cancelled"
                            if isinstance(exc, (GeneratorExit, asyncio.CancelledError))
                            else "error"
                        )
                raise
            finally:
                token = context_api.attach(active_context)
                try:
                    await iterator.aclose()
                finally:
                    if run:
                        run.finish()
                    context_api.detach(token)

        return iterate()
