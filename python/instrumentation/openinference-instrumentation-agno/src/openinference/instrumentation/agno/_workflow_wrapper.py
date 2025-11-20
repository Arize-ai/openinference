from inspect import signature
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Mapping,
    OrderedDict,
    Tuple,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation.agno.utils import (
    _AGNO_PARENT_NODE_CONTEXT_KEY,
    _flatten,
    _generate_node_id,
)
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    arguments = bound_args.arguments
    arguments = OrderedDict(
        {key: value for key, value in arguments.items() if value is not None and value != {}}
    )
    return arguments


def _get_input_from_args(arguments: Mapping[str, Any]) -> str:
    """Extract input message from workflow/step arguments."""
    step_input = arguments.get("step_input")
    if step_input is not None and hasattr(step_input, "input"):
        input_value = step_input.input
        if input_value is not None:
            if isinstance(input_value, str):
                return input_value
            elif isinstance(input_value, list):
                return "\n".join(str(item) for item in input_value)
            elif hasattr(input_value, "model_dump_json"):
                return input_value.model_dump_json(indent=2, exclude_none=True)
            elif isinstance(input_value, dict):
                import json

                return json.dumps(input_value, indent=2, ensure_ascii=False)
            else:
                return str(input_value)

    for key in ["input", "message", "messages"]:
        if value := arguments.get(key):
            if isinstance(value, str):
                return value
            elif isinstance(value, list):
                return "\n".join(str(item) for item in value)
            else:
                return str(value)
    return ""


def _extract_output(response: Any) -> str:
    """Extract output from workflow/step response."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    if hasattr(response, "content"):
        return str(response.content)
    else:
        return str(response.model_dump_json())
    return ""


def _workflow_attributes(instance: Any) -> Iterator[Tuple[str, AttributeValue]]:
    """Extract attributes from workflow instance."""
    if hasattr(instance, "name") and instance.name:
        yield GRAPH_NODE_NAME, instance.name

    if hasattr(instance, "description") and instance.description:
        yield "agno.workflow.description", instance.description

    if hasattr(instance, "steps") and instance.steps:
        yield "agno.workflow.steps_count", len(instance.steps)
        step_names = []
        for step in instance.steps:
            if hasattr(step, "name") and step.name:
                step_names.append(step.name)
        if step_names:
            yield "agno.workflow.steps", step_names


def _step_attributes(instance: Any) -> Iterator[Tuple[str, AttributeValue]]:
    """Extract attributes from step instance."""
    # Get parent from execution context
    context_parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)

    if hasattr(instance, "name") and instance.name:
        yield GRAPH_NODE_NAME, instance.name

    if context_parent_id:
        yield GRAPH_NODE_PARENT_ID, context_parent_id

    # Identify if step has a team or agent
    if hasattr(instance, "team") and instance.team:
        yield "agno.step.type", "team"
        if hasattr(instance.team, "name"):
            yield "agno.step.team_name", instance.team.name
    elif hasattr(instance, "agent") and instance.agent:
        yield "agno.step.type", "agent"
        if hasattr(instance.agent, "name"):
            yield "agno.step.agent_name", instance.agent.name


def _setup_workflow_context(node_id: str) -> Any:
    """Set up context for workflow to propagate to children."""
    workflow_ctx = context_api.set_value(_AGNO_PARENT_NODE_CONTEXT_KEY, node_id)
    return context_api.attach(workflow_ctx)


def _setup_step_context(node_id: str) -> Any:
    """Set up context for step to propagate to children."""
    step_ctx = context_api.set_value(_AGNO_PARENT_NODE_CONTEXT_KEY, node_id)
    return context_api.attach(step_ctx)


class _WorkflowWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        workflow_name = getattr(instance, "name", "Workflow").replace(" ", "_").replace("-", "_")
        # Use the actual method name for consistency
        method_name = getattr(wrapped, "__name__", "run")
        span_name = f"{workflow_name}.{method_name}"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        # Bind arguments to extract input
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_from_args(arguments),
                        **dict(_workflow_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        workflow_token = None
        result = None
        try:
            # Execute inside span context so children can find parent (like _runs_wrapper.py:253-255)
            with trace_api.use_span(span, end_on_exit=False):
                workflow_token = _setup_workflow_context(node_id)
                result = wrapped(*args, **kwargs)

                # Check if result is an iterator (streaming)
                is_streaming = hasattr(result, "__iter__") and not isinstance(result, (str, bytes))

                if is_streaming:
                    # For streaming, keep token attached and handle in stream continuation
                    return self._run_stream_continue(result, span, workflow_token)

                # Non-streaming mode - detach token immediately while still in span context
                if workflow_token:
                    try:
                        context_api.detach(workflow_token)
                        workflow_token = None
                    except Exception:
                        pass

            # Set output and return (outside use_span but token already detached)
            span.set_status(trace_api.StatusCode.OK)
            output = _extract_output(result)
            if output:
                span.set_attribute(OUTPUT_VALUE, output)
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

            return result

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            # Cleanup for non-streaming (streaming handles its own)
            if result is not None:
                is_streaming = hasattr(result, "__iter__") and not isinstance(result, (str, bytes))
            else:
                is_streaming = False

            if not is_streaming:
                if workflow_token:
                    try:
                        context_api.detach(workflow_token)
                    except Exception:
                        pass
                span.end()

    def _run_stream_continue(
        self,
        iterator: Any,
        span: Any,
        workflow_token: Any,
    ) -> Any:
        """Continue streaming workflow with existing span"""
        accumulated_output = []
        try:
            with trace_api.use_span(span, end_on_exit=False):
                try:
                    for response in iterator:
                        if hasattr(response, "content") and response.content:
                            accumulated_output.append(str(response.content))
                        yield response
                finally:
                    if workflow_token:
                        try:
                            context_api.detach(workflow_token)
                            workflow_token = None
                        except Exception:
                            pass

            span.set_status(trace_api.StatusCode.OK)
            if accumulated_output:
                span.set_attribute(OUTPUT_VALUE, "\n".join(accumulated_output))
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            if workflow_token:
                try:
                    context_api.detach(workflow_token)
                except Exception:
                    pass
            span.end()

    def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        workflow_name = getattr(instance, "name", "Workflow").replace(" ", "_").replace("-", "_")
        # Use the actual method name for consistency
        method_name = getattr(wrapped, "__name__", "arun")
        span_name = f"{workflow_name}.{method_name}"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        # Bind arguments to extract input
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_from_args(arguments),
                        **dict(_workflow_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        # Setup context and call wrapped to detect streaming
        workflow_token = None
        with trace_api.use_span(span, end_on_exit=False):
            workflow_token = _setup_workflow_context(node_id)
            result = wrapped(*args, **kwargs)

        # Check if result is an async iterator (streaming)
        if hasattr(result, "__aiter__"):
            # Streaming mode - return async generator that continues with this span
            return self._arun_stream_continue(result, span, workflow_token)

        # Non-streaming mode - return coroutine that handles it
        async def non_stream_wrapper():
            nonlocal workflow_token
            try:
                # Await the coroutine inside span context
                with trace_api.use_span(span, end_on_exit=False):
                    response = await result

                    # Detach token immediately after execution, while still in span context
                    if workflow_token:
                        try:
                            context_api.detach(workflow_token)
                            workflow_token = None
                        except Exception:
                            pass

                span.set_status(trace_api.StatusCode.OK)
                output = _extract_output(response)
                if output:
                    span.set_attribute(OUTPUT_VALUE, output)
                    span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

                return response

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise

            finally:
                if workflow_token:
                    try:
                        context_api.detach(workflow_token)
                    except Exception:
                        pass
                span.end()

        return non_stream_wrapper()

    async def _arun_stream_continue(
        self,
        async_iter: Any,
        span: Any,
        workflow_token: Any,
    ) -> Any:
        """Continue streaming async workflow with existing span"""
        accumulated_output = []
        try:
            with trace_api.use_span(span, end_on_exit=False):
                try:
                    async for response in async_iter:
                        if hasattr(response, "content") and response.content:
                            accumulated_output.append(str(response.content))
                        yield response
                finally:
                    if workflow_token:
                        try:
                            context_api.detach(workflow_token)
                            workflow_token = None
                        except Exception:
                            pass

            span.set_status(trace_api.StatusCode.OK)
            if accumulated_output:
                span.set_attribute(OUTPUT_VALUE, "\n".join(accumulated_output))
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            if workflow_token:
                try:
                    context_api.detach(workflow_token)
                except Exception:
                    pass
            span.end()


class _StepWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        step_name = getattr(instance, "name", "Step").replace(" ", "_").replace("-", "_")
        # Detect if this is execute_stream or execute by checking the wrapped method name
        method_name = getattr(wrapped, "__name__", "execute")
        span_name = f"{step_name}.{method_name}"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        # Get parent node ID from workflow context
        parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)

        # Bind arguments to extract input
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **({GRAPH_NODE_PARENT_ID: parent_id} if parent_id else {}),
                        INPUT_VALUE: _get_input_from_args(arguments),
                        **dict(_step_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        step_token = None
        result = None
        try:
            with trace_api.use_span(span, end_on_exit=False):
                step_token = _setup_step_context(node_id)
                result = wrapped(*args, **kwargs)

                # Check if result is an iterator (streaming)
                is_streaming = hasattr(result, "__iter__") and not isinstance(result, (str, bytes))

                if is_streaming:
                    # For streaming, keep token attached and handle in stream continuation
                    return self._run_stream_continue(result, span, step_token)

                # Non-streaming mode - detach token immediately while still in span context
                if step_token:
                    try:
                        context_api.detach(step_token)
                        step_token = None
                    except Exception:
                        pass

            # Set output and return (outside use_span but token already detached)
            span.set_status(trace_api.StatusCode.OK)
            output = _extract_output(result)
            if output:
                span.set_attribute(OUTPUT_VALUE, output)
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

            return result

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            # Cleanup for non-streaming (streaming handles its own)
            if result is not None:
                is_streaming = hasattr(result, "__iter__") and not isinstance(result, (str, bytes))
            else:
                is_streaming = False

            if not is_streaming:
                if step_token:
                    try:
                        context_api.detach(step_token)
                    except Exception:
                        pass
                span.end()

    def _run_stream_continue(
        self,
        iterator: Any,
        span: Any,
        step_token: Any,
    ) -> Any:
        """Continue streaming step with existing span"""
        accumulated_output = []
        try:
            with trace_api.use_span(span, end_on_exit=False):
                try:
                    for response in iterator:
                        if hasattr(response, "content") and response.content:
                            accumulated_output.append(str(response.content))
                        yield response
                finally:
                    if step_token:
                        try:
                            context_api.detach(step_token)
                            step_token = None
                        except Exception:
                            pass

            span.set_status(trace_api.StatusCode.OK)
            if accumulated_output:
                span.set_attribute(OUTPUT_VALUE, "\n".join(accumulated_output))
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            if step_token:
                try:
                    context_api.detach(step_token)
                except Exception:
                    pass
            span.end()

    def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        step_name = getattr(instance, "name", "Step").replace(" ", "_").replace("-", "_")
        # Detect if this is aexecute_stream or aexecute by checking the wrapped method name
        method_name = getattr(wrapped, "__name__", "aexecute")
        span_name = f"{step_name}.{method_name}"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        # Get parent node ID from workflow context
        parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)

        # Bind arguments to extract input
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **({GRAPH_NODE_PARENT_ID: parent_id} if parent_id else {}),
                        INPUT_VALUE: _get_input_from_args(arguments),
                        **dict(_step_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        # Setup context and call wrapped to detect streaming
        step_token = None
        with trace_api.use_span(span, end_on_exit=False):
            step_token = _setup_step_context(node_id)
            result = wrapped(*args, **kwargs)

        # Check if result is an async iterator (streaming)
        if hasattr(result, "__aiter__"):
            # Streaming mode - return async generator that continues with this span
            return self._arun_stream_continue(result, span, step_token)

        # Non-streaming mode - return coroutine that handles it
        async def non_stream_wrapper():
            nonlocal step_token
            try:
                # Await the coroutine inside span context
                with trace_api.use_span(span, end_on_exit=False):
                    response = await result

                    # Detach token immediately after execution, while still in span context
                    if step_token:
                        try:
                            context_api.detach(step_token)
                            step_token = None
                        except Exception:
                            pass

                span.set_status(trace_api.StatusCode.OK)
                output = _extract_output(response)
                if output:
                    span.set_attribute(OUTPUT_VALUE, output)
                    span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

                return response

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise

            finally:
                if step_token:
                    try:
                        context_api.detach(step_token)
                    except Exception:
                        pass
                span.end()

        return non_stream_wrapper()

    async def _arun_stream_continue(
        self,
        async_iter: Any,
        span: Any,
        step_token: Any,
    ) -> Any:
        """Continue streaming async step with existing span"""
        accumulated_output = []
        try:
            with trace_api.use_span(span, end_on_exit=False):
                try:
                    async for response in async_iter:
                        if hasattr(response, "content") and response.content:
                            accumulated_output.append(str(response.content))
                        yield response
                finally:
                    if step_token:
                        try:
                            context_api.detach(step_token)
                            step_token = None
                        except Exception:
                            pass

            span.set_status(trace_api.StatusCode.OK)
            if accumulated_output:
                span.set_attribute(OUTPUT_VALUE, "\n".join(accumulated_output))
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            if step_token:
                try:
                    context_api.detach(step_token)
                except Exception:
                    pass
            span.end()


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
GRAPH_NODE_ID = SpanAttributes.GRAPH_NODE_ID
GRAPH_NODE_NAME = SpanAttributes.GRAPH_NODE_NAME
GRAPH_NODE_PARENT_ID = SpanAttributes.GRAPH_NODE_PARENT_ID

# mime types
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

# span kinds
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
