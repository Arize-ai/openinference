from inspect import signature
from secrets import token_hex
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    OrderedDict,
    Tuple,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_AGNO_PARENT_NODE_CONTEXT_KEY = context_api.create_key("agno_parent_node_id")


def _flatten(mapping: Optional[Mapping[str, Any]]) -> Iterator[Tuple[str, AttributeValue]]:
    if not mapping:
        return
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, Mapping):
            for sub_key, sub_value in _flatten(value):
                yield f"{key}.{sub_key}", sub_value
        elif isinstance(value, list) and any(isinstance(item, Mapping) for item in value):
            for index, sub_mapping in enumerate(value):
                for sub_key, sub_value in _flatten(sub_mapping):
                    yield f"{key}.{index}.{sub_key}", sub_value
        else:
            yield key, value


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    arguments = bound_args.arguments
    arguments = OrderedDict(
        {key: value for key, value in arguments.items() if value is not None and value != {}}
    )
    return arguments


def _generate_node_id() -> str:
    return token_hex(8)  # Generates 16 hex characters (8 bytes)


def _get_input_from_args(arguments: Mapping[str, Any]) -> str:
    """Extract input message from workflow/step arguments."""
    for key in ["input", "message", "messages", "prompt"]:
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
    return str(response)


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
            result = wrapped(*args, **kwargs)
            # Return the result as-is (either iterator or regular result)
            return result
        
        # Call the wrapped method to check what it returns
        result = wrapped(*args, **kwargs)
        
        # Check if the result is an iterator (streaming mode)
        if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
            # Streaming mode - return a generator that wraps the iterator
            return self._run_stream_wrapper_iter(result, instance)
        else:
            # Non-streaming mode - return the result with instrumentation
            return self._run_wrapper_result(result, instance)
    
    def _run_wrapper_result(
        self,
        result: Any,
        instance: Any,
    ) -> Any:
        """Wrap a non-streaming workflow execution result"""
        workflow_name = getattr(instance, "name", "Workflow").replace(" ", "_").replace("-", "_")
        span_name = f"{workflow_name}.run"
        
        # Generate unique node ID for this execution
        node_id = _generate_node_id()
        
        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **dict(_workflow_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )
        
        workflow_token = None
        try:
            with trace_api.use_span(span, end_on_exit=False):
                workflow_token = _setup_workflow_context(node_id)
                # Result is already available, just add instrumentation
                pass
            
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
            if workflow_token:
                try:
                    context_api.detach(workflow_token)
                except Exception:
                    pass
            span.end()
    
    def _run_stream_wrapper_iter(
        self,
        iterator: Any,
        instance: Any,
    ) -> Any:
        """Wrap a streaming workflow execution"""
        workflow_name = getattr(instance, "name", "Workflow").replace(" ", "_").replace("-", "_")
        span_name = f"{workflow_name}.run"
        
        # Generate unique node ID for this execution
        node_id = _generate_node_id()
        
        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **dict(_workflow_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )
        
        workflow_token = None
        accumulated_output = []
        try:
            with trace_api.use_span(span, end_on_exit=False):
                workflow_token = _setup_workflow_context(node_id)
                try:
                    for response in iterator:
                        # Try to extract content for accumulated output
                        if hasattr(response, 'content') and response.content:
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
            result = wrapped(*args, **kwargs)
            # Return the result as-is (either coroutine or async iterator)
            return result
        
        # Call the wrapped method to check what it returns
        result = wrapped(*args, **kwargs)
        
        # Check if the result is an async iterator (streaming mode)
        if hasattr(result, '__aiter__'):
            # Streaming mode - return an async generator that wraps the iterator
            return self._arun_stream_wrapper_iter(result, instance)
        else:
            # Non-streaming mode - return a coroutine that awaits the result
            return self._arun_wrapper_coro(result, instance)
    
    async def _arun_wrapper_coro(
        self,
        coro: Awaitable[Any],
        instance: Any,
    ) -> Any:
        """Wrap a non-streaming async workflow execution"""
        workflow_name = getattr(instance, "name", "Workflow").replace(" ", "_").replace("-", "_")
        span_name = f"{workflow_name}.arun"
        
        # Generate unique node ID for this execution
        node_id = _generate_node_id()
        
        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **dict(_workflow_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )
        
        workflow_token = None
        try:
            with trace_api.use_span(span, end_on_exit=False):
                workflow_token = _setup_workflow_context(node_id)
                response = await coro
            
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
    
    async def _arun_stream_wrapper_iter(
        self,
        async_iter: Any,
        instance: Any,
    ) -> Any:
        """Wrap a streaming async workflow execution"""
        workflow_name = getattr(instance, "name", "Workflow").replace(" ", "_").replace("-", "_")
        span_name = f"{workflow_name}.arun"
        
        # Generate unique node ID for this execution
        node_id = _generate_node_id()
        
        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **dict(_workflow_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )
        
        workflow_token = None
        accumulated_output = []
        try:
            with trace_api.use_span(span, end_on_exit=False):
                workflow_token = _setup_workflow_context(node_id)
                try:
                    async for response in async_iter:
                        # Try to extract content for accumulated output
                        if hasattr(response, 'content') and response.content:
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
            result = wrapped(*args, **kwargs)
            return result
        
        # Call the wrapped method to check what it returns
        result = wrapped(*args, **kwargs)
        
        # Check if the result is an iterator (streaming mode)
        if hasattr(result, '__iter__') and not isinstance(result, (str, bytes, dict)):
            # Streaming mode - return a generator that wraps the iterator
            return self._run_stream_wrapper(result, instance)
        else:
            # Non-streaming mode - return the result with instrumentation
            return self._run_wrapper(result, instance)
    
    def _run_wrapper(
        self,
        result: Any,
        instance: Any,
    ) -> Any:
        """Wrap a non-streaming step execution"""
        step_name = getattr(instance, "name", "Step").replace(" ", "_").replace("-", "_")
        span_name = f"{step_name}.execute"
        
        # Generate unique node ID for this execution
        node_id = _generate_node_id()
        
        # Get parent node ID from workflow context
        parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)
        
        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **({GRAPH_NODE_PARENT_ID: parent_id} if parent_id else {}),
                        **dict(_step_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )
        
        step_token = None
        try:
            with trace_api.use_span(span, end_on_exit=False):
                step_token = _setup_step_context(node_id)
                # Result is already available, just add instrumentation
                pass
            
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
            if step_token:
                try:
                    context_api.detach(step_token)
                except Exception:
                    pass
            span.end()
    
    def _run_stream_wrapper(
        self,
        iterator: Any,
        instance: Any,
    ) -> Any:
        """Wrap a streaming step execution"""
        step_name = getattr(instance, "name", "Step").replace(" ", "_").replace("-", "_")
        span_name = f"{step_name}.execute_stream"
        
        # Generate unique node ID for this execution
        node_id = _generate_node_id()
        
        # Get parent node ID from workflow context
        parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)
        
        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **({GRAPH_NODE_PARENT_ID: parent_id} if parent_id else {}),
                        **dict(_step_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )
        
        step_token = None
        accumulated_output = []
        try:
            with trace_api.use_span(span, end_on_exit=False):
                step_token = _setup_step_context(node_id)
                try:
                    for response in iterator:
                        # Try to extract content for accumulated output
                        if hasattr(response, 'content') and response.content:
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
            result = wrapped(*args, **kwargs)
            return result
        
        # Call the wrapped method to check what it returns
        result = wrapped(*args, **kwargs)
        
        # Check if the result is an async iterator (streaming mode)
        if hasattr(result, '__aiter__'):
            # Streaming mode - return an async generator that wraps the iterator
            return self._arun_stream_wrapper(result, instance)
        else:
            # Non-streaming mode - return a coroutine that awaits the result
            return self._arun_wrapper(result, instance)
    
    async def _arun_wrapper(
        self,
        coro: Awaitable[Any],
        instance: Any,
    ) -> Any:
        """Wrap a non-streaming async step execution"""
        step_name = getattr(instance, "name", "Step").replace(" ", "_").replace("-", "_")
        span_name = f"{step_name}.aexecute"
        
        # Generate unique node ID for this execution
        node_id = _generate_node_id()
        
        # Get parent node ID from workflow context
        parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)
        
        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **({GRAPH_NODE_PARENT_ID: parent_id} if parent_id else {}),
                        **dict(_step_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )
        
        step_token = None
        try:
            with trace_api.use_span(span, end_on_exit=False):
                step_token = _setup_step_context(node_id)
                response = await coro
            
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
    
    async def _arun_stream_wrapper(
        self,
        async_iter: Any,
        instance: Any,
    ) -> Any:
        """Wrap a streaming async step execution"""
        step_name = getattr(instance, "name", "Step").replace(" ", "_").replace("-", "_")
        span_name = f"{step_name}.aexecute_stream"
        
        # Generate unique node ID for this execution
        node_id = _generate_node_id()
        
        # Get parent node ID from workflow context
        parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)
        
        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **({GRAPH_NODE_PARENT_ID: parent_id} if parent_id else {}),
                        **dict(_step_attributes(instance)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )
        
        step_token = None
        accumulated_output = []
        try:
            with trace_api.use_span(span, end_on_exit=False):
                step_token = _setup_step_context(node_id)
                try:
                    async for response in async_iter:
                        # Try to extract content for accumulated output
                        if hasattr(response, 'content') and response.content:
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

