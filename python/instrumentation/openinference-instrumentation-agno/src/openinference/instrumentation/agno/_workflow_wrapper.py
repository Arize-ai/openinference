from typing import (
    Any,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Tuple,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import get_attributes_from_context
from openinference.instrumentation.agno.utils import (
    _AGNO_ARUN_SPANNED_CONTEXT_KEY,
    _AGNO_PARENT_NODE_CONTEXT_KEY,
    _bind_arguments,
    _flatten,
    _generate_node_id,
)
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)


def _get_input_from_args(arguments: Mapping[str, Any]) -> str:
    """Extract input message from workflow/step arguments."""
    step_input = arguments.get("step_input")
    if step_input is not None:
        # For subsequent steps, try to get the actual content from previous steps
        if hasattr(step_input, "get_last_step_content"):
            try:
                content = step_input.get_last_step_content()
                if content:
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, dict):
                        import json

                        return json.dumps(content, indent=2, ensure_ascii=False)
                    else:
                        return str(content)
            except Exception:
                pass

        # Fallback to step_input.input (for first step or if no previous content)
        if hasattr(step_input, "input"):
            input_value = step_input.input
            if input_value is not None:
                if isinstance(input_value, str):
                    return input_value
                elif isinstance(input_value, list):
                    return "\n".join(str(item) for item in input_value)
                elif hasattr(input_value, "model_dump_json"):
                    return str(input_value.model_dump_json(indent=2, exclude_none=True))
                elif isinstance(input_value, dict):
                    import json

                    return json.dumps(input_value, indent=2, ensure_ascii=False)
                else:
                    return str(input_value)

    execution_input = arguments.get("execution_input")
    if execution_input is not None and hasattr(execution_input, "input"):
        input_value = execution_input.input
        if input_value is not None:
            if isinstance(input_value, str):
                return input_value
            elif isinstance(input_value, list):
                return "\n".join(str(item) for item in input_value)
            elif hasattr(input_value, "model_dump_json"):
                return str(input_value.model_dump_json(indent=2, exclude_none=True))
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


def _extract_output(response: Any) -> tuple[str, str]:
    """Extract output from workflow/step response."""
    if response is None:
        return "", TEXT
    if isinstance(response, str):
        return response, TEXT
    content = getattr(response, "content", None)
    if content is not None:
        if hasattr(content, "model_dump_json"):
            try:
                return str(content.model_dump_json()), JSON
            except Exception:
                pass
        return str(content), TEXT
    if hasattr(response, "model_dump_json"):
        try:
            return str(response.model_dump_json()), JSON
        except Exception:
            pass
    return str(response), TEXT


def _workflow_attributes(instance: Any) -> Iterator[Tuple[str, AttributeValue]]:
    """Extract attributes from workflow instance."""
    if hasattr(instance, "name") and instance.name:
        yield GRAPH_NODE_NAME, instance.name

    if hasattr(instance, "description") and instance.description:
        yield "agno.workflow.description", instance.description

    if hasattr(instance, "steps") and instance.steps:
        yield "agno.workflow.steps_count", len(instance.steps)
        step_names = []
        step_types = []
        for step in instance.steps:
            if hasattr(step, "name") and step.name:
                step_names.append(step.name)
            # Capture step type
            step_type = type(step).__name__
            step_types.append(step_type)

        if step_names:
            yield "agno.workflow.steps", step_names
        if step_types:
            yield "agno.workflow.step_types", step_types


def _step_attributes(instance: Any) -> Iterator[Tuple[str, AttributeValue]]:
    """Extract attributes from step instance."""
    # Get parent from execution context
    context_parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)

    if hasattr(instance, "name") and instance.name:
        yield GRAPH_NODE_NAME, instance.name

    if context_parent_id and isinstance(context_parent_id, str):
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


def _workflow_run_arguments(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    """Extract user_id and session_id from workflow run arguments."""
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")

    # For agno v2: session_id might be in the session object
    session = arguments.get("session")
    if session and hasattr(session, "session_id"):
        session_id = session.session_id

    if session_id:
        yield SESSION_ID, session_id
    if user_id:
        yield USER_ID, user_id


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
                        **dict(_workflow_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        workflow_token = None
        result = None
        try:
            with trace_api.use_span(span, end_on_exit=False):
                workflow_token = _setup_node_context(node_id)
                result = wrapped(*args, **kwargs)

                # Check if result is an iterator (streaming)
                is_streaming = hasattr(result, "__iter__") and not isinstance(result, (str, bytes))

                if is_streaming:
                    if workflow_token:
                        try:
                            context_api.detach(workflow_token)
                            workflow_token = None
                        except Exception:
                            pass
                    return self._run_stream_continue(result, span, node_id, instance)

                # Non-streaming mode - detach token immediately while still in span context
                if workflow_token:
                    try:
                        context_api.detach(workflow_token)
                        workflow_token = None
                    except Exception:
                        pass

            # Set output and workflow ID (outside use_span but token already detached)
            span.set_status(trace_api.StatusCode.OK)
            output, mime_type = _extract_output(result)
            if output:
                span.set_attribute(OUTPUT_VALUE, output)
                span.set_attribute(OUTPUT_MIME_TYPE, mime_type)

            # Set workflow ID after execution (it's initialized inside the wrapped method)
            if hasattr(instance, "id") and instance.id:
                span.set_attribute("agno.workflow.id", instance.id)

            # Capture the workflow run_id
            run_id = getattr(result, "run_id", None)
            if run_id:
                span.set_attribute("agno.run.id", run_id)

            # Check instance user_id
            if hasattr(instance, "user_id") and instance.user_id:
                span.set_attribute(USER_ID, instance.user_id)

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
        node_id: str,
        instance: Any = None,
    ) -> Any:
        """Continue streaming workflow with existing span"""
        final_response = None
        iterator = iter(iterator)
        try:
            while True:
                ctx_token = None
                workflow_token = None
                try:
                    ctx_token = context_api.attach(trace_api.set_span_in_context(span))
                    workflow_token = _setup_node_context(node_id)
                    response = next(iterator)
                except StopIteration:
                    break
                finally:
                    if workflow_token is not None:
                        context_api.detach(workflow_token)
                    if ctx_token is not None:
                        context_api.detach(ctx_token)

                final_response = response
                yield response

            span.set_status(trace_api.StatusCode.OK)
            # Extract output only from the final response
            if final_response is not None:
                output, mime_type = _extract_output(final_response)
                if output:
                    span.set_attribute(OUTPUT_VALUE, output)
                    span.set_attribute(OUTPUT_MIME_TYPE, mime_type)

            # Set workflow ID after execution (it's initialized inside the wrapped method)
            if instance and hasattr(instance, "id") and instance.id:
                span.set_attribute("agno.workflow.id", instance.id)

            # Capture the workflow run_id
            run_id = getattr(final_response, "run_id", None)
            if run_id:
                span.set_attribute("agno.run.id", run_id)

            # Capture user_id from instance if available
            if instance and hasattr(instance, "user_id") and instance.user_id:
                span.set_attribute(USER_ID, instance.user_id)

        except (StopIteration, StopAsyncIteration):
            raise
        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            span.end()
            if instance is not None and getattr(instance, "_oi_arun_spanned", False):
                delattr(instance, "_oi_arun_spanned")

    def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Bind arguments to extract input
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        # Background runs return a placeholder WorkflowRunOutput immediately while
        # the real work happens later so we skip span creation here entirely.
        if arguments.get("background"):
            return wrapped(*args, **kwargs)

        workflow_name = getattr(instance, "name", "Workflow").replace(" ", "_").replace("-", "_")
        # Use the actual method name for consistency
        method_name = getattr(wrapped, "__name__", "arun")
        span_name = f"{workflow_name}.{method_name}"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_from_args(arguments),
                        **dict(_workflow_attributes(instance)),
                        **dict(_workflow_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        # Setup context and call wrapped to detect streaming
        workflow_token = None
        is_streaming = False
        with trace_api.use_span(span, end_on_exit=False):
            workflow_token = _setup_node_context(node_id)
            result = wrapped(*args, **kwargs)

            # Check if result is an async iterator (streaming)
            is_streaming = hasattr(result, "__aiter__")
            if is_streaming and workflow_token:
                try:
                    context_api.detach(workflow_token)
                    workflow_token = None
                except Exception:
                    pass

        if is_streaming:
            return self._arun_stream_continue(result, span, node_id, instance)

        # Non-streaming mode - return coroutine that handles it
        async def non_stream_wrapper() -> Any:
            nonlocal workflow_token
            ctx_token = None
            spanned_token = None
            try:
                ctx_token = context_api.attach(trace_api.set_span_in_context(span))
                if workflow_token is None:
                    workflow_token = _setup_node_context(node_id)
                # Set the context-local marker so _WorkflowExecuteWrapper.aexecute
                # skips creating its own span. This token is scoped to this coroutine's
                # context and never leaks into background tasks created by _arun_background.
                spanned_token = _setup_arun_spanned_context()

                response = await result

                # Detach tokens immediately after execution completes
                if workflow_token is not None:
                    context_api.detach(workflow_token)
                    workflow_token = None
                if spanned_token is not None:
                    context_api.detach(spanned_token)
                    spanned_token = None
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                    ctx_token = None

                span.set_status(trace_api.StatusCode.OK)
                output, mime_type = _extract_output(response)
                if output:
                    span.set_attribute(OUTPUT_VALUE, output)
                    span.set_attribute(OUTPUT_MIME_TYPE, mime_type)

                # Set workflow ID after execution (it's initialized inside the wrapped method)
                if hasattr(instance, "id") and instance.id:
                    span.set_attribute("agno.workflow.id", instance.id)

                # Capture the workflow run_id
                run_id = getattr(response, "run_id", None)
                if run_id:
                    span.set_attribute("agno.run.id", run_id)

                # Capture user_id from instance if available
                if hasattr(instance, "user_id") and instance.user_id:
                    span.set_attribute(USER_ID, instance.user_id)

                return response

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise

            finally:
                if spanned_token is not None:
                    context_api.detach(spanned_token)
                if workflow_token is not None:
                    context_api.detach(workflow_token)
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()

        return non_stream_wrapper()

    async def _arun_stream_continue(
        self,
        async_iter: Any,
        span: Any,
        node_id: str,
        instance: Any = None,
    ) -> Any:
        """Continue streaming async workflow with existing span"""
        final_response = None
        iterator = async_iter.__aiter__()
        try:
            while True:
                ctx_token = None
                workflow_token = None
                spanned_token = None
                try:
                    ctx_token = context_api.attach(trace_api.set_span_in_context(span))
                    workflow_token = _setup_node_context(node_id)
                    spanned_token = _setup_arun_spanned_context()
                    response = await anext(iterator)
                except StopAsyncIteration:
                    break
                finally:
                    if spanned_token is not None:
                        context_api.detach(spanned_token)
                    if workflow_token is not None:
                        context_api.detach(workflow_token)
                    if ctx_token is not None:
                        context_api.detach(ctx_token)

                final_response = response
                yield response

            span.set_status(trace_api.StatusCode.OK)
            # Extract output only from the final response
            if final_response is not None:
                output, mime_type = _extract_output(final_response)
                if output:
                    span.set_attribute(OUTPUT_VALUE, output)
                    span.set_attribute(OUTPUT_MIME_TYPE, mime_type)

            # Set workflow ID after execution (it's initialized inside the wrapped method)
            if instance and hasattr(instance, "id") and instance.id:
                span.set_attribute("agno.workflow.id", instance.id)

            # Capture the workflow run_id
            run_id = getattr(final_response, "run_id", None)
            if run_id:
                span.set_attribute("agno.run.id", run_id)

            # Capture user_id from instance if available
            if instance and hasattr(instance, "user_id") and instance.user_id:
                span.set_attribute(USER_ID, instance.user_id)

        except (StopIteration, StopAsyncIteration):
            raise
        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            span.end()


class _WorkflowExecuteWrapper:
    """
    Wrapper for coroutines that do the actual step execution and write the real output onto the
    WorkflowRunOutput, regardless of whether the run was started (via arun or _arun_background).
    """
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    @staticmethod
    def _already_spanned() -> bool:
        # Returns True only inside the foreground coroutine/stream that owns the arun span.
        # The background tasks get their own context copy where this key is absent.
        return bool(context_api.get_value(_AGNO_ARUN_SPANNED_CONTEXT_KEY))

    @staticmethod
    def _apply_run_output_attributes(span: Any, instance: Any, run_output: Any) -> None:
        if run_output is None:
            return
        output, mime_type = _extract_output(run_output)
        if output:
            span.set_attribute(OUTPUT_VALUE, output)
            span.set_attribute(OUTPUT_MIME_TYPE, mime_type)

        if hasattr(instance, "id") and instance.id:
            span.set_attribute("agno.workflow.id", instance.id)

        run_id = getattr(run_output, "run_id", None)
        if run_id:
            span.set_attribute("agno.run.id", run_id)

        user_id = getattr(run_output, "user_id", None)
        if user_id:
            span.set_attribute(USER_ID, user_id)

    async def aexecute(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        if self._already_spanned():
            return await wrapped(*args, **kwargs)

        workflow_name = getattr(instance, "name", "Workflow").replace(" ", "_").replace("-", "_")
        method_name = getattr(wrapped, "__name__", "_aexecute")
        span_name = f"{workflow_name}.{method_name}"

        node_id = _generate_node_id()
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        workflow_run_response = arguments.get("workflow_run_response")

        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_from_args(arguments),
                        **dict(_workflow_attributes(instance)),
                        **dict(_workflow_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        workflow_token = None
        ctx_token = None
        try:
            ctx_token = context_api.attach(trace_api.set_span_in_context(span))
            workflow_token = _setup_node_context(node_id)
            response = await wrapped(*args, **kwargs)

            if workflow_token is not None:
                context_api.detach(workflow_token)
                workflow_token = None
            if ctx_token is not None:
                context_api.detach(ctx_token)
                ctx_token = None

            span.set_status(trace_api.StatusCode.OK)
            self._apply_run_output_attributes(
                span,
                instance,
                workflow_run_response if workflow_run_response is not None else response,
            )

            return response

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            if workflow_token is not None:
                try:
                    context_api.detach(workflow_token)
                except Exception:
                    pass
            if ctx_token is not None:
                try:
                    context_api.detach(ctx_token)
                except Exception:
                    pass
            span.end()

    async def aexecute_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for item in wrapped(*args, **kwargs):
                yield item
            return

        if self._already_spanned():
            async for item in wrapped(*args, **kwargs):
                yield item
            return

        workflow_name = getattr(instance, "name", "Workflow").replace(" ", "_").replace("-", "_")
        method_name = getattr(wrapped, "__name__", "_aexecute_stream")
        span_name = f"{workflow_name}.{method_name}"

        node_id = _generate_node_id()
        arguments = _bind_arguments(wrapped, *args, **kwargs)
        workflow_run_response = arguments.get("workflow_run_response")

        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_from_args(arguments),
                        **dict(_workflow_attributes(instance)),
                        **dict(_workflow_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        async_iter = wrapped(*args, **kwargs).__aiter__()
        try:
            while True:
                ctx_token = None
                workflow_token = None
                try:
                    ctx_token = context_api.attach(trace_api.set_span_in_context(span))
                    workflow_token = _setup_node_context(node_id)
                    event = await anext(async_iter)
                except StopAsyncIteration:
                    break
                finally:
                    if workflow_token is not None:
                        context_api.detach(workflow_token)
                    if ctx_token is not None:
                        context_api.detach(ctx_token)

                yield event

            span.set_status(trace_api.StatusCode.OK)
            self._apply_run_output_attributes(span, instance, workflow_run_response)

        except (StopIteration, StopAsyncIteration):
            raise
        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
        finally:
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
                step_token = _setup_node_context(node_id)
                result = wrapped(*args, **kwargs)

                # Check if result is an iterator (streaming)
                is_streaming = hasattr(result, "__iter__") and not isinstance(result, (str, bytes))

                if is_streaming:
                    if step_token:
                        try:
                            context_api.detach(step_token)
                            step_token = None
                        except Exception:
                            pass
                    return self._run_stream_continue(result, span, node_id)

                # Non-streaming mode - detach token immediately while still in span context
                if step_token:
                    try:
                        context_api.detach(step_token)
                        step_token = None
                    except Exception:
                        pass

            # Set output and return (outside use_span but token already detached)
            span.set_status(trace_api.StatusCode.OK)
            output, mime_type = _extract_output(result)
            if output:
                span.set_attribute(OUTPUT_VALUE, output)
                span.set_attribute(OUTPUT_MIME_TYPE, mime_type)

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
        node_id: str,
    ) -> Any:
        """Continue streaming step with existing span"""
        final_response = None
        iterator = iter(iterator)
        try:
            while True:
                ctx_token = None
                step_token = None
                try:
                    ctx_token = context_api.attach(trace_api.set_span_in_context(span))
                    step_token = _setup_node_context(node_id)
                    response = next(iterator)
                except StopIteration:
                    break
                finally:
                    if step_token is not None:
                        context_api.detach(step_token)
                    if ctx_token is not None:
                        context_api.detach(ctx_token)

                final_response = response
                yield response

            span.set_status(trace_api.StatusCode.OK)
            # Extract output only from the final response
            if final_response is not None:
                output, mime_type = _extract_output(final_response)
                if output:
                    span.set_attribute(OUTPUT_VALUE, output)
                    span.set_attribute(OUTPUT_MIME_TYPE, mime_type)

        except (StopIteration, StopAsyncIteration):
            raise
        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
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
        is_streaming = False
        with trace_api.use_span(span, end_on_exit=False):
            step_token = _setup_node_context(node_id)
            result = wrapped(*args, **kwargs)

            # Check if result is an async iterator (streaming)
            is_streaming = hasattr(result, "__aiter__")
            if is_streaming and step_token:
                try:
                    context_api.detach(step_token)
                    step_token = None
                except Exception:
                    pass

        if is_streaming:
            return self._arun_stream_continue(result, span, node_id)

        # Non-streaming mode - return coroutine that handles it
        async def non_stream_wrapper() -> Any:
            nonlocal step_token
            ctx_token = None
            try:
                ctx_token = context_api.attach(trace_api.set_span_in_context(span))
                if step_token is None:
                    step_token = _setup_node_context(node_id)
                response = await result

                # Detach tokens immediately after execution completes
                if step_token is not None:
                    context_api.detach(step_token)
                    step_token = None
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                    ctx_token = None

                span.set_status(trace_api.StatusCode.OK)
                output, mime_type = _extract_output(response)
                if output:
                    span.set_attribute(OUTPUT_VALUE, output)
                    span.set_attribute(OUTPUT_MIME_TYPE, mime_type)

                return response

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise

            finally:
                if step_token is not None:
                    context_api.detach(step_token)
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()

        return non_stream_wrapper()

    async def _arun_stream_continue(
        self,
        async_iter: Any,
        span: Any,
        node_id: str,
    ) -> Any:
        """Continue streaming async step with existing span"""
        final_response = None
        iterator = async_iter.__aiter__()
        try:
            while True:
                ctx_token = None
                step_token = None
                try:
                    ctx_token = context_api.attach(trace_api.set_span_in_context(span))
                    step_token = _setup_node_context(node_id)
                    response = await anext(iterator)
                except StopAsyncIteration:
                    break
                finally:
                    if step_token is not None:
                        context_api.detach(step_token)
                    if ctx_token is not None:
                        context_api.detach(ctx_token)

                final_response = response
                yield response

            span.set_status(trace_api.StatusCode.OK)
            # Extract output only from the final response
            if final_response is not None:
                output, mime_type = _extract_output(final_response)
                if output:
                    span.set_attribute(OUTPUT_VALUE, output)
                    span.set_attribute(OUTPUT_MIME_TYPE, mime_type)

        except (StopIteration, StopAsyncIteration):
            raise
        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            span.end()


class _ParallelWrapper:
    """
    Wrapper for Parallel to propagate context to worker threads.

    When Parallel executes steps using ThreadPoolExecutor, worker threads
    don't inherit the parent thread's OpenTelemetry context. This wrapper
    ensures context propagation so all parallel steps appear in the same trace
    with correct parent-child relationships.
    """

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def execute(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        parallel_name = getattr(instance, "name", "Parallel").replace(" ", "_").replace("-", "_")
        span_name = f"{parallel_name}.execute"

        # Generate unique node ID for this parallel container
        node_id = _generate_node_id()

        # Get parent node ID from workflow context
        parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)

        # Bind arguments
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        # Create span for the parallel container
        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **({GRAPH_NODE_PARENT_ID: parent_id} if parent_id else {}),
                        INPUT_VALUE: _get_input_from_args(arguments),
                        "agno.parallel.step_count": len(getattr(instance, "steps", [])),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        parallel_token = None
        result = None
        try:
            with trace_api.use_span(span, end_on_exit=False):
                parallel_token = _setup_node_context(node_id)

                # Context propagation is now handled by Agno's copy_context().run
                result = wrapped(*args, **kwargs)

                # Check if result is a sync iterator (streaming mode)
                is_streaming = hasattr(result, "__iter__") and not isinstance(result, (str, bytes))

                if is_streaming:
                    if parallel_token:
                        try:
                            context_api.detach(parallel_token)
                            parallel_token = None
                        except Exception:
                            pass
                    return self._execute_stream_continue(result, span, node_id)

                # Non-streaming mode - detach token immediately
                if parallel_token:
                    try:
                        context_api.detach(parallel_token)
                        parallel_token = None
                    except Exception:
                        pass

            # Set output
            span.set_status(trace_api.StatusCode.OK)
            output, mime_type = _extract_output(result)
            if output:
                span.set_attribute(OUTPUT_VALUE, output)
                span.set_attribute(OUTPUT_MIME_TYPE, mime_type)

            return result

        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            # Cleanup for non-streaming (streaming handles its own)
            # Note: execute only handles sync methods, no need to check for async iterators
            if result is not None:
                is_streaming = hasattr(result, "__iter__") and not isinstance(result, (str, bytes))
            else:
                is_streaming = False

            if not is_streaming:
                if parallel_token:
                    try:
                        context_api.detach(parallel_token)
                    except Exception:
                        pass
                span.end()

    def _execute_stream_continue(
        self,
        iterator: Any,
        span: Any,
        node_id: str,
    ) -> Any:
        """Continue streaming parallel execution with existing span"""
        accumulated_output = []
        iterator = iter(iterator)
        try:
            while True:
                ctx_token = None
                parallel_token = None
                try:
                    ctx_token = context_api.attach(trace_api.set_span_in_context(span))
                    parallel_token = _setup_node_context(node_id)
                    response = next(iterator)
                except StopIteration:
                    break
                finally:
                    if parallel_token is not None:
                        context_api.detach(parallel_token)
                    if ctx_token is not None:
                        context_api.detach(ctx_token)

                if hasattr(response, "content") and response.content:
                    accumulated_output.append(str(response.content))
                yield response

            span.set_status(trace_api.StatusCode.OK)
            if accumulated_output:
                span.set_attribute(OUTPUT_VALUE, "\n".join(accumulated_output))
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

        except (StopIteration, StopAsyncIteration):
            raise
        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            span.end()

    def aexecute(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Handle async parallel execution"""
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        parallel_name = getattr(instance, "name", "Parallel").replace(" ", "_").replace("-", "_")
        span_name = f"{parallel_name}.aexecute"

        # Generate unique node ID for this parallel container
        node_id = _generate_node_id()

        # Get parent node ID from workflow context
        parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)

        # Bind arguments
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        # Create span for the parallel container
        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: CHAIN,
                        GRAPH_NODE_ID: node_id,
                        **({GRAPH_NODE_PARENT_ID: parent_id} if parent_id else {}),
                        INPUT_VALUE: _get_input_from_args(arguments),
                        "agno.parallel.step_count": len(getattr(instance, "steps", [])),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        # Setup context and call wrapped to detect streaming
        parallel_token = None
        is_streaming = False
        with trace_api.use_span(span, end_on_exit=False):
            parallel_token = _setup_node_context(node_id)

            result = wrapped(*args, **kwargs)

            # Check if result is an async iterator (streaming)
            is_streaming = hasattr(result, "__aiter__")
            if is_streaming and parallel_token:
                try:
                    context_api.detach(parallel_token)
                    parallel_token = None
                except Exception:
                    pass

        if is_streaming:
            return self._aexecute_stream_continue(result, span, node_id)

        # Non-streaming async mode - return coroutine that handles it
        async def non_stream_wrapper() -> Any:
            nonlocal parallel_token
            ctx_token = None
            try:
                ctx_token = context_api.attach(trace_api.set_span_in_context(span))
                if parallel_token is None:
                    parallel_token = _setup_node_context(node_id)
                response = await result

                # Detach tokens immediately after execution completes
                if parallel_token is not None:
                    context_api.detach(parallel_token)
                    parallel_token = None
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                    ctx_token = None

                span.set_status(trace_api.StatusCode.OK)
                output, mime_type = _extract_output(response)
                if output:
                    span.set_attribute(OUTPUT_VALUE, output)
                    span.set_attribute(OUTPUT_MIME_TYPE, mime_type)

                return response

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                span.record_exception(e)
                raise

            finally:
                if parallel_token is not None:
                    context_api.detach(parallel_token)
                if ctx_token is not None:
                    context_api.detach(ctx_token)
                span.end()

        return non_stream_wrapper()

    async def _aexecute_stream_continue(
        self,
        async_iter: Any,
        span: Any,
        node_id: str,
    ) -> Any:
        """Continue streaming async parallel execution with existing span"""
        accumulated_output = []
        iterator = async_iter.__aiter__()
        try:
            while True:
                ctx_token = None
                parallel_token = None
                try:
                    ctx_token = context_api.attach(trace_api.set_span_in_context(span))
                    parallel_token = _setup_node_context(node_id)
                    response = await anext(iterator)
                except StopAsyncIteration:
                    break
                finally:
                    if parallel_token is not None:
                        context_api.detach(parallel_token)
                    if ctx_token is not None:
                        context_api.detach(ctx_token)

                if hasattr(response, "content") and response.content:
                    accumulated_output.append(str(response.content))
                yield response

            span.set_status(trace_api.StatusCode.OK)
            if accumulated_output:
                span.set_attribute(OUTPUT_VALUE, "\n".join(accumulated_output))
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)

        except (StopIteration, StopAsyncIteration):
            raise
        except Exception as e:
            span.set_status(trace_api.StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise

        finally:
            span.end()


def _setup_node_context(node_id: str) -> Any:
    """Set up context for different nodes to propagate to child steps."""
    return context_api.attach(
        context_api.set_value(_AGNO_PARENT_NODE_CONTEXT_KEY, node_id)
    )


def _setup_arun_spanned_context() -> Any:
    """Mark the current async context as already covered by a Workflow.arun span."""
    return context_api.attach(
        context_api.set_value(_AGNO_ARUN_SPANNED_CONTEXT_KEY, True)
    )


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
SESSION_ID = SpanAttributes.SESSION_ID
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
USER_ID = SpanAttributes.USER_ID
GRAPH_NODE_ID = SpanAttributes.GRAPH_NODE_ID
GRAPH_NODE_NAME = SpanAttributes.GRAPH_NODE_NAME
GRAPH_NODE_PARENT_ID = SpanAttributes.GRAPH_NODE_PARENT_ID

# mime types
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

# span kinds
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
