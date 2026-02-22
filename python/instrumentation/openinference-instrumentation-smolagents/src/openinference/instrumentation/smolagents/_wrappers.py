from collections.abc import Generator
from enum import Enum
from inspect import signature
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.util.types import AttributeValue

import openinference.instrumentation as oi
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from smolagents.tools import Tool  # type: ignore[import-untyped]


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
            if isinstance(value, Enum):
                value = value.value
            yield key, value


def _get_input_value(method: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    arguments = _bind_arguments(method, *args, **kwargs)
    arguments = _strip_method_args(arguments)
    return safe_json_dumps(arguments)


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    return bound_args.arguments


def _strip_method_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key not in ("self", "cls")}


def _smolagent_run_attributes(
    agent: Any, arguments: dict[str, Any]
) -> Iterator[Tuple[str, AttributeValue]]:
    if task := agent.task:
        yield "smolagents.task", task
    if additional_args := arguments.get("additional_args"):
        yield "smolagents.additional_args", safe_json_dumps(additional_args)
    yield "smolagents.max_steps", agent.max_steps
    yield "smolagents.tools_names", list(agent.tools.keys())
    for managed_agent_index, managed_agent in enumerate(agent.managed_agents.values()):
        yield f"smolagents.managed_agents.{managed_agent_index}.name", managed_agent.name
        yield (
            f"smolagents.managed_agents.{managed_agent_index}.description",
            managed_agent.description,
        )
        if getattr(managed_agent, "additional_prompting", None):
            yield (
                f"smolagents.managed_agents.{managed_agent_index}.additional_prompting",
                managed_agent.additional_prompting,
            )
        elif getattr(managed_agent, "managed_agent_prompt", None):
            yield (
                f"smolagents.managed_agents.{managed_agent_index}.managed_agent_prompt",
                managed_agent.managed_agent_prompt,
            )
        if getattr(managed_agent, "agent", None):
            yield (
                f"smolagents.managed_agents.{managed_agent_index}.max_steps",
                managed_agent.agent.max_steps,
            )
            yield (
                f"smolagents.managed_agents.{managed_agent_index}.tools_names",
                list(managed_agent.agent.tools.keys()),
            )


class _RunWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Union[Any, Generator[str, None, None]]:
        # Skip instrumentation if explicitly disabled
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        agent = instance
        span_name = f"{getattr(agent, 'name', None) or agent.__class__.__name__}.run"
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        # Start parent span for the full run
        span = self._tracer.start_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_smolagent_run_attributes(agent, arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        )

        # Set the tracing context for downstream spans
        context = trace_api.set_span_in_context(span)
        token = context_api.attach(context)
        agent_output = []

        try:
            agent_output = wrapped(*args, **kwargs)
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace_api.StatusCode.ERROR)
            raise

        is_generator = isinstance(agent_output, Generator)

        # Handle streaming (generator) run
        if is_generator:
            output_chunks: list[str] = []

            def wrapped_generator() -> Generator[str, None, None]:
                try:
                    # Collect chunks for final output
                    for chunk in agent_output:
                        output_chunks.append(str(chunk))
                        yield chunk
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace_api.StatusCode.ERROR)
                    raise
                finally:
                    # Set output value from the last observation
                    steps = getattr(agent.monitor, "steps", [])
                    history = getattr(agent.monitor, "history", [])

                    if steps:
                        observation = getattr(steps[-1], "observations", None)
                        if observation:
                            span.set_attribute(OUTPUT_VALUE, observation)
                    elif history:
                        observation = getattr(history[-1], "observations", None)
                        if observation:
                            span.set_attribute(OUTPUT_VALUE, observation)
                    elif output_chunks:
                        span.set_attribute(OUTPUT_VALUE, "".join(output_chunks))

                    # Record token usage metadata
                    span.set_attribute(
                        LLM_TOKEN_COUNT_PROMPT, agent.monitor.total_input_token_count
                    )
                    span.set_attribute(
                        LLM_TOKEN_COUNT_COMPLETION, agent.monitor.total_output_token_count
                    )
                    span.set_attribute(
                        LLM_TOKEN_COUNT_TOTAL,
                        agent.monitor.total_input_token_count
                        + agent.monitor.total_output_token_count,
                    )

                    span.set_status(trace_api.StatusCode.OK)
                    span.end()
                    context_api.detach(token)

            return wrapped_generator()

        # Handle non-streaming (normal) run
        else:
            try:
                # Set output value from the agent output
                span.set_attribute(OUTPUT_VALUE, str(agent_output))
                # Record token usage metadata
                span.set_attribute(LLM_TOKEN_COUNT_PROMPT, agent.monitor.total_input_token_count)
                span.set_attribute(
                    LLM_TOKEN_COUNT_COMPLETION, agent.monitor.total_output_token_count
                )
                span.set_attribute(
                    LLM_TOKEN_COUNT_TOTAL,
                    agent.monitor.total_input_token_count + agent.monitor.total_output_token_count,
                )
                return agent_output

            except Exception as e:
                span.record_exception(e)
                span.set_status(trace_api.StatusCode.ERROR)
                raise

            finally:
                span.set_status(trace_api.StatusCode.OK)
                span.end()
                context_api.detach(token)


def _finalize_step_span(
    span: trace_api.Span,
    step_log: Any,
) -> None:
    """
    Finalize the step span by recording output & setting status.

    - Attaches observations as the output value.
    - Sets status to OK if no error is present.
    - Captures & logs any errors that occur.
    """
    observations = getattr(step_log, "observations", None)
    if observations is not None:
        span.set_attribute(OUTPUT_VALUE, str(observations))

    if span.status.status_code != trace_api.StatusCode.ERROR:  # type: ignore[attr-defined]
        error = getattr(step_log, "error", None)
        if error is None:
            span.set_status(trace_api.StatusCode.OK)
        else:
            _record_step_error(span, error)


def _record_step_error(span: trace_api.Span, error: Exception) -> None:
    """
    Record error details for the step span.

    - Marks expected tool errors as recoverable (status = OK).
    - Adds structured error details as span events for expected tool errors.
    - Marks unexpected errors as failures & records the exception.
    """
    error_type = error.__class__.__name__
    expected_error_types = {"AgentToolCallError", "AgentToolExecutionError"}

    if error_type in expected_error_types:
        error_attrs: dict[str, Any]
        if hasattr(error, "dict") and callable(getattr(error, "dict")):
            error_attrs = error.dict()
        else:
            error_attrs = {"message": str(error)}

        span.add_event(
            name="agent.step_recovery",
            attributes={**error_attrs, "severity": "expected"},
        )
        span.set_status(trace_api.StatusCode.OK)
    else:
        span.record_exception(error)
        span.set_status(trace_api.StatusCode.ERROR)


class _StepWrapper:
    """
    Wrapper to instrument agent steps with OpenTelemetry spans.

    - Creates a span per step with input/output attributes.
    - Records errors & propagates exceptions.
    - Finalizes span status & preserves context for each step.
    """

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            yield from wrapped(*args, **kwargs)
            return

        agent = instance
        span_name = f"Step {agent.step_number}"
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: CHAIN,
                INPUT_VALUE: _get_input_value(wrapped, *args, **kwargs),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            try:
                yield from wrapped(*args, **kwargs)
            except Exception as execution_error:
                span.record_exception(execution_error)
                span.set_status(trace_api.StatusCode.ERROR)
                raise
            finally:
                step_log = args[0] if args else None
                if step_log is not None:
                    _finalize_step_span(span, step_log)


def _llm_input_messages(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    def process_message(idx: int, role: str, content: str) -> Iterator[Tuple[str, Any]]:
        yield f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_ROLE}", role
        yield f"{LLM_INPUT_MESSAGES}.{idx}.{MESSAGE_CONTENT}", content

    if isinstance(prompt := arguments.get("prompt"), str):
        yield from process_message(0, "user", prompt)
    elif isinstance(messages := arguments.get("messages"), list):
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role, content = message.get("role"), message.get("content")
            if isinstance(content, list) and role:
                for subcontent in content:
                    if isinstance(subcontent, dict) and (text := subcontent.get("text")):
                        yield from process_message(i, role, text)


def _llm_output_messages(output_message: Any) -> Mapping[str, AttributeValue]:
    oi_message: oi.Message = {}
    oi_message_contents: list[oi.MessageContent] = []
    if (role := getattr(output_message, "role", None)) is not None:
        oi_message["role"] = role
    if (content := getattr(output_message, "content", None)) is not None:
        oi_message_contents.append(oi.TextMessageContent(type="text", text=content))

    # Add the reasoning_content if available in raw.choices[0].message structure
    if (raw := getattr(output_message, "raw", None)) is not None:
        if (choices := getattr(raw, "choices", None)) is not None:
            if isinstance(choices, list) and len(choices) > 0:
                if (message := getattr(choices[0], "message", None)) is not None:
                    if (
                        reasoning_content := getattr(message, "reasoning_content", None)
                    ) is not None:
                        oi_message_contents.append(
                            oi.TextMessageContent(type="text", text=reasoning_content)
                        )

    oi_message["contents"] = oi_message_contents
    oi_tool_calls: list[oi.ToolCall] = []
    if isinstance(tool_calls := getattr(output_message, "tool_calls", None), list):
        for tool_call in tool_calls:
            oi_tool_call: oi.ToolCall = {}
            if (tool_call_id := getattr(tool_call, "id", None)) is not None:
                oi_tool_call["id"] = tool_call_id
            if (function := getattr(tool_call, "function", None)) is not None:
                oi_function: oi.ToolCallFunction = {}
                if (name := getattr(function, "name", None)) is not None:
                    oi_function["name"] = name
                if isinstance(arguments := getattr(function, "arguments", None), str):
                    oi_function["arguments"] = arguments
                oi_tool_call["function"] = oi_function
                oi_tool_calls.append(oi_tool_call)
    oi_message["tool_calls"] = oi_tool_calls
    return oi.get_llm_output_message_attributes(messages=[oi_message])


def _output_value_and_mime_type(output: Any) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_MIME_TYPE, JSON
    if hasattr(output, "model_dump_json") and callable(output.model_dump_json):
        try:
            yield OUTPUT_VALUE, output.model_dump_json(exclude_unset=True)
        except Exception:
            # model_dump_json() failed so convert to dict first then use safe_json_dumps
            # This handles Pydantic models with non-serializable nested objects
            if hasattr(output, "model_dump") and callable(output.model_dump):
                yield OUTPUT_VALUE, safe_json_dumps(output.model_dump())
            elif hasattr(output, "dict") and callable(output.dict):
                # Pydantic v1 compatibility
                yield OUTPUT_VALUE, safe_json_dumps(output.dict())
            else:
                yield OUTPUT_VALUE, safe_json_dumps(output)
    else:
        yield OUTPUT_VALUE, safe_json_dumps(output)


def _llm_invocation_parameters(
    model: Any, arguments: Mapping[str, Any]
) -> Iterator[Tuple[str, Any]]:
    model_kwargs = _ if isinstance(_ := getattr(model, "kwargs", {}), dict) else {}
    kwargs = _ if isinstance(_ := arguments.get("kwargs"), dict) else {}
    yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(model_kwargs | kwargs)


def _llm_tools(tools_to_call_from: list[Any]) -> Iterator[Tuple[str, Any]]:
    from smolagents import Tool
    from smolagents.models import get_tool_json_schema  # type: ignore[import-untyped]

    if not isinstance(tools_to_call_from, list):
        return
    for tool_index, tool in enumerate(tools_to_call_from):
        if isinstance(tool, Tool):
            yield (
                f"{LLM_TOOLS}.{tool_index}.{TOOL_JSON_SCHEMA}",
                safe_json_dumps(get_tool_json_schema(tool)),
            )


def _tools(tool: "Tool") -> Iterator[Tuple[str, Any]]:
    if tool_name := getattr(tool, "name", None):
        yield TOOL_NAME, tool_name
    if tool_description := getattr(tool, "description", None):
        yield TOOL_DESCRIPTION, tool_description
    yield TOOL_PARAMETERS, safe_json_dumps(tool.inputs)


def _input_value_and_mime_type(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


class _ModelWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        if _has_active_llm_parent_span():
            return wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, *args, **kwargs)
        span_name = f"{instance.__class__.__name__}.generate"
        model = instance
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(instance, arguments)),
                **dict(_llm_input_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            output_message = wrapped(*args, **kwargs)
            span.set_status(trace_api.StatusCode.OK)
            token_usage = getattr(output_message, "token_usage", None)
            if token_usage:
                input_tokens = token_usage.input_tokens
                output_tokens = token_usage.output_tokens
                total_tokens = token_usage.total_tokens
            else:
                input_tokens = model.last_input_token_count
                output_tokens = model.last_output_token_count
                total_tokens = input_tokens + output_tokens
            span.set_attribute(LLM_TOKEN_COUNT_PROMPT, input_tokens)
            span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, output_tokens)
            span.set_attribute(LLM_TOKEN_COUNT_TOTAL, total_tokens)
            span.set_attribute(LLM_MODEL_NAME, model.model_id)
            if provider := (
                infer_llm_provider_from_class_name(instance)
                or infer_llm_provider_from_endpoint(
                    extract_llm_endpoint_from_sdk_instance(instance)
                )
            ):
                span.set_attribute(LLM_PROVIDER, provider.value)
            if system := infer_llm_system_from_model(model.model_id):
                span.set_attribute(LLM_SYSTEM, system.value)
            span.set_attributes(_llm_output_messages(output_message))
            span.set_attributes(dict(_llm_tools(arguments.get("tools_to_call_from", []))))
            span.set_attributes(dict(_output_value_and_mime_type(output_message)))
        return output_message


class _ToolCallWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        span_name = f"{instance.__class__.__name__}"
        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                INPUT_VALUE: _get_input_value(
                    wrapped,
                    *args,
                    **kwargs,
                ),
                **dict(_tools(instance)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            response = wrapped(*args, **kwargs)
            span.set_status(trace_api.StatusCode.OK)
            span.set_attributes(
                dict(
                    _output_value_and_mime_type_for_tool_span(
                        response=response,
                        output_type=instance.output_type,
                    )
                )
            )
        return response


def _output_value_and_mime_type_for_tool_span(
    response: Any, output_type: str
) -> Iterator[Tuple[str, Any]]:
    if output_type in (
        "string",
        "boolean",
        "integer",
        "number",
    ) or isinstance(response, str):
        yield OUTPUT_VALUE, response
        yield OUTPUT_MIME_TYPE, TEXT
    else:
        yield OUTPUT_VALUE, safe_json_dumps(response)
        yield OUTPUT_MIME_TYPE, JSON


def _has_active_llm_parent_span() -> bool:
    """
    Returns true if there is a currently actively LLM span.
    """
    current_span = trace_api.get_current_span()
    return (
        current_span.get_span_context().is_valid
        and current_span.is_recording()
        and isinstance(current_span, ReadableSpan)
        and (current_span.attributes or {}).get(OPENINFERENCE_SPAN_KIND) == LLM
    )


def infer_llm_provider_from_class_name(
    instance: Any = None,
) -> Optional[OpenInferenceLLMProviderValues]:
    """Infer the LLM provider from an SDK instance using the model class name when possible."""
    if instance is None:
        return None

    class_name = instance.__class__.__name__

    if class_name in ["LiteLLMModel", "LiteLLMRouterModel"]:
        model_id = getattr(instance, "model_id", None)
        if isinstance(model_id, str):
            provider_prefix = model_id.split("/", 1)[0].lower()
            try:
                return OpenInferenceLLMProviderValues(provider_prefix)
            except ValueError:
                return None

    if class_name == "InferenceClientModel":
        return None

    if class_name == "OpenAIServerModel":
        return OpenInferenceLLMProviderValues.OPENAI

    if class_name == "AzureOpenAIServerModel":
        return OpenInferenceLLMProviderValues.AZURE

    if class_name == "AmazonBedrockServerModel":
        return OpenInferenceLLMProviderValues.AWS

    return None


def extract_llm_endpoint_from_sdk_instance(
    instance: Any = None,
) -> Optional[str]:
    """Extract the LLM API endpoint from an SDK instance when possible."""
    if instance is None:
        return None

    endpoint = (
        getattr(instance, "api_base", None)
        or getattr(instance, "base_url", None)
        or getattr(instance, "endpoint", None)
        or getattr(instance, "host", None)
    )

    if not isinstance(endpoint, str) and endpoint is not None:
        return str(endpoint)

    return endpoint


def infer_llm_provider_from_endpoint(
    endpoint: Optional[str] = None,
) -> Optional[OpenInferenceLLMProviderValues]:
    """Infer the LLM provider from an SDK instance using the API endpoint when possible."""
    if not isinstance(endpoint, str):
        return None

    hostname = urlparse(endpoint).hostname
    if hostname is None:
        return None

    host = hostname.lower()

    if host.endswith("api.openai.com"):
        return OpenInferenceLLMProviderValues.OPENAI

    if "openai.azure.com" in host:
        return OpenInferenceLLMProviderValues.AZURE

    if host.endswith("googleapis.com"):
        return OpenInferenceLLMProviderValues.GOOGLE

    if host.endswith("anthropic.com"):
        return OpenInferenceLLMProviderValues.ANTHROPIC

    if "bedrock" in host or host.endswith("amazonaws.com"):
        return OpenInferenceLLMProviderValues.AWS

    if host.endswith("cohere.ai"):
        return OpenInferenceLLMProviderValues.COHERE

    if host.endswith("mistral.ai"):
        return OpenInferenceLLMProviderValues.MISTRALAI

    if host.endswith("x.ai"):
        return OpenInferenceLLMProviderValues.XAI

    if host.endswith("deepseek.com"):
        return OpenInferenceLLMProviderValues.DEEPSEEK

    return None


def infer_llm_system_from_model(
    model_name: Optional[str] = None,
) -> Optional[OpenInferenceLLMSystemValues]:
    """Infer the LLM system from a model identifier when possible."""
    if not isinstance(model_name, str):
        return None

    model = model_name.lower()

    if model.startswith(
        (
            "gpt",
            "o1",
            "o3",
            "o4",
            "text-embedding",
            "davinci",
            "curie",
            "babbage",
            "ada",
            "azure",
            "openai",
        )
    ):
        return OpenInferenceLLMSystemValues.OPENAI

    if model.startswith(("anthropic", "claude", "google_anthropic_vertex")):
        return OpenInferenceLLMSystemValues.ANTHROPIC

    if model.startswith(("cohere", "command")):
        return OpenInferenceLLMSystemValues.COHERE

    if model.startswith(("mistral", "mixtral", "pixtral")):
        return OpenInferenceLLMSystemValues.MISTRALAI

    if model.startswith(("vertex", "gemini", "google")):
        return OpenInferenceLLMSystemValues.VERTEXAI

    return None


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOOLS = SpanAttributes.LLM_TOOLS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS

# message attributes
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS

# mime types
JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

# span kinds
AGENT = OpenInferenceSpanKindValues.AGENT.value
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
LLM = OpenInferenceSpanKindValues.LLM.value
TOOL = OpenInferenceSpanKindValues.TOOL.value

# tool attributes
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA

# tool call attributes
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
