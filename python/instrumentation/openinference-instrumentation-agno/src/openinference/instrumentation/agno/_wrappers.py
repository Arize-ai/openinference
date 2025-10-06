import json
import logging
from enum import Enum
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
    Union,
    cast,
)

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from agno.agent import Agent
from agno.models.base import Model
from agno.team import Team
from agno.tools.function import Function, FunctionCall
from agno.tools.toolkit import Toolkit
from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

_AGNO_PARENT_NODE_CONTEXT_KEY = context_api.create_key("agno_parent_node_id")

logger = logging.getLogger(__name__)


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
    arguments = bound_args.arguments
    arguments = OrderedDict(
        {key: value for key, value in arguments.items() if value is not None and value != {}}
    )
    return arguments


def _strip_method_args(arguments: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in arguments.items() if key not in ("self", "cls")}


def _generate_node_id() -> str:
    return token_hex(8)  # Generates 16 hex characters (8 bytes)


def _run_arguments(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, AttributeValue]]:
    user_id = arguments.get("user_id")
    session_id = arguments.get("session_id")

    # For agno v2: session_id might be in the session object for internal _run method
    session = arguments.get("session")
    if session and hasattr(session, "session_id"):
        session_id = session.session_id

    if session_id:
        yield SESSION_ID, session_id

    if user_id:
        yield USER_ID, user_id


def _agent_run_attributes(
    agent: Union[Agent, Team], key_suffix: str = ""
) -> Iterator[Tuple[str, AttributeValue]]:
    # Get parent from execution context instead of structural parent
    context_parent_id = context_api.get_value(_AGNO_PARENT_NODE_CONTEXT_KEY)

    if isinstance(agent, Team):
        # Set graph attributes for team
        if agent.name:
            yield GRAPH_NODE_NAME, agent.name

        # Use context parent instead of structural parent
        if context_parent_id:
            yield GRAPH_NODE_PARENT_ID, cast(str, context_parent_id)

        # Set legacy team attributes
        yield f"agno{key_suffix}.team", agent.name or ""
        for member in agent.members:
            yield from _agent_run_attributes(member, f".{member.name}")

    elif isinstance(agent, Agent):
        # Set graph attributes for agent
        if agent.name:
            yield GRAPH_NODE_NAME, agent.name

        # Use context parent instead of structural parent
        if context_parent_id:
            yield GRAPH_NODE_PARENT_ID, cast(str, context_parent_id)

        # Set legacy agent attributes
        if agent.name:
            yield f"agno{key_suffix}.agent", agent.name or ""

        if agent.knowledge:
            yield f"agno{key_suffix}.knowledge", agent.knowledge.__class__.__name__

        if agent.tools:
            tool_names = []
            for tool in agent.tools:
                if isinstance(tool, Function):
                    tool_names.append(tool.name)
                elif isinstance(tool, Toolkit):
                    tool_names.extend([f for f in tool.functions.keys()])
                elif callable(tool):
                    tool_names.append(tool.__name__)
                else:
                    tool_names.append(str(tool))
            yield f"agno{key_suffix}.tools", tool_names


def _setup_team_context(agent: Union[Agent, Team], node_id: str) -> Optional[Any]:
    if isinstance(agent, Team):
        team_ctx = context_api.set_value(_AGNO_PARENT_NODE_CONTEXT_KEY, node_id)
        return context_api.attach(team_ctx)
    return None


class _RunWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    """
    We need to keep track of parent/child relationships for agent logging. We do this by:
    1. Each run() method generates a unique node_id and sets it directly as GRAPH_NODE_ID in span
    attributes
    2. Team.run() sets _AGNO_PARENT_NODE_CONTEXT_KEY for child agents
    3. Agent.run() inherits _AGNO_PARENT_NODE_CONTEXT_KEY from team context for parent relationships
    4. _agent_run_attributes() uses _AGNO_PARENT_NODE_CONTEXT_KEY to set GRAPH_NODE_PARENT_ID
    5. This ensures correct parent-child relationships with unique node IDs for each execution
    """

    def run(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        arguments = _bind_arguments(wrapped, *args, **kwargs)
        
        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                run_response = wrapped(*args, **kwargs)
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                return run_response

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)

    def run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()
        arguments = _bind_arguments(wrapped, *args, **kwargs)

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                yield from wrapped(*args, **kwargs)
                # Use get_last_run_output instead of removed agent.run_response
                session_id = None
                try:
                    session_id = arguments.get("session_id")
                except Exception:
                    session_id = None

                run_response = None
                if hasattr(agent, "get_last_run_output"):
                    run_response = agent.get_last_run_output(session_id=session_id)

                span.set_status(trace_api.StatusCode.OK)
                if run_response is not None:
                    span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                    span.set_attribute(OUTPUT_MIME_TYPE, JSON)

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            response = await wrapped(*args, **kwargs)
            return response

        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                run_response = await wrapped(*args, **kwargs)
                span.set_status(trace_api.StatusCode.OK)
                span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                return run_response
            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)

    async def arun_stream(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for response in await wrapped(*args, **kwargs):
                yield response

        agent = instance
        if hasattr(agent, "name") and agent.name:
            agent_name = agent.name.replace(" ", "_").replace("-", "_")
        else:
            agent_name = "Agent"
        span_name = f"{agent_name}.run"

        # Generate unique node ID for this execution
        node_id = _generate_node_id()

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        with self._tracer.start_as_current_span(
            span_name,
            attributes=dict(
                _flatten(
                    {
                        OPENINFERENCE_SPAN_KIND: AGENT,
                        GRAPH_NODE_ID: node_id,
                        INPUT_VALUE: _get_input_value(
                            wrapped,
                            *args,
                            **kwargs,
                        ),
                        **dict(_agent_run_attributes(agent)),
                        **dict(_run_arguments(arguments)),
                        **dict(get_attributes_from_context()),
                    }
                )
            ),
        ) as span:
            team_token = _setup_team_context(agent, node_id)

            try:
                async for response in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                    yield response

                # Use get_last_run_output instead of removed agent.run_response
                session_id = None
                try:
                    session_id = arguments.get("session_id")
                except Exception:
                    session_id = None

                run_response = None
                if hasattr(agent, "get_last_run_output"):
                    run_response = agent.get_last_run_output(session_id=session_id)

                span.set_status(trace_api.StatusCode.OK)
                if run_response is not None:
                    span.set_attribute(OUTPUT_VALUE, run_response.to_json())
                    span.set_attribute(OUTPUT_MIME_TYPE, JSON)

            except Exception as e:
                span.set_status(trace_api.StatusCode.ERROR, str(e))
                raise

            finally:
                if team_token:
                    context_api.detach(team_token)


def _llm_input_messages(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Extract LLM input messages from arguments and format for OpenInference."""
    messages = arguments.get("messages", [])
    for i, message in enumerate(messages):
        if hasattr(message, 'role') and hasattr(message, 'content'):
            # Set message role and content
            yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_ROLE}", message.role
            
            # Handle different content types
            if hasattr(message, 'get_content_string'):
                content = message.get_content_string()
            elif isinstance(message.content, str):
                content = message.content
            else:
                content = str(message.content) if message.content else ""
                
            if content:
                yield f"{LLM_INPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}", content
            
            # Handle tool calls in messages - only include if they have actual data
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for j, tool_call in enumerate(message.tool_calls):
                    if isinstance(tool_call, dict):
                        if tool_call.get('id'):
                            yield f"{LLM_INPUT_MESSAGES}.{i}.tool_calls.{j}.tool_call.id", tool_call['id']
                        if tool_call.get('function', {}).get('name'):
                            yield f"{LLM_INPUT_MESSAGES}.{i}.tool_calls.{j}.tool_call.function.name", tool_call['function']['name']
                        if tool_call.get('function', {}).get('arguments'):
                            yield f"{LLM_INPUT_MESSAGES}.{i}.tool_calls.{j}.tool_call.function.arguments", tool_call['function']['arguments']

    # Handle tools array - only include if tools exist
    tools = arguments.get("tools", [])
    if tools:
        for tool_index, tool in enumerate(tools):
            # Only include tools that have meaningful data
            if tool and isinstance(tool, dict) and tool.get('function'):
                yield f"{LLM_TOOLS}.{tool_index}.{TOOL_JSON_SCHEMA}", safe_json_dumps(tool)


def _llm_output_messages(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Extract LLM output messages from arguments and format for OpenInference."""
    # Check for assistant_message in arguments
    assistant_message = arguments.get("assistant_message")
    if assistant_message and hasattr(assistant_message, 'role'):
        index = 0
        
        # Set role
        yield f"{LLM_OUTPUT_MESSAGES}.{index}.{MESSAGE_ROLE}", assistant_message.role
        
        # Handle content - only include if not None/empty
        content = None
        if hasattr(assistant_message, 'get_content_string'):
            content = assistant_message.get_content_string()
        elif hasattr(assistant_message, 'content') and assistant_message.content:
            content = assistant_message.content
            
        if content:
            yield f"{LLM_OUTPUT_MESSAGES}.{index}.{MESSAGE_CONTENT}", content
        
        # Handle tool calls in assistant message - only include if they exist
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            for j, tool_call in enumerate(assistant_message.tool_calls):
                if isinstance(tool_call, dict):
                    if tool_call.get('id'):
                        yield f"{LLM_OUTPUT_MESSAGES}.{index}.tool_calls.{j}.tool_call.id", tool_call['id']
                    if tool_call.get('function', {}).get('name'):
                        yield f"{LLM_OUTPUT_MESSAGES}.{index}.tool_calls.{j}.tool_call.function.name", tool_call['function']['name']
                    if tool_call.get('function', {}).get('arguments'):
                        yield f"{LLM_OUTPUT_MESSAGES}.{index}.tool_calls.{j}.tool_call.function.arguments", tool_call['function']['arguments']
                        
                        # Also set the LLM function call attribute for compatibility
                        function_call_data = {
                            "name": tool_call['function']['name'],
                            "arguments": tool_call['function']['arguments']
                        }
                        yield LLM_FUNCTION_CALL, safe_json_dumps(function_call_data)
                        
    # Also check for messages in arguments that have role='assistant'
    messages = arguments.get("messages", [])
    output_index = 0
    for message in messages:
        if hasattr(message, 'role') and message.role == 'assistant':
            yield f"{LLM_OUTPUT_MESSAGES}.{output_index}.{MESSAGE_ROLE}", message.role
            
            # Handle content - only include if not None/empty
            content = None
            if hasattr(message, 'get_content_string'):
                content = message.get_content_string()
            elif hasattr(message, 'content') and message.content:
                content = message.content
                
            if content:
                yield f"{LLM_OUTPUT_MESSAGES}.{output_index}.{MESSAGE_CONTENT}", content
                
            # Handle tool calls - only include if they exist
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for j, tool_call in enumerate(message.tool_calls):
                    if isinstance(tool_call, dict):
                        if tool_call.get('id'):
                            yield f"{LLM_OUTPUT_MESSAGES}.{output_index}.tool_calls.{j}.tool_call.id", tool_call['id']
                        if tool_call.get('function', {}).get('name'):
                            yield f"{LLM_OUTPUT_MESSAGES}.{output_index}.tool_calls.{j}.tool_call.function.name", tool_call['function']['name']
                        if tool_call.get('function', {}).get('arguments'):
                            yield f"{LLM_OUTPUT_MESSAGES}.{output_index}.tool_calls.{j}.tool_call.function.arguments", tool_call['function']['arguments']
            
            output_index += 1


def _llm_invocation_parameters(
    model: Model, arguments: Optional[Mapping[str, Any]] = None
) -> Iterator[Tuple[str, Any]]:
    """Extract model invocation parameters and format for OpenInference."""
    request_kwargs = {}
    
    # Extract request parameters from various model types
    if getattr(model, "request_kwargs", None):
        request_kwargs = model.request_kwargs  # type: ignore[attr-defined]
    if getattr(model, "request_params", None):
        request_kwargs = model.request_params  # type: ignore[attr-defined]
    if getattr(model, "get_request_kwargs", None):
        request_kwargs = model.get_request_kwargs()  # type: ignore[attr-defined]
    if getattr(model, "get_request_params", None):
        # Special handling for OpenAIResponses model
        if model.__class__.__name__ == "OpenAIResponses" and arguments:
            messages = arguments.get("messages", [])
            request_kwargs = model.get_request_params(messages=messages)  # type: ignore[attr-defined]
        else:
            request_kwargs = model.get_request_params()  # type: ignore[attr-defined]
    
    # Add model-specific parameters that are commonly available
    common_params = ['temperature', 'max_tokens', 'max_completion_tokens', 'top_p', 'top_k', 
                    'frequency_penalty', 'presence_penalty', 'stop', 'stream']
    for param in common_params:
        if hasattr(model, param):
            value = getattr(model, param)
            if value is not None:
                request_kwargs[param] = value

    if request_kwargs:
        filtered_kwargs = _filter_sensitive_params(request_kwargs)
        yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(filtered_kwargs)


def _filter_sensitive_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out sensitive parameters from model request parameters."""
    sensitive_keys = frozenset(
        [
            "api_key",
            "api_base",
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_access_key",
            "aws_secret_key",
            "azure_endpoint",
            "azure_deployment",
            "azure_ad_token",
            "azure_ad_token_provider",
        ]
    )

    return {
        key: "[REDACTED]"
        if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys)
        else value
        for key, value in params.items()
    }


def _filter_none_values(data):
    """Recursively filter out None/null values and empty collections from data structures."""
    if isinstance(data, dict):
        filtered = {}
        for key, value in data.items():
            if value is not None:
                if isinstance(value, (dict, list)):
                    filtered_value = _filter_none_values(value)
                    if filtered_value:  # Only add if not empty after filtering
                        filtered[key] = filtered_value
                elif isinstance(value, (str, int, float, bool)):
                    # Keep all meaningful values
                    if isinstance(value, (int, float)) and value == 0:
                        # Still keep zero values, they might be meaningful
                        filtered[key] = value
                    elif value != "" or key in ["content", "role"]:  # Keep empty content/role as they might be meaningful
                        filtered[key] = value
                elif hasattr(value, '__class__'):
                    # Handle special object types
                    class_name = str(value.__class__)
                    if 'Timer' in class_name:
                        # Skip Timer objects as they're not serializable and not useful for tracing
                        continue
                    elif 'RunStatus' in class_name:
                        # Convert enum to string value
                        filtered[key] = str(value.value) if hasattr(value, 'value') else str(value)
                    else:
                        # Keep other objects, convert to string if needed
                        filtered[key] = str(value)
                else:
                    # Keep other non-None values
                    filtered[key] = value
        return filtered
    elif isinstance(data, list):
        filtered = []
        for item in data:
            if item is not None:
                filtered_item = _filter_none_values(item)
                if filtered_item or isinstance(item, (str, int, float, bool)):  # Keep primitive types
                    filtered.append(filtered_item if isinstance(item, (dict, list)) else item)
        return filtered
    else:
        return data


def _extract_meaningful_fields(obj, obj_type="object"):
    """Extract meaningful fields from Agno objects."""
    if hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump()
        except Exception:
            pass
    elif hasattr(obj, 'dict'):
        try:
            return obj.dict()
        except Exception:
            pass
    elif hasattr(obj, '__dict__'):
        try:
            return obj.__dict__
        except Exception:
            pass
    
    # Fallback to string representation but try to extract key info
    obj_str = str(obj)
    if 'RunOutput' in obj_str:
        # Try to extract key fields from RunOutput string representation
        extracted = {"_type": "RunOutput"}
        if 'run_id=' in obj_str:
            try:
                run_id = obj_str.split("run_id='")[1].split("'")[0]
                extracted["run_id"] = run_id
            except:
                pass
        if 'agent_name=' in obj_str:
            try:
                agent_name = obj_str.split("agent_name='")[1].split("'")[0]
                extracted["agent_name"] = agent_name
            except:
                pass
        if 'model=' in obj_str:
            try:
                model = obj_str.split("model='")[1].split("'")[0]
                extracted["model"] = model
            except:
                pass
        return extracted
    elif 'AgentSession' in obj_str:
        # Try to extract key fields from AgentSession string representation  
        extracted = {"_type": "AgentSession"}
        if 'session_id=' in obj_str:
            try:
                session_id = obj_str.split("session_id='")[1].split("'")[0]
                extracted["session_id"] = session_id
            except:
                pass
        if 'agent_id=' in obj_str:
            try:
                agent_id = obj_str.split("agent_id='")[1].split("'")[0]
                extracted["agent_id"] = agent_id
            except:
                pass
        return extracted
    
    return {"_type": obj_type, "_string_repr": str(obj)}


def _input_value_and_mime_type(arguments: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Format input arguments as JSON for OpenInference."""
    yield INPUT_MIME_TYPE, JSON
    
    # Create a clean input representation
    clean_args = {}
    for key, value in arguments.items():
        if key not in ("self", "cls") and value is not None:
            # Handle special Agno objects that need better serialization
            if hasattr(value, '__class__'):
                class_name = str(value.__class__)
                if any(agno_type in class_name for agno_type in ['RunOutput', 'AgentSession', 'AgentOutput', 'Message', 'Metrics']):
                    try:
                        extracted = _extract_meaningful_fields(value, class_name)
                        filtered_data = _filter_none_values(extracted)
                        if filtered_data:
                            clean_args[key] = filtered_data
                    except Exception:
                        # Fallback to string representation
                        clean_args[key] = str(value)
                    continue
            
            # Handle objects with model_dump or dict methods
            if hasattr(value, 'model_dump') or hasattr(value, 'dict'):
                try:
                    if hasattr(value, 'model_dump'):
                        serialized = value.model_dump()
                    else:
                        serialized = value.dict()
                    # Filter out None values from the serialized data
                    filtered_data = _filter_none_values(serialized)
                    if filtered_data:  # Only add if not empty after filtering
                        clean_args[key] = filtered_data
                except Exception:
                    clean_args[key] = str(value)
            elif isinstance(value, list):
                clean_list = []
                for item in value:
                    if item is not None:
                        if hasattr(item, 'model_dump'):
                            try:
                                serialized = item.model_dump()
                                filtered_data = _filter_none_values(serialized)
                                if filtered_data:
                                    clean_list.append(filtered_data)
                            except Exception:
                                clean_list.append(str(item))
                        elif hasattr(item, 'dict'):
                            try:
                                serialized = item.dict()
                                filtered_data = _filter_none_values(serialized)
                                if filtered_data:
                                    clean_list.append(filtered_data)
                            except Exception:
                                clean_list.append(str(item))
                        else:
                            clean_list.append(item)
                if clean_list:  # Only add if list is not empty
                    clean_args[key] = clean_list
            else:
                clean_args[key] = value
                
    yield INPUT_VALUE, safe_json_dumps(clean_args)


def _output_value_and_mime_type(output: str) -> Iterator[Tuple[str, Any]]:
    """Format output value as JSON for OpenInference."""
    yield OUTPUT_MIME_TYPE, JSON
    yield OUTPUT_VALUE, output


def _parse_model_output(output: Any) -> str:
    if hasattr(output, "model_dump_json"):
        return output.model_dump_json()  # type: ignore[no-any-return]
    elif isinstance(output, dict):
        return json.dumps(output)
    else:
        return str(output)


def _set_token_usage_attributes(span: trace_api.Span, response: Any) -> None:
    """Extract and set token usage attributes from response_usage if available."""
    if hasattr(response, "response_usage") and response.response_usage:
        metrics = response.response_usage

        # Set token usage attributes
        if hasattr(metrics, "input_tokens") and metrics.input_tokens:
            span.set_attribute(LLM_TOKEN_COUNT_PROMPT, metrics.input_tokens)

        if hasattr(metrics, "output_tokens") and metrics.output_tokens:
            span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, metrics.output_tokens)
            
        # Calculate total tokens
        if hasattr(metrics, "total_tokens") and metrics.total_tokens:
            span.set_attribute(LLM_TOKEN_COUNT_TOTAL, metrics.total_tokens)
        elif hasattr(metrics, "input_tokens") and hasattr(metrics, "output_tokens"):
            if metrics.input_tokens and metrics.output_tokens:
                span.set_attribute(LLM_TOKEN_COUNT_TOTAL, metrics.input_tokens + metrics.output_tokens)

        # Set cache-related tokens if available
        if hasattr(metrics, "cache_read_tokens") and metrics.cache_read_tokens:
            span.set_attribute("llm.token_count.prompt.details.cache_read", metrics.cache_read_tokens)

        if hasattr(metrics, "cache_write_tokens") and metrics.cache_write_tokens:
            span.set_attribute("llm.token_count.prompt.details.cache_write", metrics.cache_write_tokens)
            
        # Set reasoning tokens if available
        if hasattr(metrics, "reasoning_tokens") and metrics.reasoning_tokens:
            span.set_attribute("llm.token_count.completion.details.reasoning", metrics.reasoning_tokens)
            
        # Set audio tokens if available
        if hasattr(metrics, "audio_input_tokens") and metrics.audio_input_tokens:
            span.set_attribute("llm.token_count.prompt.details.audio", metrics.audio_input_tokens)
            
        if hasattr(metrics, "audio_output_tokens") and metrics.audio_output_tokens:
            span.set_attribute("llm.token_count.completion.details.audio", metrics.audio_output_tokens)

    else:
        logger.debug("No response_usage found in response")


class _ModelWrapper:
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

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.invoke"
        attributes = {
            OPENINFERENCE_SPAN_KIND: LLM,
            **dict(_input_value_and_mime_type(arguments)),
            **dict(_llm_invocation_parameters(model, arguments)),
            **dict(_llm_input_messages(arguments)),
            **dict(_llm_output_messages(arguments)),
            **dict(get_attributes_from_context()),
        }
        with self._tracer.start_as_current_span(
            span_name,
            attributes=attributes,
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)

            response = wrapped(*args, **kwargs)
            output_message = _parse_model_output(response)

            # Extract and set token usage from the response
            _set_token_usage_attributes(span, response)

            span.set_attributes(dict(_output_value_and_mime_type(output_message)))
            return response

    def run_stream(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.invoke_stream"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(model, arguments)),
                **dict(_llm_input_messages(arguments)),
                **dict(_llm_output_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)
            # Token usage will be set after streaming completes based on final response

            responses = []
            for chunk in wrapped(*args, **kwargs):
                responses.append(chunk)
                yield chunk

            # Find the final response with complete metrics (usually the last one with response_usage)
            final_response_with_metrics = None
            for response in reversed(responses):  # Check from last to first
                if hasattr(response, "response_usage") and response.response_usage:
                    final_response_with_metrics = response
                    break

            # Extract and set token usage from the final response
            if final_response_with_metrics and final_response_with_metrics.response_usage:
                metrics = final_response_with_metrics.response_usage

                # Set token usage attributes
                if hasattr(metrics, "input_tokens") and metrics.input_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_PROMPT, metrics.input_tokens)

                if hasattr(metrics, "output_tokens") and metrics.output_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, metrics.output_tokens)
                    
                # Calculate total tokens
                if hasattr(metrics, "total_tokens") and metrics.total_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_TOTAL, metrics.total_tokens)
                elif hasattr(metrics, "input_tokens") and hasattr(metrics, "output_tokens"):
                    if metrics.input_tokens and metrics.output_tokens:
                        span.set_attribute(LLM_TOKEN_COUNT_TOTAL, metrics.input_tokens + metrics.output_tokens)

                # Set cache-related tokens if available
                if hasattr(metrics, "cache_read_tokens") and metrics.cache_read_tokens:
                    span.set_attribute("llm.token_count.prompt.details.cache_read", metrics.cache_read_tokens)

                if hasattr(metrics, "cache_write_tokens") and metrics.cache_write_tokens:
                    span.set_attribute("llm.token_count.prompt.details.cache_write", metrics.cache_write_tokens)

            else:
                logger.debug("No final response with metrics found in streaming response")

            output_message = json.dumps([_parse_model_output(response) for response in responses])
            span.set_attributes(dict(_output_value_and_mime_type(output_message)))

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.ainvoke"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(model, arguments)),
                **dict(_llm_input_messages(arguments)),
                **dict(_llm_output_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)

            response = await wrapped(*args, **kwargs)
            output_message = _parse_model_output(response)

            # Extract and set token usage from the response
            _set_token_usage_attributes(span, response)

            span.set_attributes(dict(_output_value_and_mime_type(output_message)))
            return response

    async def arun_stream(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for response in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                yield response
            return

        arguments = _bind_arguments(wrapped, *args, **kwargs)

        model = instance
        model_name = model.name
        span_name = f"{model_name}.ainvoke_stream"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: LLM,
                **dict(_input_value_and_mime_type(arguments)),
                **dict(_llm_invocation_parameters(model, arguments)),
                **dict(_llm_input_messages(arguments)),
                **dict(_llm_output_messages(arguments)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            span.set_status(trace_api.StatusCode.OK)
            span.set_attribute(LLM_MODEL_NAME, model.id)
            span.set_attribute(LLM_PROVIDER, model.provider)
            # Token usage will be set after streaming completes based on final response

            responses = []
            async for chunk in wrapped(*args, **kwargs):  # type: ignore[attr-defined]
                responses.append(chunk)
                yield chunk

            # Find the final response with complete metrics (usually the last one with response_usage)
            final_response_with_metrics = None
            for response in reversed(responses):  # Check from last to first
                if hasattr(response, "response_usage") and response.response_usage:
                    final_response_with_metrics = response
                    break

            # Extract and set token usage from the final response
            if final_response_with_metrics and final_response_with_metrics.response_usage:
                metrics = final_response_with_metrics.response_usage

                # Set token usage attributes
                if hasattr(metrics, "input_tokens") and metrics.input_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_PROMPT, metrics.input_tokens)

                if hasattr(metrics, "output_tokens") and metrics.output_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_COMPLETION, metrics.output_tokens)
                    
                # Calculate total tokens
                if hasattr(metrics, "total_tokens") and metrics.total_tokens:
                    span.set_attribute(LLM_TOKEN_COUNT_TOTAL, metrics.total_tokens)
                elif hasattr(metrics, "input_tokens") and hasattr(metrics, "output_tokens"):
                    if metrics.input_tokens and metrics.output_tokens:
                        span.set_attribute(LLM_TOKEN_COUNT_TOTAL, metrics.input_tokens + metrics.output_tokens)

                # Set cache-related tokens if available
                if hasattr(metrics, "cache_read_tokens") and metrics.cache_read_tokens:
                    span.set_attribute("llm.token_count.prompt.details.cache_read", metrics.cache_read_tokens)

                if hasattr(metrics, "cache_write_tokens") and metrics.cache_write_tokens:
                    span.set_attribute("llm.token_count.prompt.details.cache_write", metrics.cache_write_tokens)

            else:
                logger.debug("No final response with metrics found in streaming response")

            output_message = json.dumps([_parse_model_output(response) for response in responses])
            span.set_attributes(dict(_output_value_and_mime_type(output_message)))


def _function_call_attributes(function_call: FunctionCall) -> Iterator[Tuple[str, Any]]:
    function = function_call.function
    function_name = function.name
    function_arguments = function_call.arguments

    yield TOOL_NAME, function_name

    if function_description := getattr(function, "description", None):
        yield TOOL_DESCRIPTION, function_description
    yield TOOL_PARAMETERS, safe_json_dumps(function_arguments)


def _input_value_and_mime_type_for_tool_span(
    arguments: Mapping[str, Any],
) -> Iterator[Tuple[str, Any]]:
    yield INPUT_MIME_TYPE, JSON
    yield INPUT_VALUE, safe_json_dumps(arguments)


def _output_value_and_mime_type_for_tool_span(result: Any) -> Iterator[Tuple[str, Any]]:
    yield OUTPUT_VALUE, str(result)
    yield OUTPUT_MIME_TYPE, TEXT


class _FunctionCallWrapper:
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

        function_call = instance
        function = function_call.function
        function_name = function.name
        function_arguments = function_call.arguments

        span_name = f"{function_name}"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                **dict(_input_value_and_mime_type_for_tool_span(function_arguments)),
                **dict(_function_call_attributes(function_call)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            response = wrapped(*args, **kwargs)

            if response.status == "success":
                function_result = function_call.result
                span.set_status(trace_api.StatusCode.OK)
                span.set_attributes(
                    dict(
                        _output_value_and_mime_type_for_tool_span(
                            result=function_result,
                        )
                    )
                )
            elif response.status == "failure":
                function_error_message = function_call.error
                span.set_status(trace_api.StatusCode.ERROR, function_error_message)
                span.set_attribute(OUTPUT_VALUE, function_error_message)
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)
            else:
                span.set_status(trace_api.StatusCode.ERROR, "Unknown function call status")

        return response

    async def arun(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        function_call = instance
        function = function_call.function
        function_name = function.name
        function_arguments = function_call.arguments

        span_name = f"{function_name}"

        with self._tracer.start_as_current_span(
            span_name,
            attributes={
                OPENINFERENCE_SPAN_KIND: TOOL,
                **dict(_input_value_and_mime_type_for_tool_span(function_arguments)),
                **dict(_function_call_attributes(function_call)),
                **dict(get_attributes_from_context()),
            },
        ) as span:
            response = await wrapped(*args, **kwargs)

            if response.status == "success":
                function_result = function_call.result
                span.set_status(trace_api.StatusCode.OK)
                span.set_attributes(
                    dict(
                        _output_value_and_mime_type_for_tool_span(
                            result=function_result,
                        )
                    )
                )
            elif response.status == "failure":
                function_error_message = function_call.error
                span.set_status(trace_api.StatusCode.ERROR, function_error_message)
                span.set_attribute(OUTPUT_VALUE, function_error_message)
                span.set_attribute(OUTPUT_MIME_TYPE, TEXT)
            else:
                span.set_status(trace_api.StatusCode.ERROR, "Unknown function call status")

        return response


# span attributes
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
SESSION_ID = SpanAttributes.SESSION_ID
LLM_TOOLS = SpanAttributes.LLM_TOOLS
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_FUNCTION_CALL = SpanAttributes.LLM_FUNCTION_CALL
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS
USER_ID = SpanAttributes.USER_ID
GRAPH_NODE_ID = SpanAttributes.GRAPH_NODE_ID
GRAPH_NODE_NAME = SpanAttributes.GRAPH_NODE_NAME
GRAPH_NODE_PARENT_ID = SpanAttributes.GRAPH_NODE_PARENT_ID

# message attributes
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS

# mime types
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value

# span kinds
AGENT = OpenInferenceSpanKindValues.AGENT.value
LLM = OpenInferenceSpanKindValues.LLM.value
TOOL = OpenInferenceSpanKindValues.TOOL.value

# tool attributes
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA

# tool call attributes
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID


