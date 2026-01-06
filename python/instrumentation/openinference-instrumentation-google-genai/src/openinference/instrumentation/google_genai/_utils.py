import base64
import inspect
import logging
import warnings
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

from google.genai import types
from google.genai.types import Content, FunctionCall, FunctionResponse, Part, UserContent
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.google_genai._with_span import _WithSpan
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

T = TypeVar("T", bound=type)


class _ValueAndType(NamedTuple):
    value: str
    type: OpenInferenceMimeTypeValues


def _io_value_and_type(obj: Any) -> _ValueAndType:
    if hasattr(obj, "model_dump_json") and callable(obj.model_dump_json):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # `warnings=False` in `model_dump_json()` is only supported in Pydantic v2
                value = obj.model_dump_json(exclude_unset=True)
            assert isinstance(value, str)
        except Exception:
            logger.debug("Failed to get model dump json")
        else:
            return _ValueAndType(value, OpenInferenceMimeTypeValues.JSON)
    if not isinstance(obj, str) and isinstance(obj, (Sequence, Mapping)):
        try:
            value = safe_json_dumps(obj)
        except Exception:
            logger.debug("Failed to dump json")
        else:
            return _ValueAndType(value, OpenInferenceMimeTypeValues.JSON)

    return _ValueAndType(str(obj), OpenInferenceMimeTypeValues.TEXT)


def _as_input_attributes(
    value_and_type: Optional[_ValueAndType],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not value_and_type:
        return
    yield SpanAttributes.INPUT_VALUE, value_and_type.value
    # It's assumed to be TEXT by default, so we can skip to save one attribute.
    if value_and_type.type is not OpenInferenceMimeTypeValues.TEXT:
        yield SpanAttributes.INPUT_MIME_TYPE, value_and_type.type.value


def _as_output_attributes(
    value_and_type: Optional[_ValueAndType],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not value_and_type:
        return
    yield SpanAttributes.OUTPUT_VALUE, value_and_type.value
    # It's assumed to be TEXT by default, so we can skip to save one attribute.
    if value_and_type.type is not OpenInferenceMimeTypeValues.TEXT:
        yield SpanAttributes.OUTPUT_MIME_TYPE, value_and_type.type.value


def _finish_tracing(
    with_span: _WithSpan,
    attributes: Iterable[Tuple[str, AttributeValue]],
    extra_attributes: Iterable[Tuple[str, AttributeValue]],
    status: Optional[trace_api.Status] = None,
) -> None:
    try:
        attributes_dict = dict(attributes)
    except Exception:
        logger.debug("Failed to get attributes")
    try:
        extra_attributes_dict = dict(extra_attributes)
    except Exception:
        logger.debug("Failed to get extra attributes")
    try:
        with_span.finish_tracing(
            status=status,
            attributes=attributes_dict,
            extra_attributes=extra_attributes_dict,
        )
    except Exception:
        logger.debug("Failed to finish tracing")


def get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr_name, default)
    return getattr(obj, attr_name, default)


def _convert_automatic_function_to_schema(func: Callable[..., Any]) -> Dict[str, Any]:
    """Convert a Python function to a tool schema for automatic function calling."""
    import inspect
    from typing import get_type_hints

    # Get function metadata - cast to tell mypy that functions have these attributes
    func_name = cast(Any, func).__name__
    func_doc = cast(Any, func).__doc__ or ""

    # Get function signature
    sig = inspect.signature(func)
    type_hints = get_type_hints(func)

    # Build parameters schema
    parameters_schema = _build_parameters_schema(sig, type_hints)

    # Return in Google GenAI Tool format
    return {
        "function_declarations": [
            {
                "name": func_name,
                "description": func_doc.strip() or f"Function {func_name}",
                "parameters": parameters_schema,
            }
        ]
    }


def _convert_tool_to_dict(tool: Any) -> Dict[str, Any]:
    """Convert a Tool object to a dictionary representation."""
    try:
        # Handle Google GenAI Tool object
        if hasattr(tool, "function_declarations"):
            func_declarations = []
            for func_decl in tool.function_declarations:
                if hasattr(func_decl, "model_dump"):
                    func_declarations.append(func_decl.model_dump(exclude_none=True))
                else:
                    # Manually extract attributes
                    func_dict = {
                        "name": getattr(func_decl, "name", None),
                        "description": getattr(func_decl, "description", None),
                    }

                    # Handle parameters which might be a Schema object
                    parameters = getattr(func_decl, "parameters", None)
                    if parameters is not None:
                        if hasattr(parameters, "model_dump"):
                            func_dict["parameters"] = parameters.model_dump(exclude_none=True)
                        elif hasattr(parameters, "__dict__"):
                            func_dict["parameters"] = parameters.__dict__
                        else:
                            func_dict["parameters"] = parameters

                    # Remove None values
                    func_dict = {k: v for k, v in func_dict.items() if v is not None}
                    func_declarations.append(func_dict)

            return {"function_declarations": func_declarations}
        else:
            # Fallback: convert all attributes to dict
            return {k: v for k, v in tool.__dict__.items() if not k.startswith("_")}
    except Exception:
        logger.debug(f"Failed to convert tool to dict: {tool}")
        return {}


def _convert_to_flattened_format(func_decl: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Google GenAI function declaration to flattened format."""
    flattened_tool = {}

    # Copy name and description directly
    if "name" in func_decl:
        flattened_tool["name"] = func_decl["name"]
    if "description" in func_decl:
        flattened_tool["description"] = func_decl["description"]

    # flatten parameters
    if "parameters" in func_decl:
        flattened_tool["parameters"] = func_decl["parameters"]

    return flattened_tool


def _serialize_config_safely(config: Any) -> str:
    """
    Serialize a config object to JSON, handling Pydantic model classes
    that can't be directly serialized.
    """
    # Create a copy with serializable values - excluding tools which may contain a python
    # function object which will fail to serialize correctly later on
    config_dict = config.model_dump(exclude_none=True, exclude={"tools"})

    # Handle response_schema if it's a Pydantic model class
    if "response_schema" in config_dict:
        response_schema = config_dict["response_schema"]
        if hasattr(response_schema, "__pydantic_core_schema__"):
            # It's a Pydantic model class, convert to schema name or string representation
            config_dict["response_schema"] = getattr(
                response_schema, "__name__", str(response_schema)
            )
        elif hasattr(response_schema, "model_json_schema"):
            # It's a Pydantic model instance, use its schema
            config_dict["response_schema"] = response_schema.model_json_schema()

    return safe_json_dumps(config_dict)


def _build_parameters_schema(sig: inspect.Signature, type_hints: Dict[str, type]) -> Dict[str, Any]:
    """Build JSON schema for function parameters."""
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        # Determine if parameter is required (no default value)
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

        # Get parameter type information
        param_type = type_hints.get(param_name, str)
        param_info = {"type": _python_type_to_json_schema_type(param_type)}

        # Add description from docstring if available
        param_info["description"] = f"Parameter {param_name}"

        properties[param_name] = param_info

    # Build the schema
    parameters_schema = {"type": "object", "properties": properties}

    if required:
        parameters_schema["required"] = required

    return parameters_schema


def _python_type_to_json_schema_type(python_type: type) -> str:
    """Convert Python type to JSON schema type string."""
    if python_type is str:
        return "string"
    elif python_type is int:
        return "integer"
    elif python_type is float:
        return "number"
    elif python_type is bool:
        return "boolean"
    elif python_type is list:
        return "array"
    elif python_type is dict:
        return "object"
    else:
        # Default to string for unknown types
        return "string"


def _extract_tool_call_index(attr: str) -> int:
    """Extract tool call index from message tool call attribute key.

    Example: 'message.tool_calls.0.function_name' -> 0
    """
    parts = attr.split(".")
    if len(parts) >= 3 and parts[2].isdigit():
        return int(parts[2])
    return 0


def is_iterable_of(lst: Iterable[object], tp: T) -> bool:
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)


def _get_attributes_from_artifacts(
    inline_data: Any, tool_call_index: int
) -> Iterator[Tuple[str, AttributeValue]]:
    mime_type = get_attribute(inline_data, "mime_type")
    if (
        mime_type
        and "image" in mime_type
        and (data := get_attribute(inline_data, "data")) is not None
    ):
        prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.{tool_call_index}."
        image_url = f"data:{mime_type};base64,{base64.b64encode(data).decode()}"
        yield (
            f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}",
            image_url,
        )
        yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "image"


def _get_attributes_from_content_text(
    text: str, index: int, only_text: bool
) -> Iterator[Tuple[str, AttributeValue]]:
    if only_text:
        yield MessageAttributes.MESSAGE_CONTENT, text
    else:
        yield (
            f"{MessageAttributes.MESSAGE_CONTENTS}.{index}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
            text,
        )
        yield (
            f"{MessageAttributes.MESSAGE_CONTENTS}.{index}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
            "text",
        )


def _get_tools_from_config(config: Any) -> Iterator[Tuple[str, AttributeValue]]:
    """Extract tools from the GenerateContentConfig object."""
    if not hasattr(config, "tools") or not config.tools:
        return

    tools = config.tools
    if not isinstance(tools, list):
        return

    tool_index = 0  # Track across all function declarations

    for tool in tools:
        try:
            # Handle different types of tools
            if callable(tool):
                # Python function for automatic function calling
                tool_dict = _convert_automatic_function_to_schema(tool)
            elif hasattr(tool, "model_dump"):
                # Pydantic model
                tool_dict = tool.model_dump(exclude_none=True)
            elif hasattr(tool, "__dict__"):
                # Regular object with attributes
                tool_dict = _convert_tool_to_dict(tool)
            else:
                # Already a dict or other serializable format
                tool_dict = tool

            # Extract function declarations and create separate tool entries
            if isinstance(tool_dict, dict) and "function_declarations" in tool_dict:
                function_declarations = tool_dict["function_declarations"]
                if isinstance(function_declarations, list):
                    for func_decl in function_declarations:
                        # Convert Google GenAI format to flattened format
                        flattened_format = _convert_to_flattened_format(func_decl)
                        yield (
                            f"{SpanAttributes.LLM_TOOLS}.{tool_index}.{ToolAttributes.TOOL_JSON_SCHEMA}",
                            safe_json_dumps(flattened_format),
                        )
                        tool_index += 1
            else:
                # Tool doesn't have function_declarations, use as-is
                yield (
                    f"{SpanAttributes.LLM_TOOLS}.{tool_index}.{ToolAttributes.TOOL_JSON_SCHEMA}",
                    safe_json_dumps(tool_dict),
                )
                tool_index += 1

        except Exception:
            logger.debug(f"Failed to serialize tool: {tool}")


def _get_attributes_from_content(content: Content) -> Iterator[Tuple[str, AttributeValue]]:
    if role := get_attribute(content, "role"):
        yield (
            MessageAttributes.MESSAGE_ROLE,
            role.value if isinstance(role, Enum) else role,
        )
    else:
        yield (
            MessageAttributes.MESSAGE_ROLE,
            "user",
        )
    # Flatten parts into a single message content
    if parts := get_attribute(content, "parts"):
        yield from _flatten_parts(parts)


def _get_attributes_from_function_call(
    function_call: FunctionCall, tool_call_index: int
) -> Iterator[Tuple[str, AttributeValue]]:
    if name := get_attribute(function_call, "name"):
        if isinstance(name, str):
            yield (
                MessageAttributes.MESSAGE_TOOL_CALLS
                + f".{tool_call_index}."
                + ToolCallAttributes.TOOL_CALL_FUNCTION_NAME,
                name,
            )
    if args := get_attribute(function_call, "args"):
        yield (
            MessageAttributes.MESSAGE_TOOL_CALLS
            + f".{tool_call_index}."
            + ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
            safe_json_dumps(args),
        )

    if id := get_attribute(function_call, "id"):
        yield (
            MessageAttributes.MESSAGE_TOOL_CALLS
            + f".{tool_call_index}."
            + ToolCallAttributes.TOOL_CALL_ID,
            id,
        )


def _get_attributes_from_function_response(
    function_response: FunctionResponse,
) -> Iterator[Tuple[str, AttributeValue]]:
    if response := get_attribute(function_response, "response"):
        yield (MessageAttributes.MESSAGE_CONTENT, safe_json_dumps(response))
    if id := get_attribute(function_response, "id"):
        yield (
            MessageAttributes.MESSAGE_TOOL_CALL_ID,
            id,
        )


def _flatten_parts(parts: list[Part]) -> Iterator[Tuple[str, AttributeValue]]:
    for index, part in enumerate(parts):
        for attr, value in _get_attributes_from_part(part, index, len(parts) == 1):
            yield attr, value


def _get_attributes_from_part(
    part: Part, index: int, is_single_part: bool = False
) -> Iterator[Tuple[str, AttributeValue]]:
    # https://github.com/googleapis/python-genai/blob/main/google/genai/types.py#L566
    if text := get_attribute(part, "text"):
        yield from _get_attributes_from_content_text(text, index, is_single_part)
    elif function_call := get_attribute(part, "function_call"):
        yield from _get_attributes_from_function_call(function_call, index)
    elif function_response := get_attribute(part, "function_response"):
        yield from _get_attributes_from_function_response(function_response)
    if inline_data := get_attribute(part, "inline_data"):
        yield from _get_attributes_from_artifacts(inline_data, index)


def _get_attributes_from_message_param(
    input_contents: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    # https://github.com/googleapis/python-genai/blob/6e55222895a6639d41e54202e3d9a963609a391f/google/genai/models.py#L3890 # noqa: E501
    if isinstance(input_contents, str):
        # When provided a string, the GenAI SDK ingests it as
        # a UserContent object with role "user"
        # https://googleapis.github.io/python-genai/index.html#provide-a-string
        yield (MessageAttributes.MESSAGE_CONTENT, input_contents)
        yield (MessageAttributes.MESSAGE_ROLE, "user")
    elif isinstance(input_contents, Content) or isinstance(input_contents, UserContent):
        yield from _get_attributes_from_content(input_contents)
    elif isinstance(input_contents, Part):
        yield from _get_attributes_from_part(input_contents, 0, True)
    else:
        # TODO: Implement for File, PIL_Image
        logger.debug(f"Unexpected input contents type: {type(input_contents)}")


def _get_attributes_from_generate_content_usage(
    obj: types.GenerateContentResponseUsageMetadata,
) -> Iterator[Tuple[str, AttributeValue]]:
    if total := obj.total_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total
    if obj.prompt_tokens_details:
        prompt_details_audio = 0
        for modality_token_count in obj.prompt_tokens_details:
            if (
                modality_token_count.modality is types.MediaModality.AUDIO
                and modality_token_count.token_count
            ):
                prompt_details_audio += modality_token_count.token_count
        if prompt_details_audio:
            yield (
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO,
                prompt_details_audio,
            )
    if prompt := obj.prompt_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt
    if obj.candidates_tokens_details:
        completion_details_audio = 0
        for modality_token_count in obj.candidates_tokens_details:
            if (
                modality_token_count.modality is types.MediaModality.AUDIO
                and modality_token_count.token_count
            ):
                completion_details_audio += modality_token_count.token_count
        if completion_details_audio:
            yield (
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO,
                completion_details_audio,
            )
    completion = 0
    if candidates := obj.candidates_token_count:
        completion += candidates
    if thoughts := obj.thoughts_token_count:
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING, thoughts
        completion += thoughts
    if completion:
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion


def _get_attributes_from_automatic_function_calling_history(
    history: Iterable[object],
) -> Iterator[Tuple[str, AttributeValue]]:
    """Extract function call information from automatic_function_calling_history.

    This history contains the sequence of model->function call->function response
    that happened during automatic function calling.
    """
    tool_call_index = 0

    for content_entry in history:
        # Each entry is a Content object with parts
        if not hasattr(content_entry, "parts") or not hasattr(content_entry, "role"):
            continue

        # Look for model responses that contain function calls
        if getattr(content_entry, "role") == "model":
            parts = getattr(content_entry, "parts", [])
            for part in parts:
                if function_call := getattr(part, "function_call", None):
                    # Extract function call details for the span
                    yield from _get_attributes_from_function_call(function_call, tool_call_index)
                    tool_call_index += 1


def _get_attributes_from_generate_content(response: Any) -> Iterator[Tuple[str, AttributeValue]]:
    # https://github.com/googleapis/python-genai/blob/e9e84aa38726e7b65796812684d9609461416b11/google/genai/types.py#L2981  # noqa: E501
    if model_version := getattr(response, "model_version", None):
        yield SpanAttributes.LLM_MODEL_NAME, model_version
    if usage_metadata := getattr(response, "usage_metadata", None):
        yield from _get_attributes_from_generate_content_usage(usage_metadata)
    if (candidates := getattr(response, "candidates", None)) and isinstance(candidates, Iterable):
        index = -1
        for candidate in candidates:
            # TODO: This is a hack to get the index of the candidate.
            #       Might be a better way to do this.
            # Keep track of previous index to increment if index not found
            index = (
                index + 1 if getattr(candidate, "index") is None else getattr(candidate, "index")
            )
            if content := getattr(candidate, "content", None):
                for key, value in _get_attributes_from_message_param(content):
                    yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value

    # Handle automatic function calling history
    # For automatic function calling, the function call details are stored separately
    if automatic_history := getattr(response, "automatic_function_calling_history", None):
        yield from _get_attributes_from_automatic_function_calling_history(automatic_history)
