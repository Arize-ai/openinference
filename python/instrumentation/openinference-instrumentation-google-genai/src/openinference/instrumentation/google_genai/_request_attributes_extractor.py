import inspect
import logging
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Tuple, TypeVar, cast

from google.genai.types import Content, Part, UserContent
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.google_genai._utils import (
    _as_input_attributes,
    _io_value_and_type,
)
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
)

__all__ = ("_RequestAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _RequestAttributesExtractor:
    __slots__ = ()

    def get_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value
        yield SpanAttributes.LLM_PROVIDER, OpenInferenceLLMProviderValues.GOOGLE.value
        try:
            yield from _as_input_attributes(
                _io_value_and_type(request_parameters),
            )
        except Exception:
            logger.exception(
                f"Failed to get input attributes from request parameters of "
                f"type {type(request_parameters)}"
            )

        # Extract tools as high-priority attributes (avoid 128 attribute limit dropping them)
        if isinstance(request_parameters, Mapping):
            if config := request_parameters.get("config", None):
                yield from self._get_tools_from_config(config)

    def get_extra_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        # Start an index for the messages since we want to start with system instruction
        input_messages_index = 0
        if not isinstance(request_parameters, Mapping):
            return

        request_params_dict = dict(request_parameters)
        request_params_dict.pop("contents", None)  # Remove LLM input contents
        if config := request_params_dict.get("config", None):
            # config can either be a TypedDict or a pydantic object so we need to handle both cases
            if isinstance(config, dict):
                config_json = safe_json_dumps(config)
            else:
                config_json = self._serialize_config_safely(config)
            yield (
                SpanAttributes.LLM_INVOCATION_PARAMETERS,
                config_json,
            )

            # We push the system instruction to the first message for replay and consistency
            system_instruction = getattr(config, "system_instruction", None)
            if system_instruction:
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{MessageAttributes.MESSAGE_CONTENT}",
                    system_instruction,
                )
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{MessageAttributes.MESSAGE_ROLE}",
                    "system",
                )
                input_messages_index += 1

            # Tools are now extracted in get_attributes_from_request for higher priority

        if input_contents := request_parameters.get("contents"):
            if isinstance(input_contents, list):
                for input_content in input_contents:
                    for attr, value in self._get_attributes_from_message_param(input_content):
                        yield (
                            f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{attr}",
                            value,
                        )
                    # Move on to the next message
                    input_messages_index += 1
            else:
                for attr, value in self._get_attributes_from_message_param(input_contents):
                    # Default to index 0 for a single message
                    yield (
                        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{input_messages_index}.{attr}",
                        value,
                    )

    def _serialize_config_safely(self, config: Any) -> str:
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

    def _get_tools_from_config(self, config: Any) -> Iterator[Tuple[str, AttributeValue]]:
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
                    tool_dict = self._convert_automatic_function_to_schema(tool)
                elif hasattr(tool, "model_dump"):
                    # Pydantic model
                    tool_dict = tool.model_dump(exclude_none=True)
                elif hasattr(tool, "__dict__"):
                    # Regular object with attributes
                    tool_dict = self._convert_tool_to_dict(tool)
                else:
                    # Already a dict or other serializable format
                    tool_dict = tool

                # Extract function declarations and create separate tool entries
                if isinstance(tool_dict, dict) and "function_declarations" in tool_dict:
                    function_declarations = tool_dict["function_declarations"]
                    if isinstance(function_declarations, list):
                        for func_decl in function_declarations:
                            # Convert Google GenAI format to flattened format
                            flattened_format = self._convert_to_flattened_format(func_decl)
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
                logger.exception(f"Failed to serialize tool: {tool}")

    def _convert_to_flattened_format(self, func_decl: Dict[str, Any]) -> Dict[str, Any]:
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

    def _convert_automatic_function_to_schema(self, func: Callable[..., Any]) -> Dict[str, Any]:
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
        parameters_schema = self._build_parameters_schema(sig, type_hints)

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

    def _build_parameters_schema(
        self, sig: inspect.Signature, type_hints: Dict[str, type]
    ) -> Dict[str, Any]:
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
            param_info = {"type": self._python_type_to_json_schema_type(param_type)}

            # Add description from docstring if available
            param_info["description"] = f"Parameter {param_name}"

            properties[param_name] = param_info

        # Build the schema
        parameters_schema = {"type": "object", "properties": properties}

        if required:
            parameters_schema["required"] = required

        return parameters_schema

    def _python_type_to_json_schema_type(self, python_type: type) -> str:
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

    def _convert_tool_to_dict(self, tool: Any) -> Dict[str, Any]:
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
            logger.exception(f"Failed to convert tool to dict: {tool}")
            return {}

    def _get_attributes_from_message_param(
        self,
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
            yield from self._get_attributes_from_content(input_contents)
        elif isinstance(input_contents, Part):
            yield from self._get_attributes_from_part(input_contents)
        else:
            # TODO: Implement for File, PIL_Image
            logger.exception(f"Unexpected input contents type: {type(input_contents)}")

    def _get_attributes_from_content(
        self, content: Content
    ) -> Iterator[Tuple[str, AttributeValue]]:
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
            yield from self._flatten_parts(parts)

    def _flatten_parts(self, parts: list[Part]) -> Iterator[Tuple[str, AttributeValue]]:
        text_values = []
        for part in parts:
            for attr, value in self._get_attributes_from_part(part):
                if isinstance(value, str):
                    text_values.append(value)
            else:
                # TODO: Handle other types of parts
                logger.debug(f"Non-text part encountered: {part}")
        if text_values:
            yield (MessageAttributes.MESSAGE_CONTENT, "\n\n".join(text_values))

    def _get_attributes_from_part(self, part: Part) -> Iterator[Tuple[str, AttributeValue]]:
        # https://github.com/googleapis/python-genai/blob/main/google/genai/types.py#L566
        if text := get_attribute(part, "text"):
            yield (
                MessageAttributes.MESSAGE_CONTENT,
                text,
            )
        else:
            logger.exception("Other field types of parts are not supported yet")


T = TypeVar("T", bound=type)


def is_iterable_of(lst: Iterable[object], tp: T) -> bool:
    return isinstance(lst, Iterable) and all(isinstance(x, tp) for x in lst)


def get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr_name, default)
    return getattr(obj, attr_name, default)
