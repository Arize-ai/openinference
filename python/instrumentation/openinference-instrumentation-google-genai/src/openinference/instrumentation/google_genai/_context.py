"""ContextVar-based capture of SDK-transformed request parameters.

The Google GenAI SDK normalizes and serializes the full request (contents, tools,
system instruction, config) before passing it to `BaseApiClient.request` /
`request_streamed`. By monkey-patching those methods we capture the final,
fully serializable request dict.

The captured dict is used for: input_value, tools, invocation_parameters,
and embedding text.
"""

import logging
import re
from contextvars import ContextVar, Token
from typing import Any, Callable, Iterator, Mapping

from opentelemetry import context as context_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    SpanAttributes,
    ToolAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_UNSET = object()
_captured_request: ContextVar[dict[str, Any] | None] = ContextVar("_captured_request")


_CAMEL_TO_SNAKE_RE1 = re.compile(r"([A-Z]+)([A-Z][a-z])")
_CAMEL_TO_SNAKE_RE2 = re.compile(r"([a-z0-9])([A-Z])")


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s1 = _CAMEL_TO_SNAKE_RE1.sub(r"\1_\2", name)
    return _CAMEL_TO_SNAKE_RE2.sub(r"\1_\2", s1).lower()


def _convert_keys_to_snake(obj: Any) -> Any:
    """Recursively convert dict keys from camelCase to snake_case."""
    if isinstance(obj, dict):
        return {_camel_to_snake(k): _convert_keys_to_snake(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_keys_to_snake(item) for item in obj]
    return obj


class CapturedRequestScope:
    """Manages the lifecycle of the _captured_request ContextVar."""

    __slots__ = ("_reset_token",)

    def __init__(self) -> None:
        self._reset_token: Token[dict[str, Any] | None] | None = None

    def __enter__(self) -> "CapturedRequestScope":
        self._reset_token = _captured_request.set(None)  # None = active, awaiting capture
        return self

    def __exit__(self, *args: Any) -> None:
        if self._reset_token is not None:
            _captured_request.reset(self._reset_token)


def get_captured_request() -> dict[str, Any] | None:
    """Read captured request dict from ContextVar."""
    request = _captured_request.get(None)
    return request if isinstance(request, dict) else None


def get_input_attributes() -> Iterator[tuple[str, AttributeValue]]:
    """Read captured request and yield input value/mime type attributes."""
    request = _captured_request.get(None)
    if not isinstance(request, dict):
        return
    try:
        json_str = safe_json_dumps(request)
        yield SpanAttributes.INPUT_VALUE, json_str
        yield SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
    except Exception:
        logger.exception("Failed to serialize captured request as input value")


def get_tool_attributes() -> Iterator[tuple[str, AttributeValue]]:
    """Read captured request and yield tool span attributes."""
    request = _captured_request.get(None)
    if not isinstance(request, dict):
        return

    tools = request.get("tools")
    if not tools or not isinstance(tools, list):
        return

    tool_index = 0
    for tool in tools:
        try:
            if not isinstance(tool, dict):
                continue
            function_declarations = tool.get("function_declarations")
            if function_declarations and isinstance(function_declarations, list):
                for func_decl in function_declarations:
                    schema: dict[str, Any] = {}
                    if "name" in func_decl:
                        schema["name"] = func_decl["name"]
                    if "description" in func_decl:
                        schema["description"] = func_decl["description"]
                    if "parameters" in func_decl:
                        schema["parameters"] = func_decl["parameters"]
                    yield (
                        f"{SpanAttributes.LLM_TOOLS}.{tool_index}.{ToolAttributes.TOOL_JSON_SCHEMA}",
                        safe_json_dumps(schema),
                    )
                    tool_index += 1
            else:
                # Tool without function_declarations (e.g. code_execution)
                yield (
                    f"{SpanAttributes.LLM_TOOLS}.{tool_index}.{ToolAttributes.TOOL_JSON_SCHEMA}",
                    safe_json_dumps(tool),
                )
                tool_index += 1
        except Exception:
            logger.exception(f"Failed to extract tool attributes: {tool}")


def get_llm_invocation_parameters() -> str | None:
    """Extract generationConfig from the captured generate content request."""
    request = _captured_request.get(None)
    if not isinstance(request, dict):
        return None
    if config := request.get("generation_config"):
        return safe_json_dumps(config)
    return None


def get_embedding_invocation_parameters() -> str | None:
    """Extract embed config from the captured embed content request.

    The SDK flattens embed config fields into each request/instance entry:
      - Gemini API: {"requests": [{"content": ..., "taskType": ..., ...}]}
      - Vertex AI: {"instances": [{"content": ..., "task_type": ..., ...}]}
    """
    request = _captured_request.get(None)
    if not isinstance(request, dict):
        return None
    # Gemini API path
    if requests := request.get("requests"):
        if isinstance(requests, list) and requests:
            params = {k: v for k, v in requests[0].items() if k not in ("content", "model")}
            if params:
                return safe_json_dumps(params)
    # Vertex AI path
    if instances := request.get("instances"):
        if isinstance(instances, list) and instances:
            params = {k: v for k, v in instances[0].items() if k not in ("content",)}
            if params:
                return safe_json_dumps(params)
    return None


class _CapturedRequestWrapper:
    """wrapt-compatible wrapper for BaseApiClient.request / request_streamed.

    Captures the fully serialized request_dict into a ContextVar.
    Only writes if a CapturedRequestScope is active (value is None).
    """

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        # Only capture if a scope is active (set to None) and not already captured.
        # Without a scope, .get() returns _UNSET — no work to do.
        if _captured_request.get(_UNSET) is None:
            try:
                # request(http_method, path, request_dict, http_options=None)
                request_dict = args[2] if len(args) > 2 else kwargs.get("request_dict")
                if request_dict is not None and isinstance(request_dict, dict):
                    # Strip internal SDK keys (_url, _query) that are consumed
                    # before the HTTP request but not popped from the dict
                    cleaned = {k: v for k, v in request_dict.items() if not k.startswith("_")}
                    _captured_request.set(_convert_keys_to_snake(cleaned))
            except Exception:
                logger.exception("Failed to capture request dict")
        return wrapped(*args, **kwargs)
