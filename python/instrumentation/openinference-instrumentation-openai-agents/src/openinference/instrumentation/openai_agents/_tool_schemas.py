"""Tool schemas for function spans.

``FunctionSpanData`` carries only a tool's name, input and output, so a function span has
no way of its own to report ``tool.description`` or ``tool.parameters``. Every other
OpenInference instrumentor reads those straight off the live tool object at the point the
tool is invoked, so this does the same: it wraps the SDK step that executes function tool
calls, which receives the ``FunctionTool`` objects, and publishes their schemas for the
duration of that step. The processor reads them back while ending each function span.

The schemas live in a ``ContextVar`` rather than on the processor because the execution
step owns the function spans it creates: the value cannot outlive the spans that need it,
so there is nothing to evict, cap, or clean up when a trace ends.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
from contextvars import ContextVar
from typing import Any, Callable, Mapping, Optional

from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

from openinference.instrumentation import safe_json_dumps

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# A tool's schema as recorded for a span: (description, serialized parameters).
ToolSchema = tuple[Optional[str], Optional[str]]

_EMPTY: Mapping[str, ToolSchema] = {}

_tool_schemas: ContextVar[Mapping[str, ToolSchema]] = ContextVar(
    "openinference_openai_agents_tool_schemas", default=_EMPTY
)

TOOL_EXECUTION_ATTRIBUTE = "execute_function_tool_calls"

# Modules that have defined, re-exported or called the step across SDK versions. Imported
# before scanning so that every binding is visible; the scan, not this list, decides what
# gets patched, so a module missing here only needs adding if nothing else imports it.
_TOOL_EXECUTION_MODULES: tuple[str, ...] = (
    # 0.18+, after the run internals were split up
    "agents.run_internal.tool_execution",
    "agents.run_internal.tool_planning",
    "agents.run_internal.run_loop",
    # 0.2.x through 0.17
    "agents._run_impl",
)


def find_tool_execution_bindings() -> list[tuple[Any, str]]:
    """Every place the SDK binds the function-tool execution step, as (owner, attribute).

    The step is imported by name at its call sites, so each importing module holds its own
    reference and patching only the defining module leaves the real caller untouched. The
    SDK has reorganised these modules more than once, so rather than track the list by
    hand, the known modules are imported and then every loaded ``agents`` module is scanned
    for a binding of the same underlying function. ``inspect.unwrap`` is used for the
    comparison so an already-patched binding still matches.
    """
    bindings: list[tuple[Any, str]] = []

    # Up to 0.17 the step was a classmethod, so the owner is the class rather than a module.
    try:
        run_impl = importlib.import_module("agents._run_impl")
        run_impl_class = getattr(run_impl, "RunImpl", None)
        if run_impl_class is not None and TOOL_EXECUTION_ATTRIBUTE in run_impl_class.__dict__:
            bindings.append((run_impl_class, TOOL_EXECUTION_ATTRIBUTE))
    except Exception:
        logger.debug("agents._run_impl.RunImpl not present", exc_info=True)

    canonical: Optional[Any] = None
    for module_name in _TOOL_EXECUTION_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if canonical is None and callable(found := getattr(module, TOOL_EXECUTION_ATTRIBUTE, None)):
            canonical = inspect.unwrap(found)

    if canonical is not None:
        for module_name, module in list(sys.modules.items()):
            if module is None or not module_name.startswith("agents."):
                continue
            try:
                bound = module.__dict__.get(TOOL_EXECUTION_ATTRIBUTE)
            except Exception:
                continue
            if bound is not None and inspect.unwrap(bound) is canonical:
                bindings.append((module, TOOL_EXECUTION_ATTRIBUTE))
    return bindings


def get_tool_schema(tool_name: str) -> Optional[ToolSchema]:
    """The schema of ``tool_name`` if it is among the tools currently being executed."""
    return _tool_schemas.get().get(tool_name)


def schemas_from_tool_runs(tool_runs: Any) -> dict[str, ToolSchema]:
    """Extract ``{tool name: schema}`` from the SDK's list of pending tool runs.

    Written defensively with ``getattr`` because it reads a private SDK dataclass: an
    unexpected shape should cost the two attributes, not raise inside a tool call.
    """
    schemas: dict[str, ToolSchema] = {}
    try:
        iterator = iter(tool_runs or ())
    except TypeError:
        return schemas
    for tool_run in iterator:
        tool = getattr(tool_run, "function_tool", None)
        if not isinstance(name := getattr(tool, "name", None), str):
            continue
        description = getattr(tool, "description", None)
        # The SDK's own FunctionTool calls this params_json_schema; the OpenAI Responses
        # type of the same name calls it parameters. Accept either.
        parameters = getattr(tool, "params_json_schema", None)
        if parameters is None:
            parameters = getattr(tool, "parameters", None)
        schemas[name] = (
            description if isinstance(description, str) else None,
            safe_json_dumps(parameters) if parameters is not None else None,
        )
    return schemas


def make_execute_function_tools_wrapper() -> Callable[..., Any]:
    """Wrapper publishing the schemas of the tools a run step is about to execute."""

    async def wrapper(
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Everything below is enrichment for spans the processor is about to end, so
        # under suppression there is nothing to enrich: OITracer hands out a
        # non-recording span, which drops these attributes anyway. Skipping saves
        # serializing every tool's schema for a span that will discard it.
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        try:
            schemas = schemas_from_tool_runs(kwargs.get("tool_runs"))
        except Exception:
            logger.debug("could not read tool schemas from tool_runs", exc_info=True)
            schemas = {}
        if not schemas:
            return await wrapped(*args, **kwargs)
        # Merged over any enclosing step so that a nested run, as created by an agent
        # exposed as a tool, does not hide the outer step's tools.
        token = _tool_schemas.set({**_tool_schemas.get(), **schemas})
        try:
            return await wrapped(*args, **kwargs)
        finally:
            _tool_schemas.reset(token)

    return wrapper
