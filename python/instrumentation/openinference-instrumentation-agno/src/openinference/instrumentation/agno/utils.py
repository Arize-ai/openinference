from collections import OrderedDict
from enum import Enum
from inspect import signature
from secrets import token_hex
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

from opentelemetry import context as context_api

# opentelemetry-api>=1.45.0 defines ``AttributeValue`` via a chained assignment
# (``AnyValue = AttributeValue = ...``), which mypy treats as a plain variable
# rather than a type alias, raising ``Variable ... is not valid as a type``.
# Define our own alias (equivalent to opentelemetry's pre-1.45 definition) so it
# can be used in type annotations regardless of the installed opentelemetry version.
AttributeValue = Union[
    str,
    bool,
    int,
    float,
    Sequence[str],
    Sequence[bool],
    Sequence[int],
    Sequence[float],
]

_AGNO_PARENT_NODE_CONTEXT_KEY = context_api.create_key("agno_parent_node_id")
# Marks that the current async task is already covered by a Workflow.arun span,
# so _WorkflowExecuteWrapper should not create its own span.
_AGNO_ARUN_SPANNED_CONTEXT_KEY = context_api.create_key("agno_workflow_arun_spanned")


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


def _generate_node_id() -> str:
    return token_hex(8)  # Generates 16 hex characters (8 bytes)


def _bind_arguments(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    method_signature = signature(method)
    bound_args = method_signature.bind(*args, **kwargs)
    bound_args.apply_defaults()
    arguments = bound_args.arguments
    arguments = OrderedDict(
        {key: value for key, value in arguments.items() if value is not None and value != {}}
    )
    return arguments
