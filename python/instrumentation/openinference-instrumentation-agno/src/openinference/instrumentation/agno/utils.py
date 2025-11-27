from enum import Enum
from secrets import token_hex
from typing import Any, Iterator, Mapping, Optional, Tuple

from opentelemetry import context as context_api
from opentelemetry.util.types import AttributeValue

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
            if isinstance(value, Enum):
                value = value.value
            yield key, value


def _generate_node_id() -> str:
    return token_hex(8)  # Generates 16 hex characters (8 bytes)
