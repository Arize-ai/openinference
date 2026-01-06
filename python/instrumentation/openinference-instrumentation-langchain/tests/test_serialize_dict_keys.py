from enum import Enum
from typing import Any

from openinference.instrumentation.langchain._tracer import _serialize_dict_keys


class TelemetryAttribute(Enum):
    USER_EMAIL = "user.email"
    REQUEST_ID = "request.id"


def test_serialize_dict_keys_converts_all_keys_to_strings() -> None:
    """Converts all dictionary keys to strings to ensure JSON compatibility."""
    input_attributes: dict[Any, Any] = {
        TelemetryAttribute.USER_EMAIL: "alice@example.com",
        TelemetryAttribute.REQUEST_ID: "req-123",
        404: "not_found",
        3.5: "latency_seconds",
        True: "is_cached",
        None: "unset_value",
        "already_string": "ok",
    }

    expected_attributes: dict[str, Any] = {
        str(TelemetryAttribute.USER_EMAIL): "alice@example.com",
        str(TelemetryAttribute.REQUEST_ID): "req-123",
        "404": "not_found",
        "3.5": "latency_seconds",
        "True": "is_cached",
        "None": "unset_value",
        "already_string": "ok",
    }

    serialized_attributes = _serialize_dict_keys(input_attributes)

    assert serialized_attributes == expected_attributes


def test_serialize_dict_keys_leaves_non_dict_input_unchanged() -> None:
    """Returns non-dictionary inputs without modification."""
    non_dict_input = ["unexpected", "list", "value"]

    assert _serialize_dict_keys(non_dict_input) is non_dict_input
