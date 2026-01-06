import json
from enum import Enum
from typing import Any

from openinference.instrumentation.langchain._tracer import _json_dumps


class TelemetryAttribute(Enum):
    USER_EMAIL = "user.email"
    REQUEST_ID = "request.id"


def test_json_dumps_converts_dict_keys_to_strings() -> None:
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

    result = _json_dumps(input_attributes)
    parsed = json.loads(result)

    assert parsed == expected_attributes


def test_json_dumps_handles_non_dict_input() -> None:
    """Handles non-dictionary inputs correctly."""
    non_dict_input = ["unexpected", "list", "value"]

    result = _json_dumps(non_dict_input)
    parsed = json.loads(result)

    assert parsed == non_dict_input
