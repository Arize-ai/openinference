from enum import Enum
from typing import Any

from openinference.instrumentation.langchain._tracer import _serialize_enum_dict_keys


class PersonalInfoSubtype(Enum):
    EMAIL_ADDRESS = "email"
    PHONE_NUMBER = "phone"


def test_serialize_enum_dict_keys_replaces_enum_keys_with_values() -> None:
    """Test that Enum dictionary keys are replaced with their underlying values."""
    input_obj: dict[Any, Any] = {
        PersonalInfoSubtype.EMAIL_ADDRESS: "alice@example.com",
        "numbers_set": {1, 2, 3},
    }

    expected_obj: dict[str, Any] = {
        "email": "alice@example.com",
        "numbers_set": {1, 2, 3},
    }

    normalized_obj = _serialize_enum_dict_keys(input_obj)
    assert normalized_obj == expected_obj


def test_serialize_enum_dict_keys_returns_non_dict_unchanged() -> None:
    """Test that non-dictionary inputs are unchanged."""
    input_data = ["not", "a", "dict"]

    assert _serialize_enum_dict_keys(input_data) is input_data
