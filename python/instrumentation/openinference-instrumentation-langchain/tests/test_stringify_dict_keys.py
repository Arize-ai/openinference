from enum import Enum
from typing import Any

from openinference.instrumentation.langchain._tracer import _stringify_dict_keys


class PersonalInfoSubtype(Enum):
    EMAIL = 1
    PHONE = 2


def test_stringify_dict_keys_basic() -> None:
    """Test normalization of dict keys and containers."""
    input_obj: dict[Any, Any] = {
        PersonalInfoSubtype.EMAIL: "alice@example.com",
        "numbers_set": {1, 2, 3},
        "nested_items": [
            {("tuple_key", 1): "value"},
            (4, 5)
        ]
    }

    expected_obj: dict[str, Any] = {
        "PersonalInfoSubtype.EMAIL": "alice@example.com",
        "numbers_set": [1, 2, 3],
        "nested_items": [
            {"('tuple_key', 1)": "value"},
            (4, 5)
        ]
    }

    normalized_obj = _stringify_dict_keys(input_obj)
    assert normalized_obj == expected_obj


def test_stringify_dict_keys_additional() -> None:
    """Test normalization with frozenset keys, nested sets/tuples, and primitives."""
    input_obj: Any = {
        frozenset({1, 2}): {"inner_set": {3, 4}},
        "tuple_list": [(1, 2), {5, 6}],
        "primitive_value": 42
    }

    expected_obj: Any = {
        "frozenset({1, 2})": {"inner_set": [3, 4]},
        "tuple_list": [(1, 2), [5, 6]],
        "primitive_value": 42
    }

    normalized_obj = _stringify_dict_keys(input_obj)
    assert normalized_obj == expected_obj
