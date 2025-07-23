# ruff: noqa: E501
import dataclasses
import datetime
import json
from enum import Enum
from typing import Any, Optional, Union

import pytest
from pydantic import BaseModel

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.langchain._tracer import _convert_io
from openinference.semconv.trace import OpenInferenceMimeTypeValues


class TestConvertIO:
    """
    Test cases for the _convert_io function.

    This function converts input/output data for OpenInference spans with specific behavior:

    1. String optimization: Single string values return directly (no MIME type)
       - Performance optimization for common LangChain cases like {"input": "user message"}

    2. Input/output keys: Use _json_dumps with conditional MIME type
       - Only structured data (objects/arrays) get JSON MIME type
       - Primitives (numbers, booleans) return without MIME type to reduce clutter

    3. General cases: Always use _json_dumps with JSON MIME type
       - Multiple keys or non-input/output keys are always structured objects
       - Always include MIME type since these represent complex data

    4. JSON validity: All output with JSON MIME type must be parseable with json.loads()
       - Ensures consistency between MIME type declaration and content format

    5. Advanced data type support: Comprehensive handling of Python types
       - Enums: serialized by underlying value
       - Dataclasses: field-based serialization with optional None field filtering
       - Date/time objects: ISO format strings for standardization
       - Timedeltas: total seconds for numeric representation
       - Pydantic models: via model_dump_json() for framework integration
    """

    @pytest.mark.parametrize(
        "input_obj,expected_output",
        [
            pytest.param(
                None,
                [],
                id="none_input",
            ),
            pytest.param(
                {},
                [],
                id="empty_dict",
            ),
        ],
    )
    def test_convert_io_empty_cases(
        self,
        input_obj: Optional[dict[str, Any]],
        expected_output: list[str],
    ) -> None:
        """Test that None and empty dict return empty iterators."""
        result = list(_convert_io(input_obj))
        assert result == expected_output

    @pytest.mark.parametrize(
        "input_obj,expected_output",
        [
            pytest.param(
                {"key": "simple string"},
                ["simple string"],
                id="single_string_value",
            ),
            pytest.param(
                {"message": "Hello, world!"},
                ["Hello, world!"],
                id="single_string_message",
            ),
            pytest.param(
                {"text": ""},
                [""],
                id="single_empty_string",
            ),
        ],
    )
    def test_convert_io_single_string_value(
        self,
        input_obj: dict[str, str],
        expected_output: list[str],
    ) -> None:
        """Test that single string values are returned as-is."""
        result = list(_convert_io(input_obj))
        assert result == expected_output

    @pytest.mark.parametrize(
        "input_obj,expected_first_item",
        [
            pytest.param(
                {"input": {"nested": "data"}},
                '{"nested": "data"}',
                id="input_key_nested_dict",
            ),
            pytest.param(
                {"output": [1, 2, 3]},
                "[1, 2, 3]",
                id="output_key_list",
            ),
            pytest.param(
                {"input": (1, 2, 3)},
                "[1, 2, 3]",
                id="input_key_tuple",
            ),
            pytest.param(
                {"input": {"name": "test", "value": 123}},
                '{"name": "test", "value": 123}',
                id="input_key_complex_dict",
            ),
        ],
    )
    def test_convert_io_input_output_keys(
        self,
        input_obj: dict[str, Any],
        expected_first_item: str,
    ) -> None:
        """Test that input/output keys trigger JSON serialization with mime type when conditions are met."""
        result = list(_convert_io(input_obj))
        assert len(result) == 2
        assert result[0] == expected_first_item
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

        # Verify arrays are valid JSON
        if expected_first_item.startswith("["):
            # Arrays from _json_dumps should be valid JSON
            parsed = json.loads(expected_first_item)
            assert isinstance(parsed, list)
        elif expected_first_item.startswith("{"):
            # Objects from _json_dumps should be valid JSON
            parsed = json.loads(expected_first_item)
            assert isinstance(parsed, dict)

    @pytest.mark.parametrize(
        "input_obj,expected_first_item",
        [
            pytest.param(
                {"input": 42},
                "42",
                id="input_key_number",
            ),
            pytest.param(
                {"output": True},
                "true",
                id="output_key_boolean",
            ),
            pytest.param(
                {"input": ["simple", "string"]},
                '["simple", "string"]',
                id="input_key_string_array",
            ),
            pytest.param(
                {"output": float("nan")},
                "null",
                id="output_key_nan",
            ),
            pytest.param(
                {"input": float("inf")},
                "null",
                id="input_key_infinity",
            ),
        ],
    )
    def test_convert_io_input_output_keys_all_types(
        self, input_obj: dict[str, Any], expected_first_item: str
    ) -> None:
        """Test that input/output keys trigger _json_dumps, but only complex types get MIME type."""
        result = list(_convert_io(input_obj))
        # Only arrays get MIME type (start with '['), simple values don't
        if expected_first_item.startswith("["):
            assert len(result) == 2, (
                f"Expected 2 items for {input_obj}, got {len(result)} items: {result}"
            )
            assert result[0] == expected_first_item, (
                f"Expected '{expected_first_item}', got '{result[0]}' for input {input_obj}"
            )
            assert result[1] == OpenInferenceMimeTypeValues.JSON.value, (
                f"Expected JSON MIME type, got '{result[1]}' for input {input_obj}"
            )
            # Verify arrays are valid JSON
            parsed = json.loads(expected_first_item)
            assert isinstance(parsed, list)
        else:
            # Simple values (numbers, booleans, null) don't get MIME type
            assert len(result) == 1, (
                f"Expected 1 item for {input_obj}, got {len(result)} items: {result}"
            )
            assert result[0] == expected_first_item, (
                f"Expected '{expected_first_item}', got '{result[0]}' for input {input_obj}"
            )
            # Verify simple JSON values can be parsed
            if (
                expected_first_item in ("null", "true", "false")
                or expected_first_item.replace(".", "").replace("-", "").isdigit()
            ):
                # Values should already be in JSON format
                parsed = json.loads(expected_first_item)
                assert parsed is not None or expected_first_item == "null"

    @pytest.mark.parametrize(
        "input_obj",
        [
            pytest.param(
                {"key": 123},
                id="single_non_string_value",
            ),
            pytest.param(
                {"key": [1, 2, 3]},
                id="single_list_value",
            ),
            pytest.param(
                {"key": {"nested": "value"}},
                id="single_dict_value",
            ),
            pytest.param(
                {"key1": "value1", "key2": "value2"},
                id="multiple_string_values",
            ),
            pytest.param(
                {"name": "test", "age": 25, "active": True},
                id="multiple_mixed_values",
            ),
        ],
    )
    def test_convert_io_non_special_cases(self, input_obj: dict[str, Any]) -> None:
        """Test that non-special cases return _json_dumps serialization with mime type."""
        result = list(_convert_io(input_obj))
        assert len(result) == 2

        # First item should be _json_dumps serialization of the input
        # Note: _json_dumps now produces valid JSON with quoted keys
        json_output = result[0]
        # Verify the structure is valid JSON
        assert json_output.startswith("{")
        assert json_output.endswith("}")
        # Verify it's actually valid JSON
        parsed = json.loads(json_output)
        assert isinstance(parsed, dict)

        # Second item should be JSON mime type
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

    @pytest.mark.parametrize(
        "input_obj,expected_json_contains",
        [
            pytest.param(
                {"value": float("nan")},
                '"value": null',
                id="nan_float_value",
            ),
            pytest.param(
                {"normal": 1.5, "infinite": float("inf"), "negative_inf": float("-inf")},
                '"infinite": null',
                id="mixed_finite_infinite_values",
            ),
            pytest.param(
                {"data": {"nested_nan": float("nan"), "normal": 42}},
                '"normal": 42',
                id="nested_nan_values",
            ),
        ],
    )
    def test_convert_io_nan_replacement(
        self, input_obj: dict[str, Any], expected_json_contains: str
    ) -> None:
        """Test that NaN and infinite float values are converted to null via _json_dumps."""
        result = list(_convert_io(input_obj))
        assert len(result) == 2

        # Verify that the expected content is in the JSON output
        json_output = result[0]
        assert expected_json_contains in json_output

        # Second item should be JSON mime type
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

    def test_convert_io_complex_object(self) -> None:
        """Test with a complex nested object structure."""
        complex_obj = {
            "user": {
                "name": "John Doe",
                "preferences": ["reading", "swimming"],
                "metadata": {"score": 85.5, "active": True, "nan_value": float("nan")},
            },
            "session": "abc123",
        }

        result = list(_convert_io(complex_obj))
        assert len(result) == 2

        # Verify it contains expected structure (now goes through _json_dumps)
        json_output = result[0]
        assert '"name": "John Doe"' in json_output, f"Expected name field in: {json_output}"
        assert '"reading"' in json_output, f"Expected reading in: {json_output}"
        assert '"swimming"' in json_output, f"Expected swimming in: {json_output}"
        assert '"score": 85.5' in json_output, f"Expected score field in: {json_output}"
        assert '"active": true' in json_output, f"Expected active: true in: {json_output}"
        assert '"session": "abc123"' in json_output, f"Expected session field in: {json_output}"
        # NaN values are now converted to null by _json_dumps
        assert '"nan_value": null' in json_output, f"Expected nan_value: null in: {json_output}"

        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

    def test_convert_io_with_model_dump_json(self) -> None:
        """Test with an object that has model_dump_json method."""

        class ModelInfo(BaseModel):
            model: str
            version: str

        model_info = ModelInfo(model="test", version="1.0")
        # Use 'input' key to trigger _json_dumps path instead of safe_json_dumps
        input_obj = {"input": model_info}
        result = list(_convert_io(input_obj))

        assert len(result) == 2
        # The _json_dumps function should use model_dump_json for the Pydantic model
        # Pydantic's model_dump_json() returns compact JSON without spaces
        json_output = result[0]
        assert '"model":"test"' in json_output
        assert '"version":"1.0"' in json_output
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

        # Verify Pydantic model produces valid JSON
        parsed = json.loads(json_output)
        assert isinstance(parsed, dict)
        assert parsed["model"] == "test"
        assert parsed["version"] == "1.0"

    def test_convert_io_pydantic_with_input_output_keys(self) -> None:
        """Test that _json_dumps properly handles Pydantic models via input/output keys."""

        class ComplexModel(BaseModel):
            name: str
            metadata: dict[str, Any]
            values: list[int]

        model = ComplexModel(
            name="test_model", metadata={"type": "example", "version": 2}, values=[1, 2, 3]
        )

        # Test with 'input' key to trigger _json_dumps path
        input_obj = {"input": model}
        result = list(_convert_io(input_obj))

        assert len(result) == 2
        json_output = result[0]

        # Pydantic model_dump_json() returns compact JSON
        assert '"name":"test_model"' in json_output
        assert '"type":"example"' in json_output
        assert '"version":2' in json_output
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

        # Verify Pydantic model produces valid JSON
        parsed = json.loads(json_output)
        assert isinstance(parsed, dict)
        assert parsed["name"] == "test_model"
        assert parsed["metadata"]["type"] == "example"
        assert parsed["values"] == [1, 2, 3]

    def test_convert_io_edge_case_special_key_non_string(self) -> None:
        """Test edge case where input/output key has non-string value that becomes string."""
        input_obj = {"other_key": 42}  # Not input/output key, single non-string value
        result = list(_convert_io(input_obj))

        assert len(result) == 2
        # Should go through the general case since it's not input/output key
        # Now uses _json_dumps with proper JSON format
        expected_json = '{"other_key": 42}'
        assert result[0] == expected_json
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

    @pytest.mark.parametrize(
        "input_obj,expected_contains",
        [
            pytest.param(
                {"input": {"nested": {"deep_nan": float("nan"), "value": 42}}},
                ['"deep_nan": null', '"value": 42'],
                id="nested_nan_via_json_dumps",
            ),
            pytest.param(
                {"output": {"data": [1, float("inf"), 3]}},
                ['"data": [1, null, 3]'],
                id="nan_in_array_via_json_dumps",
            ),
        ],
    )
    def test_convert_io_nan_replacement_in_json_dumps_path(
        self, input_obj: dict[str, Any], expected_contains: list[str]
    ) -> None:
        """Test NaN handling works in _json_dumps path (input/output keys) converting to null."""
        result = list(_convert_io(input_obj))
        assert len(result) == 2

        json_output = result[0]
        for expected in expected_contains:
            assert expected in json_output
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

        # Verify arrays are valid JSON
        if json_output.startswith("["):
            parsed = json.loads(json_output)
            assert isinstance(parsed, list)
        elif json_output.startswith("{"):
            parsed = json.loads(json_output)
            assert isinstance(parsed, dict)

    @pytest.mark.parametrize(
        "input_obj,expected_output",
        [
            pytest.param(
                {"input": []},
                "[]",
                id="empty_list",
            ),
            pytest.param(
                {"output": {}},
                "{}",
                id="empty_dict",
            ),
            pytest.param(
                {"input": [[], {}, ""]},
                '[[], {}, ""]',
                id="nested_empty_collections",
            ),
        ],
    )
    def test_convert_io_empty_collections_via_json_dumps(
        self, input_obj: dict[str, Any], expected_output: str
    ) -> None:
        """Test empty collections through _json_dumps path."""
        result = list(_convert_io(input_obj))
        assert len(result) == 2
        assert result[0] == expected_output
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

        # Verify the result is valid JSON
        parsed = json.loads(expected_output)
        assert parsed is not None

    def test_convert_io_bytes_handling(self) -> None:
        """Test that bytes with input/output keys go through _json_dumps."""
        byte_data = b"hello world"
        input_obj = {"input": byte_data}
        result = list(_convert_io(input_obj))

        # Bytes fallback to safe_json_dumps, which doesn't start with { or [, so no MIME type
        assert len(result) == 1
        assert result[0] == safe_json_dumps(byte_data)

    @pytest.mark.parametrize(
        "iterable_obj,expected_contains",
        [
            pytest.param(
                (1, 2, 3),
                "[1, 2, 3]",
                id="tuple",
            ),
        ],
    )
    def test_convert_io_other_iterables_via_json_dumps(
        self, iterable_obj: Any, expected_contains: str
    ) -> None:
        """Test various iterable types through _json_dumps."""
        input_obj = {"input": iterable_obj}
        result = list(_convert_io(input_obj))

        assert len(result) == 2
        json_output = result[0]
        assert expected_contains == json_output
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

        # Verify arrays are valid JSON
        if json_output.startswith("["):
            parsed = json.loads(json_output)
            assert isinstance(parsed, list)
        elif json_output.startswith("{"):
            parsed = json.loads(json_output)
            assert isinstance(parsed, dict)

    def test_convert_io_set_via_json_dumps(self) -> None:
        """Test that sets with input/output keys go through _json_dumps and get converted to arrays."""
        set_obj = {1, 2, 3}
        input_obj = {"input": set_obj}
        result = list(_convert_io(input_obj))

        assert len(result) == 2
        # With simplified condition, sets go through _json_dumps and become arrays
        json_output = result[0]
        assert json_output.startswith("[")
        assert json_output.endswith("]")
        # Check all elements are present (order may vary for sets)
        for item in set_obj:
            assert str(item) in json_output
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

        # Verify the array is valid JSON
        parsed = json.loads(json_output)
        assert isinstance(parsed, list)
        assert set(parsed) == set_obj  # Compare as sets since order may vary

    def test_convert_io_deeply_nested_structure(self) -> None:
        """Test deeply nested structures with various data types."""
        complex_obj = {
            "input": {
                "level1": {
                    "level2": {
                        "numbers": [1, 2.5, float("nan")],
                        "strings": ["a", "b"],
                        "mixed": {"key": "value", "count": 42, "invalid": float("inf")},
                    }
                },
                "other": [{"nested": True}, {"nested": False}],
            }
        }

        result = list(_convert_io(complex_obj))
        assert len(result) == 2

        json_output = result[0]
        # Verify structure (uses _json_dumps for dict)
        assert '"level1":' in json_output, f"Expected level1 in: {json_output}"
        assert '"level2":' in json_output, f"Expected level2 in: {json_output}"
        assert '"numbers":' in json_output, f"Expected numbers in: {json_output}"
        assert '"strings":' in json_output, f"Expected strings in: {json_output}"
        assert '"mixed":' in json_output, f"Expected mixed in: {json_output}"

        # Verify proper formatting (quoted keys in valid JSON)
        assert '"key": "value"' in json_output, f"Expected key-value pair in: {json_output}"
        assert '"count": 42' in json_output, f"Expected count field in: {json_output}"

        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

    def test_convert_io_custom_object_fallback(self) -> None:
        """Test that custom objects with input/output keys go through _json_dumps and fallback to safe_json_dumps."""

        class CustomObject:
            def __init__(self, value: str):
                self.value = value

            def __str__(self) -> str:
                return f"CustomObject({self.value})"

        custom_obj = CustomObject("test")
        input_obj = {"input": custom_obj}
        result = list(_convert_io(input_obj))

        # Custom objects fallback to safe_json_dumps, which doesn't start with { or [, so no MIME type
        assert len(result) == 1
        assert result[0] == safe_json_dumps(custom_obj)

    def test_convert_io_string_vs_bytes_distinction(self) -> None:
        """Test how strings and bytes are handled with current input/output conditions."""
        # Test string - single string value gets returned directly (special case)
        input_obj = {"input": "hello"}
        result = list(_convert_io(input_obj))
        # Single string value gets returned directly
        assert result[0] == "hello"

        # Test bytes - goes through _json_dumps and falls back to safe_json_dumps
        byte_obj = {"input": b"hello"}
        result = list(_convert_io(byte_obj))
        expected_json = safe_json_dumps(b"hello")
        # Bytes don't start with { or [, so no MIME type
        assert len(result) == 1
        assert result[0] == expected_json

        # Test that dict with string value still works
        string_dict = {"input": {"key": "abc"}}  # This should trigger _json_dumps
        result = list(_convert_io(string_dict))
        # Objects start with {, so they get MIME type
        assert len(result) == 2
        assert result[0] == '{"key": "abc"}'
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

    def test_convert_io_simplified_condition_logic(self) -> None:
        """
        Test the core behavior patterns of _convert_io function.

        This test validates the three main code paths:
        1. String optimization: Single strings return directly (1 item, no MIME type)
        2. Input/output conditionals: Structured data gets MIME type, primitives don't
        3. General case: Complex objects always get MIME type

        The different behaviors ensure optimal performance for common cases while
        providing proper metadata for structured data.
        """

        # Category 1: String optimization cases (single string values)
        # These bypass JSON processing entirely for performance
        direct_string_cases: list[tuple[dict[str, Any], str]] = [
            ({"input": "hello"}, "hello"),
            ({"output": "world"}, "world"),
        ]

        # Category 2a: Input/output keys with structured data (SHOULD get MIME type)
        # These are complex data structures that benefit from MIME type indication
        complex_cases: list[tuple[dict[str, Any], str]] = [
            ({"input": []}, "[]"),
            ({"input": {}}, "{}"),
            ({"input": (1, 2)}, "[1, 2]"),
            ({"input": {"key": "value"}}, '{"key": "value"}'),
            ({"output": [1, 2, 3]}, "[1, 2, 3]"),
        ]

        # Category 2b: Input/output keys with primitive data (should NOT get MIME type)
        # These are simple values where MIME type would add unnecessary overhead
        simple_cases: list[tuple[dict[str, Any], str]] = [
            ({"input": 42}, "42"),
            ({"input": True}, "true"),
            ({"input": float("nan")}, "null"),
            ({"output": float("inf")}, "null"),
        ]

        # Test complex cases (should get MIME type)
        for input_obj, expected in complex_cases:
            result = list(_convert_io(input_obj))
            assert len(result) == 2, (
                f"Expected 2 items for {input_obj}, got {len(result)} items: {result}"
            )
            assert result[0] == expected, (
                f"Expected '{expected}', got '{result[0]}' for input {input_obj}"
            )
            assert result[1] == OpenInferenceMimeTypeValues.JSON.value, (
                f"Expected JSON MIME type, got '{result[1]}' for input {input_obj}"
            )

            # Verify arrays are valid JSON
            if expected.startswith("["):
                parsed = json.loads(expected)
                assert isinstance(parsed, list)
            elif expected.startswith("{"):
                parsed = json.loads(expected)
                assert isinstance(parsed, dict)

        # Test simple cases (should NOT get MIME type)
        for input_obj, expected in simple_cases:
            result = list(_convert_io(input_obj))
            assert len(result) == 1, (
                f"Expected 1 item for {input_obj}, got {len(result)} items: {result}"
            )
            assert result[0] == expected, (
                f"Expected '{expected}', got '{result[0]}' for input {input_obj}"
            )

        # Test direct string cases (single string values)
        for input_obj, expected in direct_string_cases:
            result = list(_convert_io(input_obj))
            assert len(result) == 1, (
                f"Expected 1 item for {input_obj}, got {len(result)} items: {result}"
            )  # Direct string return, no mime type
            assert result[0] == expected, (
                f"Expected '{expected}', got '{result[0]}' for input {input_obj}"
            )

        # Category 3: General cases (non-input/output keys or multiple keys)
        # These always get MIME type since they represent structured objects/metadata
        general_cases: list[dict[str, Any]] = [
            {"data": 42},
            {"metadata": {"type": "test"}},  # Changed from string to avoid optimization
            {"key1": "value1", "key2": "value2"},  # Multiple keys
        ]

        for input_obj in general_cases:
            result = list(_convert_io(input_obj))
            assert len(result) == 2
            # Now uses _json_dumps with quoted keys instead of safe_json_dumps
            json_output = result[0]
            assert json_output.startswith("{")
            assert json_output.endswith("}")
            # Verify it's actually valid JSON
            parsed = json.loads(json_output)
            assert isinstance(parsed, dict)
            assert result[1] == OpenInferenceMimeTypeValues.JSON.value

    def test_convert_io_improved_nan_handling(self) -> None:
        """
        Test that NaN handling is now built into _json_dumps and converts to null.

        Design rationale: Converting NaN/infinity to null ensures valid JSON while
        maintaining clear indication that the original value was problematic.
        This approach provides better interoperability with downstream systems
        compared to preserving NaN as quoted strings or throwing errors.
        """

        # Test NaN handling in various contexts through input/output keys
        test_cases: list[tuple[dict[str, Any], str]] = [
            # Direct NaN values - now null
            ({"input": float("nan")}, "null"),
            ({"output": float("inf")}, "null"),
            ({"input": float("-inf")}, "null"),
            # NaN in arrays - null in arrays
            ({"input": [1, float("nan"), 3]}, "[1, null, 3]"),
            ({"output": [float("inf"), 2.5]}, "[null, 2.5]"),
            # NaN in nested objects - null in objects
            ({"input": {"valid": 42, "invalid": float("nan")}}, '{"valid": 42, "invalid": null}'),
            ({"output": {"data": [1, float("-inf")]}}, '{"data": [1, null]}'),
        ]

        for input_obj, expected in test_cases:
            result = list(_convert_io(input_obj))
            if expected.startswith("{") or expected.startswith("["):
                # Objects and arrays get MIME type
                assert len(result) == 2
                assert result[0] == expected
                assert result[1] == OpenInferenceMimeTypeValues.JSON.value

                # Verify arrays are valid JSON (objects have unquoted keys)
                if expected.startswith("["):
                    parsed = json.loads(expected)
                    assert isinstance(parsed, list)
                elif expected.startswith("{"):
                    parsed = json.loads(expected)
                    assert isinstance(parsed, dict)
            else:
                # Simple values (null) don't get MIME type
                assert len(result) == 1
                assert result[0] == expected

    def test_convert_io_enum_support(self) -> None:
        """Test that Enum objects are serialized by their underlying value."""

        class StatusEnum(Enum):
            PENDING = "pending"
            COMPLETED = "completed"
            FAILED = 42

        # Test string enum value
        string_enum_obj: dict[str, Any] = {"input": StatusEnum.PENDING}
        result = list(_convert_io(string_enum_obj))
        assert len(result) == 1, (
            f"Expected 1 item for string enum, got {len(result)} items: {result}"
        )
        assert result[0] == '"pending"', f"Expected quoted string for enum value, got {result[0]}"

        # Test integer enum value
        int_enum_obj: dict[str, Any] = {"input": StatusEnum.FAILED}
        result = list(_convert_io(int_enum_obj))
        assert len(result) == 1, f"Expected 1 item for int enum, got {len(result)} items: {result}"
        assert result[0] == "42", f"Expected number string for enum value, got {result[0]}"

        # Test enum in complex structure
        complex_obj: dict[str, Any] = {"data": {"status": StatusEnum.COMPLETED, "count": 5}}
        result = list(_convert_io(complex_obj))
        assert len(result) == 2, f"Expected 2 items for complex object, got {len(result)}"

        # Verify the JSON is valid and contains enum value
        parsed = json.loads(result[0])
        assert parsed["data"]["status"] == "completed"
        assert parsed["data"]["count"] == 5

    def test_convert_io_dataclass_support(self) -> None:
        """Test that dataclass objects are serialized with field-based JSON."""

        @dataclasses.dataclass
        class PersonInfo:
            name: str
            age: int
            email: Optional[str] = None
            active: bool = True

        # Test dataclass with all fields
        person = PersonInfo(name="John Doe", age=30, email="john@example.com")
        full_person_obj: dict[str, Any] = {"input": person}
        result = list(_convert_io(full_person_obj))

        assert len(result) == 2, (
            f"Expected 2 items for dataclass, got {len(result)} items: {result}"
        )
        json_output = result[0]
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

        # Verify the JSON is valid and contains all fields
        parsed = json.loads(json_output)
        assert parsed["name"] == "John Doe"
        assert parsed["age"] == 30
        assert parsed["email"] == "john@example.com"
        assert parsed["active"] is True

        # Test dataclass with optional None field (should be excluded)
        person_no_email = PersonInfo(name="Jane Doe", age=25)
        minimal_person_obj: dict[str, Any] = {"input": person_no_email}
        result = list(_convert_io(minimal_person_obj))

        json_output = result[0]
        parsed = json.loads(json_output)
        assert "email" not in parsed, f"Expected None optional field to be excluded, got {parsed}"
        assert parsed["name"] == "Jane Doe"
        assert parsed["age"] == 25
        assert parsed["active"] is True

    def test_convert_io_datetime_support(self) -> None:
        """Test that date/time objects are serialized as ISO format strings."""

        # Test datetime
        dt = datetime.datetime(2023, 12, 25, 15, 30, 45)
        dt_input_obj: dict[str, Any] = {"input": dt}
        result = list(_convert_io(dt_input_obj))
        assert len(result) == 1, f"Expected 1 item for datetime, got {len(result)} items: {result}"
        assert result[0] == '"2023-12-25T15:30:45"', (
            f"Expected ISO format datetime, got {result[0]}"
        )

        # Test date
        d = datetime.date(2023, 12, 25)
        date_input_obj: dict[str, Any] = {"input": d}
        result = list(_convert_io(date_input_obj))
        assert len(result) == 1, f"Expected 1 item for date, got {len(result)} items: {result}"
        assert result[0] == '"2023-12-25"', f"Expected ISO format date, got {result[0]}"

        # Test time
        t = datetime.time(15, 30, 45)
        time_input_obj: dict[str, Any] = {"input": t}
        result = list(_convert_io(time_input_obj))
        assert len(result) == 1, f"Expected 1 item for time, got {len(result)} items: {result}"
        assert result[0] == '"15:30:45"', f"Expected ISO format time, got {result[0]}"

        # Test datetime with timezone
        dt_tz = datetime.datetime(2023, 12, 25, 15, 30, 45, tzinfo=datetime.timezone.utc)
        tz_input_obj: dict[str, Any] = {"input": dt_tz}
        result = list(_convert_io(tz_input_obj))
        assert result[0] == '"2023-12-25T15:30:45+00:00"', (
            f"Expected ISO format with timezone, got {result[0]}"
        )

        # Test in complex structure
        complex_obj: dict[str, Any] = {"metadata": {"created_at": dt, "updated_at": d}}
        result = list(_convert_io(complex_obj))
        assert len(result) == 2

        parsed = json.loads(result[0])
        assert parsed["metadata"]["created_at"] == "2023-12-25T15:30:45"
        assert parsed["metadata"]["updated_at"] == "2023-12-25"

    def test_convert_io_timedelta_support(self) -> None:
        """Test that timedelta objects are serialized as total seconds."""

        # Test simple timedelta
        td = datetime.timedelta(hours=2, minutes=30, seconds=15)
        td_obj: dict[str, Any] = {"input": td}
        result = list(_convert_io(td_obj))
        assert len(result) == 1, f"Expected 1 item for timedelta, got {len(result)} items: {result}"

        expected_seconds = 2 * 3600 + 30 * 60 + 15  # 9015
        # timedelta.total_seconds() always returns a float, even for whole numbers
        assert result[0] == str(float(expected_seconds)), f"Expected total seconds, got {result[0]}"

        # Test fractional seconds
        td_frac = datetime.timedelta(seconds=1.5)
        td_frac_obj: dict[str, Any] = {"input": td_frac}
        result = list(_convert_io(td_frac_obj))
        assert result[0] == "1.5", f"Expected fractional seconds, got {result[0]}"

        # Test in complex structure
        complex_obj: dict[str, Any] = {"performance": {"duration": td, "timeout": td_frac}}
        result = list(_convert_io(complex_obj))
        assert len(result) == 2

        parsed = json.loads(result[0])
        assert parsed["performance"]["duration"] == float(
            expected_seconds
        )  # JSON contains float from total_seconds()
        assert parsed["performance"]["timeout"] == 1.5

    def test_convert_io_mixed_advanced_types(self) -> None:
        """Test combinations of advanced data types in complex structures."""

        class Priority(Enum):
            LOW = 1
            HIGH = 10

        @dataclasses.dataclass
        class TaskInfo:
            name: str
            priority: Priority
            created_at: datetime.datetime
            duration: Optional[datetime.timedelta] = None

        task = TaskInfo(
            name="Process Data",
            priority=Priority.HIGH,
            created_at=datetime.datetime(2023, 12, 25, 10, 0, 0),
            duration=datetime.timedelta(minutes=45),
        )

        complex_obj: dict[str, Any] = {
            "task": task,
            "metadata": {"timestamp": datetime.datetime.now(), "status": Priority.LOW},
        }

        result = list(_convert_io(complex_obj))
        assert len(result) == 2, f"Expected 2 items for complex mixed types, got {len(result)}"
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

        # Verify the JSON is valid and properly structured
        parsed = json.loads(result[0])

        # Check task dataclass fields
        assert parsed["task"]["name"] == "Process Data"
        assert parsed["task"]["priority"] == 10  # Enum value
        assert parsed["task"]["created_at"] == "2023-12-25T10:00:00"
        assert parsed["task"]["duration"] == 2700.0  # 45 minutes in seconds

        # Check metadata
        assert "timestamp" in parsed["metadata"]
        assert parsed["metadata"]["status"] == 1  # Enum value

    def test_convert_io_advanced_types_json_validation(self) -> None:
        """Test that all advanced types produce valid JSON when used with MIME types."""

        class Color(Enum):
            RED = "#FF0000"
            BLUE = "#0000FF"

        @dataclasses.dataclass
        class Config:
            name: str
            value: Union[str, int]
            timestamp: datetime.datetime
            color: Color

        config = Config(
            name="test_config",
            value=42,
            timestamp=datetime.datetime(2023, 1, 1, 12, 0, 0),
            color=Color.RED,
        )

        # Test as input/output key (should get MIME type for complex object)
        input_obj = {"input": config}
        result = list(_convert_io(input_obj))
        assert len(result) == 2
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

        # Verify JSON validity
        parsed = json.loads(result[0])
        assert isinstance(parsed, dict)
        assert parsed["name"] == "test_config"
        assert parsed["value"] == 42
        assert parsed["timestamp"] == "2023-01-01T12:00:00"
        assert parsed["color"] == "#FF0000"

        # Test as general case (should always get MIME type)
        general_obj: dict[str, Any] = {"config": config, "extra": "data"}
        result = list(_convert_io(general_obj))
        assert len(result) == 2
        assert result[1] == OpenInferenceMimeTypeValues.JSON.value

        # Verify JSON validity for general case
        parsed = json.loads(result[0])
        assert isinstance(parsed, dict)
        assert "config" in parsed
        assert "extra" in parsed

    def test_convert_io_advanced_types_edge_cases(self) -> None:
        """Test edge cases and special scenarios for advanced data types."""

        # Test empty dataclass
        @dataclasses.dataclass
        class EmptyClass:
            pass

        empty_obj = EmptyClass()
        input_obj: dict[str, Any] = {"input": empty_obj}
        result = list(_convert_io(input_obj))
        assert len(result) == 2  # Empty dataclass still gets MIME type
        parsed = json.loads(result[0])
        assert parsed == {}, f"Expected empty object for empty dataclass, got {parsed}"

        # Test dataclass with all None optional fields
        @dataclasses.dataclass
        class OptionalClass:
            required: str
            optional1: Optional[str] = None
            optional2: Optional[int] = None

        minimal_obj = OptionalClass(required="test")
        minimal_obj_input: dict[str, Any] = {"input": minimal_obj}
        result = list(_convert_io(minimal_obj_input))
        parsed = json.loads(result[0])
        assert "optional1" not in parsed
        assert "optional2" not in parsed
        assert parsed["required"] == "test"

        # Test extreme datetime values
        min_date = datetime.date.min  # year 1
        max_date = datetime.date.max  # year 9999

        dates_obj: dict[str, Any] = {"dates": {"min": min_date, "max": max_date}}
        result = list(_convert_io(dates_obj))
        parsed = json.loads(result[0])
        assert parsed["dates"]["min"] == "0001-01-01"
        assert parsed["dates"]["max"] == "9999-12-31"

        # Test zero timedelta
        zero_td = datetime.timedelta(0)
        zero_td_obj: dict[str, Any] = {"input": zero_td}
        result = list(_convert_io(zero_td_obj))
        assert result[0] == "0.0", f"Expected zero for zero timedelta, got {result[0]}"

        # Test negative timedelta
        negative_td = datetime.timedelta(days=-1)
        negative_td_obj: dict[str, Any] = {"input": negative_td}
        result = list(_convert_io(negative_td_obj))
        assert result[0] == str(-86400.0), (
            f"Expected negative seconds for negative timedelta, got {result[0]}"
        )
