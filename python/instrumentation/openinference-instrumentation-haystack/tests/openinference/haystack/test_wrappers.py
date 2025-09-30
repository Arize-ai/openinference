import base64
import struct
from typing import Any, List

import pytest

from openinference.instrumentation.haystack._wrappers import _decode_embedding_vector


def _make_base64_vector(floats: List[float]) -> str:
    """Helper to create base64-encoded float32 vector."""
    return base64.b64encode(struct.pack(f"{len(floats)}f", *floats)).decode("ascii")


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),  # None
        ("", None),  # Empty string
        ([], []),  # Empty list
        ("AAAA!", None),  # Invalid base64
        ("AAA=", None),  # Valid base64, wrong length
        ([1, 2, 3], [1, 2, 3]),  # List of ints
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),  # List of floats
        ((1.0, 2.0), [1.0, 2.0]),  # Tuple converts to list
        (["a", "b"], None),  # Non-numeric list
        ([[1, 2]], None),  # Nested structure
        ({"a": 1}, None),  # Dict
        (42, None),  # Single number
        (_make_base64_vector([1.0, 2.0, 3.0, 4.0]), [1.0, 2.0, 3.0, 4.0]),  # Base64-encoded
    ],
)
def test_decode_embedding_vector(value: Any, expected: Any) -> None:
    """Test _decode_embedding_vector with various edge cases."""
    result = _decode_embedding_vector(value)
    if expected is None:
        assert result is None
    elif isinstance(expected, list) and all(isinstance(x, float) for x in expected):
        assert result is not None
        assert len(result) == len(expected)
        for i, val in enumerate(result):
            assert abs(val - expected[i]) < 1e-6
    else:
        assert result == expected
