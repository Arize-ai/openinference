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
        pytest.param(None, None, id="none"),
        pytest.param("", None, id="empty-string"),
        pytest.param([], [], id="empty-list"),
        pytest.param("AAAA!", None, id="invalid-base64"),
        pytest.param("AAA=", None, id="valid-base64-wrong-length"),
        pytest.param([1, 2, 3], [1, 2, 3], id="list-of-ints"),
        pytest.param([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], id="list-of-floats"),
        pytest.param((1.0, 2.0), [1.0, 2.0], id="tuple-converts-to-list"),
        pytest.param(["a", "b"], None, id="non-numeric-list"),
        pytest.param([[1, 2]], None, id="nested-structure"),
        pytest.param({"a": 1}, None, id="dict"),
        pytest.param(42, None, id="single-number"),
        pytest.param(
            _make_base64_vector([1.0, 2.0, 3.0, 4.0]), [1.0, 2.0, 3.0, 4.0], id="base64-encoded"
        ),
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
