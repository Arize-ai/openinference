import base64
import math
import struct
from typing import List, Union

import pytest

from openinference.instrumentation.haystack._base64 import decode_base64_float32


@pytest.mark.parametrize(
    "value, expected",
    [
        ("", None),  # Empty
        ("AAAA!", None),  # Invalid
        ("AAA=", None),  # Wrong length
        (
            base64.b64encode(struct.pack("4f", 1.0, 2.0, 3.0, 4.0)).decode("ascii"),
            [1.0, 2.0, 3.0, 4.0],
        ),  # Valid
        (
            base64.b64encode(struct.pack("2f", 1.0, -2.5)).decode("ascii"),
            [1.0, -2.5],
        ),  # Two floats
        (
            base64.b64encode(struct.pack("f", float("inf"))).decode("ascii"),
            [float("inf")],
        ),  # inf
        (
            base64.b64encode(struct.pack("f", math.nan)).decode("ascii"),
            "nan",
        ),  # nan (special marker)
    ],
)
def test_decode_base64_float32(value: str, expected: Union[List[float], str, None]) -> None:
    """Test base64 float32 decoding."""
    result = decode_base64_float32(value)
    if expected is None:
        assert result is None
    elif expected == "nan":
        assert result is not None
        assert len(result) == 1
        assert math.isnan(result[0])
    else:
        assert result is not None
        assert isinstance(expected, list)
        assert len(result) == len(expected)
        for i, val in enumerate(result):
            if math.isinf(expected[i]):
                assert math.isinf(val) and val == expected[i]
            else:
                assert abs(val - expected[i]) < 1e-6
