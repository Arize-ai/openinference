import base64
import struct
from typing import TYPE_CHECKING, Any, Dict, List

import pytest

from openinference.instrumentation.haystack._wrappers import (
    _decode_embedding_vector,
    _get_run_method_input_types,
)

if TYPE_CHECKING:
    # Visible to mypy, but never imported at runtime -- `Path` is absent from this module's
    # runtime globals, so `get_type_hints` on a function annotated with it raises `NameError`,
    # the same as a type only ever available under `TYPE_CHECKING` in real component code.
    from pathlib import Path


def _run_with_unresolvable_type_hint(documents: "Path") -> Dict[str, Any]:
    return {"documents": documents}


def _run_with_resolvable_type_hints(documents: List[str]) -> Dict[str, Any]:
    return {"documents": documents}


def test_get_run_method_input_types_returns_none_instead_of_raising_on_unresolvable_hint() -> None:
    """`get_type_hints` raises `NameError` for an annotation it can't resolve (e.g. a type
    only available under `TYPE_CHECKING`). That must not propagate -- a type-detection
    failure should be treated the same as a component with no type hints at all.
    """
    assert _get_run_method_input_types(_run_with_unresolvable_type_hint) is None


def test_get_run_method_input_types_still_resolves_normal_hints() -> None:
    result = _get_run_method_input_types(_run_with_resolvable_type_hints)
    assert result is not None
    assert result.get("documents") == List[str]


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
