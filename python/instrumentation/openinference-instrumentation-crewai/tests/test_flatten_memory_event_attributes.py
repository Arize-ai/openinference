"""Regression tests for _flatten losing or crashing on list values in memory event attributes."""

from typing import Any, Dict, List

from openinference.instrumentation.crewai._wrappers import _flatten


def test_flatten_with_mixed_type_list_does_not_raise() -> None:
    mapping: Dict[str, Any] = {
        "metadata": {"tags": [{"kind": "review"}, "follow-up"]},
        "source_type": "long_term_memory",
    }

    flattened = dict(_flatten(mapping))

    assert flattened["metadata.tags.0.kind"] == "review"
    assert flattened["metadata.tags.1"] == "follow-up"
    assert flattened["source_type"] == "long_term_memory"


def test_flatten_with_string_list_keeps_the_attribute() -> None:
    # Lists of strings used to fall into a branch that never yielded, so the
    # attribute disappeared from the span event entirely.
    flattened = dict(_flatten({"metadata": {"tags": ["a", "b"]}}))

    assert flattened["metadata.tags"] == "a, b"


def test_flatten_with_scalar_list_yields_the_list() -> None:
    flattened = dict(_flatten({"metadata": {"scores": [1, 2]}}))

    assert flattened["metadata.scores"] == [1, 2]


def test_flatten_with_circular_mapping_yields_placeholder() -> None:
    circular: Dict[str, Any] = {}
    circular["self"] = circular

    flattened = dict(_flatten({"metadata": circular}))

    assert flattened["metadata.self"] == "<circular reference>"


def test_flatten_with_circular_list_yields_placeholder() -> None:
    circular: List[Any] = []
    circular.append(circular)

    flattened = dict(_flatten({"metadata": {"items": [{"kind": "review"}, circular]}}))

    assert flattened["metadata.items.0.kind"] == "review"
    assert flattened["metadata.items.1"] == "<circular reference>"
