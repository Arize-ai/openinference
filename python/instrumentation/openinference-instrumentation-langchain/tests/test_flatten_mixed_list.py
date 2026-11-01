"""Regression test for _flatten crashing on mixed-type lists."""

from typing import Any, Dict, List

from openinference.instrumentation.langchain._tracer import _flatten


def test_flatten_with_mixed_type_list_does_not_raise() -> None:
    key = "llm.input_messages.0.message.contents"
    items: Dict[str, Any] = {
        key: [{"message_content": {"type": "text", "text": "hi"}}, "follow-up"],
        "llm.model_name": "gpt-4o",
    }

    flattened = dict(_flatten(items.items()))

    assert flattened[f"{key}.0.message_content.type"] == "text"
    assert flattened[f"{key}.0.message_content.text"] == "hi"
    assert flattened[f"{key}.1"] == "follow-up"
    assert flattened["llm.model_name"] == "gpt-4o"


def test_flatten_with_circular_list_yields_placeholder() -> None:
    key = "llm.input_messages.0.message.content"
    circular: List[Any] = []
    circular.append(circular)
    mapping: Dict[str, Any] = {key: [{"type": "text", "text": "hi"}, circular]}

    flattened = dict(_flatten(mapping.items()))

    assert flattened[f"{key}.0.type"] == "text"
    assert flattened[f"{key}.0.text"] == "hi"
    assert flattened[f"{key}.1"] == "<circular reference>"


def test_flatten_with_circular_mapping_yields_placeholder() -> None:
    circular: Dict[str, Any] = {}
    circular["self"] = circular

    flattened = dict(_flatten({"payload": circular}.items()))

    assert flattened["payload.self"] == "<circular reference>"


def test_flatten_keeps_non_circular_lists() -> None:
    # Regression: _contains_cycle wrapped in @stop_on_exception became a
    # generator function, so every list read as circular and was dropped.
    flattened = dict(_flatten({"tags": ["a", "b", "c"]}.items()))
    assert flattened["tags"] == ["a", "b", "c"]

    mixed = dict(_flatten({"messages": [{"role": "user"}, ["x", "y"]]}.items()))
    assert mixed["messages.0.role"] == "user"
    assert mixed["messages.1"] == ["x", "y"]
