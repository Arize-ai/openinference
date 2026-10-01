"""Regression test for _flatten crashing on mixed-type lists."""

from typing import Any, Dict

from openinference.instrumentation.google_genai._wrappers import _flatten


def test_flatten_with_mixed_type_list_does_not_raise() -> None:
    key = "llm.input_messages.0.message.contents"
    mapping: Dict[str, Any] = {
        key: [{"message_content": {"type": "text", "text": "hi"}}, "follow-up"],
        "llm.model_name": "gpt-4o",
    }

    flattened = dict(_flatten(mapping))

    assert flattened[f"{key}.0.message_content.type"] == "text"
    assert flattened[f"{key}.0.message_content.text"] == "hi"
    assert flattened[f"{key}.1"] == "follow-up"
    assert flattened["llm.model_name"] == "gpt-4o"
