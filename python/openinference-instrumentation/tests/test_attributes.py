from openinference.instrumentation import (
    get_llm_input_message_attributes,
    get_llm_output_message_attributes,
)


class TestMessageAttributeTypeGuards:
    def test_list_content_emits_typed_content_blocks(self) -> None:
        attrs = dict(
            get_llm_input_message_attributes(
                [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]  # type: ignore[typeddict-item]
            )
        )
        assert "llm.input_messages.0.message.content" not in attrs
        assert attrs["llm.input_messages.0.message.contents.0.message_content.type"] == "text"
        assert attrs["llm.input_messages.0.message.contents.0.message_content.text"] == "hello"

    def test_string_content_unchanged(self) -> None:
        attrs = dict(get_llm_input_message_attributes([{"role": "user", "content": "hello"}]))
        assert attrs["llm.input_messages.0.message.role"] == "user"
        assert attrs["llm.input_messages.0.message.content"] == "hello"

    def test_non_string_role_is_dropped(self) -> None:
        attrs = dict(get_llm_input_message_attributes([{"role": 123, "content": "plain"}]))  # type: ignore[typeddict-item]
        assert "llm.input_messages.0.message.role" not in attrs
        assert attrs["llm.input_messages.0.message.content"] == "plain"

    def test_contents_field_takes_precedence_over_list_content(self) -> None:
        attrs = dict(
            get_llm_input_message_attributes(
                [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "from-content"}],  # type: ignore[typeddict-item]
                        "contents": [{"type": "text", "text": "from-contents"}],
                    }
                ]
            )
        )
        assert (
            attrs["llm.input_messages.0.message.contents.0.message_content.text"] == "from-contents"
        )

    def test_output_messages_apply_same_guards(self) -> None:
        attrs = dict(
            get_llm_output_message_attributes(
                [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]  # type: ignore[typeddict-item]
            )
        )
        assert "llm.output_messages.0.message.content" not in attrs
        assert attrs["llm.output_messages.0.message.contents.0.message_content.text"] == "hi"
