"""A Bedrock Converse message may carry several toolResult blocks; all of them are input.

`get_message_objects` wrote `tool_call_id` and `content` as scalars while looping over the
blocks, so every block but the last was overwritten. The typed `Message` has exactly one
`tool_call_id`, and `MessageContent` has no tool-result variant, so the extra results have
nowhere to go in one message: each tool result has to become its own message.
"""

from openinference.instrumentation.bedrock._converse_attributes import get_message_objects


def test_parallel_tool_results_keep_every_id_and_output() -> None:
    message_list = [
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "call_weather", "content": [{"text": "sunny"}]}},
                {"toolResult": {"toolUseId": "call_flight", "content": [{"text": "$420"}]}},
            ],
        }
    ]

    messages = get_message_objects(message_list)  # type: ignore[arg-type]

    assert {message["tool_call_id"]: message["content"] for message in messages} == {
        "call_weather": "sunny",
        "call_flight": "$420",
    }


def test_tool_result_with_several_content_blocks_keeps_them_all() -> None:
    message_list = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "call_report",
                        "content": [{"text": "headline"}, {"json": {"rows": 2}}],
                    }
                }
            ],
        }
    ]

    messages = get_message_objects(message_list)  # type: ignore[arg-type]

    assert len(messages) == 1
    content = messages[0]["content"]
    assert "headline" in content
    assert '{"rows": 2}' in content


def test_single_tool_result_keeps_one_message_and_indexes() -> None:
    # The shape the existing Converse tests assert on must not move: one API message
    # with one result stays one OpenInference message with the same fields.
    message_list = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tooluse_ZQEZysOVRqitr-89GxHizA",
                        "content": [{"text": "Rock and Roll Hall"}],
                    }
                }
            ],
        }
    ]

    messages = get_message_objects(message_list)  # type: ignore[arg-type]

    assert messages == [
        {
            "role": "user",
            "tool_call_id": "tooluse_ZQEZysOVRqitr-89GxHizA",
            "content": "Rock and Roll Hall",
        }
    ]
