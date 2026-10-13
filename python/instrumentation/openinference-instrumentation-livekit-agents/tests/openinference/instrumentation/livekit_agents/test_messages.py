from openinference.instrumentation.livekit_agents._messages import oi_messages
from openinference.instrumentation.livekit_agents._processor import _provider


def test_tool_round_trip_messages() -> None:
    messages = oi_messages(
        '[{"role": "user", "parts": [{"type": "text", "content": "weather?"}]},'
        ' {"role": "assistant", "parts": [{"type": "tool_call", "id": "call_1",'
        ' "name": "get_weather", "arguments": {"city": "London"}}]},'
        ' {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call_1",'
        ' "name": "get_weather", "response": "18 degrees"}]}]',
        '[{"type": "text", "content": "Answer briefly."}]',
    )
    assert messages == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "18 degrees"},
    ]


def test_unparseable_messages_are_dropped() -> None:
    assert oi_messages("not json") == []
    assert oi_messages('{"role": "user"}') == []
    assert oi_messages(None, "[]") == []


def test_provider_host_maps_to_well_known_value() -> None:
    assert _provider("api.openai.com") == "openai"
    assert _provider("openai") == "openai"
    assert _provider("deepgram") == "deepgram"
    assert _provider("") is None
