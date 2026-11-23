from typing import Any

from google.adk.models.llm_request import LlmRequest
from google.genai import types

from openinference.instrumentation.google_adk._wrappers import _TraceCallLlm


class _Span:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


def test_parallel_function_responses_do_not_overwrite_the_next_message() -> None:
    llm_request = LlmRequest(
        contents=[
            types.Content(role="user", parts=[types.Part(text="q")]),
            types.Content(
                role="model",
                parts=[
                    types.Part(function_call=types.FunctionCall(id="a", name="tool_a", args={})),
                    types.Part(function_call=types.FunctionCall(id="b", name="tool_b", args={})),
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="a", name="tool_a", response={"result": "A"}
                        )
                    ),
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="b", name="tool_b", response={"result": "B"}
                        )
                    ),
                ],
            ),
            types.Content(role="model", parts=[types.Part(text="done")]),
        ]
    )
    span = _Span()

    _TraceCallLlm._derive_request_attributes(span, llm_request)

    attributes = span.attributes
    assert attributes["llm.input_messages.2.message.role"] == "tool"
    assert attributes["llm.input_messages.2.message.tool_call_id"] == "a"
    assert attributes["llm.input_messages.3.message.role"] == "tool"
    assert attributes["llm.input_messages.3.message.tool_call_id"] == "b"
    assert attributes["llm.input_messages.3.message.content"] == '{"result": "B"}'
    assert attributes["llm.input_messages.4.message.role"] == "model"
    assert attributes["llm.input_messages.4.message.contents.0.message_content.text"] == "done"
    assert not any(key.startswith("llm.input_messages.3.message.contents.") for key in attributes)


def test_text_after_a_function_response_does_not_overwrite_the_next_message() -> None:
    llm_request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="a", name="tool_a", response={"result": "A"}
                        )
                    ),
                    types.Part(text="note"),
                ],
            ),
            types.Content(role="model", parts=[types.Part(text="done")]),
        ]
    )
    span = _Span()

    _TraceCallLlm._derive_request_attributes(span, llm_request)

    attributes = span.attributes
    assert attributes["llm.input_messages.0.message.role"] == "tool"
    assert attributes["llm.input_messages.1.message.contents.1.message_content.text"] == "note"
    assert attributes["llm.input_messages.2.message.role"] == "model"
    assert attributes["llm.input_messages.2.message.contents.0.message_content.text"] == "done"
