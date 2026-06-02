import base64
from typing import Any

import pytest
from google.genai import types

from openinference.instrumentation.google_genai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.instrumentation.google_genai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.google_genai._stream import (
    _ResponseAccumulator,
    _ResponseExtractor,
)
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

_OM = SpanAttributes.LLM_OUTPUT_MESSAGES
_IM = SpanAttributes.LLM_INPUT_MESSAGES


def _om(msg_index: int, *parts: str) -> str:
    return f"{_OM}.{msg_index}." + ".".join(parts)


def _im(msg_index: int, *parts: str) -> str:
    return f"{_IM}.{msg_index}." + ".".join(parts)


def _content_key(msg_index: int, content_index: int, attr: str, output: bool = True) -> str:
    prefix = _om if output else _im
    return prefix(
        msg_index,
        MessageAttributes.MESSAGE_CONTENTS,
        str(content_index),
        attr,
    )


@pytest.mark.parametrize(
    "candidate_parts, expected_output_attrs",
    [
        pytest.param(
            [types.Part(thought=True, text="I should answer Paris.")],
            {
                _content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TYPE): "reasoning",
                _content_key(
                    0, 0, MessageContentAttributes.MESSAGE_CONTENT_TEXT
                ): "I should answer Paris.",
            },
            id="thought_text_only",
        ),
        pytest.param(
            [
                types.Part(thought=True, text="Let me think.", thought_signature=b"CiQBsig"),
            ],
            {
                _content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TYPE): "reasoning",
                _content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TEXT): "Let me think.",
                _content_key(
                    0, 0, MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE
                ): base64.b64encode(b"CiQBsig").decode(),
            },
            id="thought_text_and_signature",
        ),
        pytest.param(
            [types.Part(thought=True, thought_signature=b"\x00\xff\xfe")],
            {
                _content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TYPE): "reasoning",
                _content_key(
                    0, 0, MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE
                ): base64.b64encode(b"\x00\xff\xfe").decode(),
            },
            id="thought_signature_only_no_text",
        ),
        pytest.param(
            [types.Part(thought=True)],
            {},
            id="empty_thought_skipped",
        ),
        pytest.param(
            [
                types.Part(thought=True, text="Thinking..."),
                types.Part(text="Paris."),
            ],
            {
                _content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TYPE): "reasoning",
                _content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TEXT): "Thinking...",
                _content_key(0, 1, MessageContentAttributes.MESSAGE_CONTENT_TYPE): "text",
                _content_key(0, 1, MessageContentAttributes.MESSAGE_CONTENT_TEXT): "Paris.",
            },
            id="thought_followed_by_text",
        ),
        pytest.param(
            [
                types.Part(thought=True, text="Thinking..."),
                types.Part(text="Paris.", thought_signature=b"SigOnTextPart"),
            ],
            {
                _content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TYPE): "reasoning",
                _content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TEXT): "Thinking...",
                _content_key(0, 1, MessageContentAttributes.MESSAGE_CONTENT_TYPE): "text",
                _content_key(0, 1, MessageContentAttributes.MESSAGE_CONTENT_TEXT): "Paris.",
                _content_key(
                    0, 1, MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE
                ): base64.b64encode(b"SigOnTextPart").decode(),
            },
            id="thought_signature_on_text_part",
        ),
    ],
)
def test_response_thought_parts(
    candidate_parts: list[types.Part],
    expected_output_attrs: dict[str, Any],
) -> None:
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(role="model", parts=candidate_parts),
            )
        ]
    )
    extractor = _ResponseAttributesExtractor()
    attrs = dict(extractor._get_attributes_from_generate_content(response, {}))
    output_attrs = {k: v for k, v in attrs.items() if k.startswith(_OM)}
    role_key = _om(0, MessageAttributes.MESSAGE_ROLE)
    output_attrs.pop(role_key, None)
    assert output_attrs == expected_output_attrs


def test_response_function_call_with_thought_signature() -> None:
    sig = b"CiQBthought"
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="get_weather",
                                args={"location": "Paris"},
                            ),
                            thought_signature=sig,
                        )
                    ],
                ),
            )
        ]
    )
    extractor = _ResponseAttributesExtractor()
    attrs = dict(extractor._get_attributes_from_generate_content(response, {}))
    tc_prefix = f"{_OM}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_weather"
    assert (
        attrs[f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_REASONING_SIGNATURE}"]
        == base64.b64encode(sig).decode()
    )


def test_response_function_call_without_thought_signature() -> None:
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="get_weather",
                                args={"location": "Paris"},
                            ),
                        )
                    ],
                ),
            )
        ]
    )
    extractor = _ResponseAttributesExtractor()
    attrs = dict(extractor._get_attributes_from_generate_content(response, {}))
    assert not any(ToolCallAttributes.TOOL_CALL_REASONING_SIGNATURE in k for k in attrs)


@pytest.mark.parametrize(
    "input_parts, expected_input_attrs",
    [
        pytest.param(
            [types.Part(thought=True, text="Prior thought.")],
            {
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "reasoning",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "Prior thought.",
            },
            id="request_thought_text_only",
        ),
        pytest.param(
            [
                types.Part(thought=True, text="Prior thought.", thought_signature=b"SigBytes"),
            ],
            {
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "reasoning",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "Prior thought.",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE,
                ): base64.b64encode(b"SigBytes").decode(),
            },
            id="request_thought_text_and_signature",
        ),
        pytest.param(
            [types.Part(thought=True)],
            {},
            id="request_empty_thought_skipped",
        ),
        pytest.param(
            [
                types.Part(thought=True, text="Thinking..."),
                types.Part(text="What is the capital of France?"),
            ],
            {
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "reasoning",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "Thinking...",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "1",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "text",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "1",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "What is the capital of France?",
            },
            id="request_thought_followed_by_text",
        ),
        pytest.param(
            [
                types.Part(thought=True, text="Thinking..."),
                types.Part(text="Paris.", thought_signature=b"SigOnTextPart"),
            ],
            {
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "reasoning",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "Thinking...",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "1",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "text",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "1",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "Paris.",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "1",
                    MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE,
                ): base64.b64encode(b"SigOnTextPart").decode(),
            },
            id="request_thought_signature_on_text_part",
        ),
    ],
)
def test_request_thought_parts(
    input_parts: list[types.Part],
    expected_input_attrs: dict[str, Any],
) -> None:
    extractor = _RequestAttributesExtractor()
    request_parameters = {
        "contents": types.Content(role="user", parts=input_parts),
    }
    attrs = dict(extractor.get_attributes_from_request(request_parameters))
    input_attrs = {
        k: v
        for k, v in attrs.items()
        if k.startswith(_IM) and MessageAttributes.MESSAGE_ROLE not in k
    }
    assert input_attrs == expected_input_attrs


def _make_stream_chunk(
    parts: list[dict[str, Any]],
    role: str = "model",
    model_version: str = "gemini-3.5-flash",
) -> "types.GenerateContentResponse":
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(role=role, parts=[types.Part(**p) for p in parts]),
            )
        ],
        model_version=model_version,
    )


def _stream_attrs(chunks: list["types.GenerateContentResponse"]) -> dict[str, Any]:
    acc = _ResponseAccumulator()
    for chunk in chunks:
        acc.process_chunk(chunk)
    attrs = dict(_ResponseExtractor(acc).get_attributes())
    return {k: v for k, v in attrs.items() if k.startswith(_OM)}


@pytest.mark.parametrize(
    "chunks, expected",
    [
        pytest.param(
            # Both parts in the same chunk → accumulator keeps them at index 0 and 1
            [
                _make_stream_chunk(
                    [
                        {"thought": True, "text": "Let me think..."},
                        {"text": "Paris.", "thought_signature": b"SigBytes"},
                    ]
                ),
            ],
            {
                _content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TYPE): "reasoning",
                _content_key(
                    0, 0, MessageContentAttributes.MESSAGE_CONTENT_TEXT
                ): "Let me think...",
                _content_key(0, 1, MessageContentAttributes.MESSAGE_CONTENT_TYPE): "text",
                _content_key(0, 1, MessageContentAttributes.MESSAGE_CONTENT_TEXT): "Paris.",
                _content_key(
                    0, 1, MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE
                ): base64.b64encode(b"SigBytes").decode(),
            },
            id="stream_thought_then_text_with_signature",
        ),
        pytest.param(
            [
                _make_stream_chunk([{"thought": True, "text": "Thinking..."}]),
                _make_stream_chunk([{"thought": True, "text": " more thinking."}]),
            ],
            {
                _content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TYPE): "reasoning",
                _content_key(
                    0, 0, MessageContentAttributes.MESSAGE_CONTENT_TEXT
                ): "Thinking... more thinking.",
            },
            id="stream_thought_text_concatenated",
        ),
        pytest.param(
            [_make_stream_chunk([{"thought": True}])],
            {},
            id="stream_empty_thought_skipped",
        ),
        pytest.param(
            [_make_stream_chunk([{"thought": True, "thought_signature": b"SigOnly"}])],
            {
                _content_key(0, 0, MessageContentAttributes.MESSAGE_CONTENT_TYPE): "reasoning",
                _content_key(
                    0, 0, MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE
                ): base64.b64encode(b"SigOnly").decode(),
            },
            id="stream_thought_signature_only",
        ),
    ],
)
def test_stream_reasoning_content(
    chunks: list["types.GenerateContentResponse"],
    expected: dict[str, Any],
) -> None:
    attrs = _stream_attrs(chunks)
    role_key = _om(0, MessageAttributes.MESSAGE_ROLE)
    attrs.pop(role_key, None)
    assert attrs == expected


def test_stream_function_call_with_thought_signature() -> None:
    sig = b"StreamSig"
    chunks = [
        _make_stream_chunk(
            [
                {
                    "function_call": types.FunctionCall(name="search", args={"q": "Paris"}),
                    "thought_signature": sig,
                }
            ]
        )
    ]
    attrs = _stream_attrs(chunks)
    tc_prefix = f"{_OM}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "search"
    assert (
        attrs[f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_REASONING_SIGNATURE}"]
        == base64.b64encode(sig).decode()
    )


def test_request_function_call_with_thought_signature() -> None:
    sig = b"ReqSigBytes"
    extractor = _RequestAttributesExtractor()
    request_parameters = {
        "contents": types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="lookup",
                        args={"q": "Paris"},
                    ),
                    thought_signature=sig,
                )
            ],
        )
    }
    attrs = dict(extractor.get_attributes_from_request(request_parameters))
    tc_prefix = f"{_IM}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "lookup"
    assert (
        attrs[f"{tc_prefix}.{ToolCallAttributes.TOOL_CALL_REASONING_SIGNATURE}"]
        == base64.b64encode(sig).decode()
    )
