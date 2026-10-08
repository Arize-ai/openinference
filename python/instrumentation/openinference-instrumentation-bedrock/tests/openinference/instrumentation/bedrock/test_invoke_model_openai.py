"""Integration tests for OpenAI models (gpt-oss, GPT-5.x, GPT-6) via the invoke_model APIs.

These models take and return an OpenAI Chat Completions body on InvokeModel.
Cassettes recorded against the real AWS Bedrock API and replayed for CI.
"""

import json
from typing import Any, Sequence, cast
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.eventstream import EventStream
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from openinference.instrumentation import OITracer, TraceConfig, using_attributes
from openinference.instrumentation.bedrock._wrappers import _InvokeModelWithResponseStream
from openinference.instrumentation.bedrock.utils._extract_invoke_model_attributes import (
    set_input_attributes,
    set_response_attributes,
)
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

_MODEL_ID = "us.openai.gpt-6-sol"
_CLIENT_KWARGS = {
    "region_name": "us-east-1",
    "aws_access_key_id": "123",
    "aws_secret_access_key": "321",
}


class TestOpenAIInvokeModel:
    @pytest.mark.vcr(
        before_record_request=lambda r: r.headers.clear() or r,
    )
    def test_invoke_model(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        messages = [
            {"role": "system", "content": "Answer in one word."},
            {"role": "user", "content": "say pong"},
        ]
        client = boto3.client("bedrock-runtime", **_CLIENT_KWARGS)
        response = client.invoke_model(
            modelId=_MODEL_ID,
            body=json.dumps({"messages": messages, "max_completion_tokens": 256}),
        )
        response_body = json.loads(response["body"].read())

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes = dict(span.attributes or {})

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
        assert attributes.pop(LLM_MODEL_NAME) == _MODEL_ID
        assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "aws"
        assert attributes.pop(INPUT_VALUE) == json.dumps(messages)
        assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "Answer in one word."
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}") == "say pong"
        assert json.loads(str(attributes.pop(LLM_INVOCATION_PARAMETERS))) == {
            "max_completion_tokens": 256
        }
        message = response_body["choices"][0]["message"]
        assert attributes.pop(OUTPUT_VALUE) == message["content"]
        assert attributes.pop(OUTPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == message["content"]
        assert attributes.pop(LLM_FINISH_REASON) == "stop"
        usage = response_body["usage"]
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == usage["prompt_tokens"]
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == usage["completion_tokens"]
        assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == usage["total_tokens"]
        assert attributes == {}


class TestOpenAIInvokeModelWithResponseStream:
    @pytest.mark.vcr(
        before_record_request=lambda r: r.headers.clear() or r,
    )
    def test_streaming(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        messages = [{"role": "user", "content": "count 1 to 5"}]
        request_body = {
            "messages": messages,
            "max_completion_tokens": 256,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        client = boto3.client("bedrock-runtime", **_CLIENT_KWARGS)
        response = client.invoke_model_with_response_stream(
            modelId=_MODEL_ID,
            body=json.dumps(request_body),
        )
        text = ""
        usage: dict[str, Any] = {}
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            for choice in chunk["choices"]:
                text += choice["delta"].get("content") or ""
            usage = chunk.get("usage") or usage
        assert text

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.status.is_ok
        attributes = dict(span.attributes or {})

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
        assert attributes.pop(LLM_MODEL_NAME) == _MODEL_ID
        assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "aws"
        assert attributes.pop(INPUT_VALUE) == json.dumps(messages)
        assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "count 1 to 5"
        assert json.loads(str(attributes.pop(LLM_INVOCATION_PARAMETERS))) == {
            "max_completion_tokens": 256,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        assert attributes.pop(OUTPUT_VALUE) == text
        assert attributes.pop(OUTPUT_MIME_TYPE) == "text/plain"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == text
        assert attributes.pop(LLM_FINISH_REASON) == "stop"
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == usage["prompt_tokens"]
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == usage["completion_tokens"]
        assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == usage["total_tokens"]
        assert attributes == {}


def test_stream_without_usage_takes_tokens_from_invocation_metrics(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """gpt-oss streams have no ``usage`` unless ``stream_options.include_usage`` is set."""
    chunks = (
        {"choices": [{"delta": {"content": "pong"}, "finish_reason": None, "index": 0}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
            "amazon-bedrock-invocationMetrics": {
                "inputTokenCount": 73,
                "outputTokenCount": 67,
                "invocationLatency": 1226,
                "firstByteLatency": 367,
            },
        },
    )
    events = [{"chunk": {"bytes": json.dumps(c).encode()}} for c in chunks]
    stream = MagicMock(spec=EventStream)
    stream.__iter__.return_value = iter(events)
    response = {"body": stream}
    span = tracer_provider.get_tracer(__name__).start_span(
        "bedrock.invoke_model_with_response_stream"
    )
    _InvokeModelWithResponseStream.handle_response(
        response,
        {
            "modelId": "openai.gpt-oss-120b-1:0",
            "body": json.dumps({"messages": [{"role": "user", "content": "say pong"}]}),
        },
        span,
    )
    for _ in response["body"]:
        pass

    (span_data,) = in_memory_span_exporter.get_finished_spans()
    assert span_data.status.is_ok
    attributes = dict(span_data.attributes or {})
    assert attributes.pop(OUTPUT_VALUE) == "pong"
    assert attributes.pop(OUTPUT_MIME_TYPE) == "text/plain"
    assert attributes.pop(LLM_FINISH_REASON) == "stop"
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 73
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 67
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 140
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "pong"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "say pong"
    assert attributes.pop(INPUT_VALUE) == json.dumps([{"role": "user", "content": "say pong"}])
    assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(LLM_INVOCATION_PARAMETERS) == "{}"
    assert attributes.pop(LLM_MODEL_NAME) == "openai.gpt-oss-120b-1:0"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "aws"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert attributes == {}


_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _stream_response(chunks: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    events = [{"chunk": {"bytes": json.dumps(c).encode()}} for c in chunks]
    stream = MagicMock(spec=EventStream)
    stream.__iter__.return_value = iter(events)
    return {"body": stream}


def test_invoke_model_tool_calls(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Assistant replies that only carry tool_calls still record their output."""
    request_body = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        ],
        "tools": [_TOOL],
        "max_completion_tokens": 64,
    }
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Lyon"}'},
            }
        ],
    }
    response_body = {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}]}
    kwargs = {"modelId": _MODEL_ID, "body": json.dumps(request_body)}
    span = tracer_provider.get_tracer(__name__).start_span("bedrock.invoke_model")
    set_input_attributes(span, json.loads(kwargs["body"]), kwargs)
    set_response_attributes(span, kwargs, response_body, {})
    span.end()

    (span_data,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(span_data.attributes or {})
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
        == "Weather in Paris?"
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "assistant"
    prefix = f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_TOOL_CALLS}.0."
    assert attributes.pop(f"{prefix}{TOOL_CALL_ID}") == "call_1"
    assert attributes.pop(f"{prefix}{TOOL_CALL_FUNCTION_NAME}") == "get_weather"
    assert attributes.pop(f"{prefix}{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}") == '{"city": "Paris"}'
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_ROLE}") == "tool"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_TOOL_CALL_ID}") == "call_1"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_CONTENT}") == "Sunny"
    assert json.loads(str(attributes.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}"))) == _TOOL
    assert json.loads(str(attributes.pop(LLM_INVOCATION_PARAMETERS))) == {
        "max_completion_tokens": 64
    }
    assert json.loads(str(attributes.pop(OUTPUT_VALUE))) == message
    assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    prefix = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0."
    assert attributes.pop(f"{prefix}{TOOL_CALL_ID}") == "call_2"
    assert attributes.pop(f"{prefix}{TOOL_CALL_FUNCTION_NAME}") == "get_weather"
    assert attributes.pop(f"{prefix}{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}") == '{"city": "Lyon"}'
    assert attributes.pop(LLM_FINISH_REASON) == "tool_calls"
    assert attributes.pop(INPUT_VALUE) == json.dumps(request_body["messages"])
    assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(LLM_MODEL_NAME) == _MODEL_ID
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "aws"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert attributes == {}


def _consume_stream(
    tracer_provider: TracerProvider,
    chunks: tuple[dict[str, Any], ...],
    request_body: dict[str, Any],
) -> None:
    response = _stream_response(chunks)
    span = tracer_provider.get_tracer(__name__).start_span(
        "bedrock.invoke_model_with_response_stream"
    )
    _InvokeModelWithResponseStream.handle_response(
        response, {"modelId": _MODEL_ID, "body": json.dumps(request_body)}, span
    )
    for _ in response["body"]:
        pass


def _pop_request_attributes(
    attributes: dict[str, Any],
    request_body: dict[str, Any],
    invocation_parameters: dict[str, Any],
) -> None:
    """Pop the request attributes shared by every OpenAI span in these tests."""
    assert attributes.pop(INPUT_VALUE) == json.dumps(request_body["messages"])
    assert attributes.pop(INPUT_MIME_TYPE) == "application/json"
    assert json.loads(str(attributes.pop(LLM_INVOCATION_PARAMETERS))) == invocation_parameters
    assert attributes.pop(LLM_MODEL_NAME) == _MODEL_ID
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "aws"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value


def test_stream_tool_calls(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Tool call fragments in the stream are joined into the output message."""
    chunks: tuple[dict[str, Any], ...] = (
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": '}}]},
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    )
    request_body = {
        "messages": [{"role": "user", "content": "Weather in Paris?"}],
        "tools": [_TOOL],
    }
    _consume_stream(tracer_provider, chunks, request_body)

    (span_data,) = in_memory_span_exporter.get_finished_spans()
    assert span_data.status.is_ok
    attributes = dict(span_data.attributes or {})
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
    }
    assert json.loads(str(attributes.pop(OUTPUT_VALUE))) == {
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call],
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    prefix = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0."
    assert attributes.pop(f"{prefix}{TOOL_CALL_ID}") == "call_1"
    assert attributes.pop(f"{prefix}{TOOL_CALL_FUNCTION_NAME}") == "get_weather"
    assert attributes.pop(f"{prefix}{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}") == '{"city": "Paris"}'
    assert attributes.pop(LLM_FINISH_REASON) == "tool_calls"
    assert json.loads(str(attributes.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}"))) == _TOOL
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "Weather in Paris?"
    _pop_request_attributes(attributes, request_body, {})
    assert attributes == {}


def test_stream_refusal(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """A refusal-only stream records the refusal as the output text."""
    chunks: tuple[dict[str, Any], ...] = (
        {"choices": [{"index": 0, "delta": {"role": "assistant", "refusal": "I can't "}}]},
        {"choices": [{"index": 0, "delta": {"refusal": "help with that."}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    request_body = {"messages": [{"role": "user", "content": "something unsafe"}]}
    _consume_stream(tracer_provider, chunks, request_body)

    (span_data,) = in_memory_span_exporter.get_finished_spans()
    assert span_data.status.is_ok
    attributes = dict(span_data.attributes or {})
    assert attributes.pop(OUTPUT_VALUE) == "I can't help with that."
    assert attributes.pop(OUTPUT_MIME_TYPE) == "text/plain"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "I can't help with that."
    assert attributes.pop(LLM_FINISH_REASON) == "stop"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "something unsafe"
    _pop_request_attributes(attributes, request_body, {})
    assert attributes == {}


def test_stream_multiple_choices(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """With n > 1, each choice index is kept as its own output message."""
    chunks: tuple[dict[str, Any], ...] = (
        {
            "choices": [
                {"index": 1, "delta": {"role": "assistant", "content": "Hi"}},
                {"index": 0, "delta": {"role": "assistant", "content": "Hello"}},
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_a",
                                "function": {"name": "get_weather", "arguments": '{"city": "A"}'},
                            }
                        ]
                    },
                },
                {
                    "index": 1,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_b",
                                "function": {"name": "get_weather", "arguments": '{"city": "B"}'},
                            }
                        ]
                    },
                },
            ]
        },
        {
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "tool_calls"},
                {"index": 1, "delta": {"content": " there"}, "finish_reason": "stop"},
            ]
        },
    )
    request_body = {"messages": [{"role": "user", "content": "greet"}], "n": 2}
    _consume_stream(tracer_provider, chunks, request_body)

    (span_data,) = in_memory_span_exporter.get_finished_spans()
    assert span_data.status.is_ok
    attributes = dict(span_data.attributes or {})

    def tool_call(call_id: str, city: str) -> dict[str, Any]:
        return {
            "id": call_id,
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": city})},
        }

    assert json.loads(str(attributes.pop(OUTPUT_VALUE))) == [
        {"role": "assistant", "content": "Hello", "tool_calls": [tool_call("call_a", "A")]},
        {"role": "assistant", "content": "Hi there", "tool_calls": [tool_call("call_b", "B")]},
    ]
    assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
    for i, (text, call_id, city) in enumerate(
        (("Hello", "call_a", "A"), ("Hi there", "call_b", "B"))
    ):
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.{i}.{MESSAGE_ROLE}") == "assistant"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.{i}.{MESSAGE_CONTENT}") == text
        prefix = f"{LLM_OUTPUT_MESSAGES}.{i}.{MESSAGE_TOOL_CALLS}.0."
        assert attributes.pop(f"{prefix}{TOOL_CALL_ID}") == call_id
        assert attributes.pop(f"{prefix}{TOOL_CALL_FUNCTION_NAME}") == "get_weather"
        assert attributes.pop(f"{prefix}{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}") == json.dumps(
            {"city": city}
        )
    # llm.finish_reason holds a single value, taken from the first choice.
    assert attributes.pop(LLM_FINISH_REASON) == "tool_calls"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "greet"
    _pop_request_attributes(attributes, request_body, {"n": 2})
    assert attributes == {}


def test_invoke_model_multiple_choices_and_refusal(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Every choice is recorded in index order; a refusal is kept as message content."""
    request_body = {"messages": [{"role": "user", "content": "hi"}], "n": 2}
    first = {"role": "assistant", "content": "Hello"}
    second = {"role": "assistant", "content": None, "refusal": "I can't help with that."}
    response_body = {
        "choices": [
            {"index": 1, "finish_reason": "stop", "message": second},
            {"index": 0, "finish_reason": "stop", "message": first},
        ]
    }
    kwargs = {"modelId": _MODEL_ID, "body": json.dumps(request_body)}
    span = tracer_provider.get_tracer(__name__).start_span("bedrock.invoke_model")
    set_input_attributes(span, json.loads(kwargs["body"]), kwargs)
    set_response_attributes(span, kwargs, response_body, {})
    span.end()

    (span_data,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(span_data.attributes or {})
    assert json.loads(str(attributes.pop(OUTPUT_VALUE))) == [first, second]
    assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "Hello"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.1.{MESSAGE_CONTENT}") == "I can't help with that."
    assert attributes.pop(LLM_FINISH_REASON) == "stop"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "hi"
    _pop_request_attributes(attributes, request_body, {"n": 2})
    assert attributes == {}


def test_stream_keeps_context_attributes(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Context attributes set around the call are kept when the stream is read later."""
    chunks = ({"choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": "stop"}]},)
    response = _stream_response(chunks)
    request_body = {"messages": [{"role": "user", "content": "say pong"}]}
    tracer = cast(Tracer, OITracer(tracer_provider.get_tracer(__name__), config=TraceConfig()))
    with using_attributes(
        session_id="session-1",
        user_id="user-1",
        metadata={"key": "value"},
        tags=["tag-1"],
    ):
        _InvokeModelWithResponseStream(tracer)._sync_call(
            lambda **_: response,
            (),
            {"modelId": _MODEL_ID, "body": json.dumps(request_body)},
        )
    for _ in response["body"]:
        pass

    (span_data,) = in_memory_span_exporter.get_finished_spans()
    assert span_data.status.is_ok
    attributes = dict(span_data.attributes or {})
    assert attributes.pop(SESSION_ID) == "session-1"
    assert attributes.pop(USER_ID) == "user-1"
    assert json.loads(str(attributes.pop(METADATA))) == {"key": "value"}
    assert list(cast(Sequence[str], attributes.pop(TAG_TAGS))) == ["tag-1"]
    assert attributes.pop(OUTPUT_VALUE) == "pong"
    assert attributes.pop(OUTPUT_MIME_TYPE) == "text/plain"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "pong"
    assert attributes.pop(LLM_FINISH_REASON) == "stop"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "say pong"
    _pop_request_attributes(attributes, request_body, {})
    assert attributes == {}


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_FINISH_REASON = SpanAttributes.LLM_FINISH_REASON
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_TOOLS = SpanAttributes.LLM_TOOLS
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
METADATA = SpanAttributes.METADATA
TAG_TAGS = SpanAttributes.TAG_TAGS
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA
