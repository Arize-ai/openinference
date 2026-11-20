import io
import json
from typing import Any, Dict, List, cast
from unittest.mock import MagicMock

import pytest
from botocore.eventstream import EventStream
from botocore.response import StreamingBody
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode, Tracer

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.bedrock import _model_invocation_wrapper
from openinference.instrumentation.bedrock._wrappers import _InvokeModelWithResponseStream

_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
_PAYLOAD = json.dumps(
    {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "hi"}],
    }
).encode()
_PREFIX = b"PREFIX"


def _tracer(tracer_provider: TracerProvider) -> Tracer:
    return cast(Tracer, OITracer(tracer_provider.get_tracer(__name__), config=TraceConfig()))


def _body(offset: bool) -> io.BytesIO:
    body = io.BytesIO(_PREFIX + _PAYLOAD if offset else _PAYLOAD)
    body.seek(len(_PREFIX) if offset else 0)
    return body


@pytest.mark.parametrize("offset", [False, True], ids=["start", "offset"])
def test_invoke_model_reads_file_like_body_without_consuming_it(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    offset: bool,
) -> None:
    sent: List[bytes] = []
    response = json.dumps(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": _MODEL_ID,
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
    ).encode()

    def invoke_model(**kwargs: Any) -> Dict[str, Any]:
        sent.append(kwargs["body"].read())
        return {"body": StreamingBody(io.BytesIO(response), len(response))}

    client = MagicMock()
    client._unwrapped_invoke_model.side_effect = invoke_model
    _model_invocation_wrapper(_tracer(tracer_provider))(client)(
        modelId=_MODEL_ID, body=_body(offset)
    )

    assert sent == [_PAYLOAD]
    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes["llm.input_messages.0.message.role"] == "user"
    assert attributes["llm.input_messages.0.message.content"] == "hi"
    assert json.loads(str(attributes["input.value"])) == json.loads(_PAYLOAD)


@pytest.mark.parametrize("consumed", [False, True], ids=["unread", "read-by-botocore"])
def test_stream_with_file_like_body_ends_span_with_input(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    consumed: bool,
) -> None:
    chunks = (
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": _MODEL_ID,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 5, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    )
    event_stream = MagicMock(spec=EventStream)
    event_stream.__iter__.return_value = iter(
        [{"chunk": {"bytes": json.dumps(c).encode()}} for c in chunks]
    )

    def invoke_model_with_response_stream(**kwargs: Any) -> Dict[str, Any]:
        if consumed:
            kwargs["body"].read()
        return {"body": event_stream}

    response = _InvokeModelWithResponseStream(_tracer(tracer_provider))._sync_call(
        invoke_model_with_response_stream, (), {"modelId": _MODEL_ID, "body": _body(False)}
    )
    for _ in response["body"]:
        pass

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.OK
    attributes = dict(span.attributes or {})
    assert attributes["llm.input_messages.0.message.role"] == "user"
    assert attributes["llm.output_messages.0.message.contents.0.message_content.text"] == "ok"


def test_stream_call_that_raises_ends_span_with_error(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def invoke_model_with_response_stream(**kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("throttled")

    with pytest.raises(RuntimeError, match="throttled"):
        _InvokeModelWithResponseStream(_tracer(tracer_provider))._sync_call(
            invoke_model_with_response_stream, (), {"modelId": _MODEL_ID, "body": _PAYLOAD}
        )

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "throttled"
    assert [event.name for event in span.events] == ["exception"]
