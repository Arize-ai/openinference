import json
from typing import Any, Dict

import pytest
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util._importlib_metadata import entry_points
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    JSONContent,
    Noul,
    Question,
    RetryPolicy,
    Score,
    TypeSafeAuthenticationError,
    TypeSafeClient,
)

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    suppress_tracing,
    using_attributes,
)
from openinference.instrumentation.config import REDACTED_VALUE
from openinference.instrumentation.typesafe import TypeSafeAIInstrumentor
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from .conftest import SYSTEM_ONE_RESPONSE, RecordingTransport

STATE = "I was charged twice. Please help ASAP."
QUESTIONS: Dict[str, Question] = {
    "billing": Noul(instructions="Is this about billing?"),
    "tone": Choice(instructions="What is the tone?", criteria={"calm": None, "angry": None}),
    "urgency": Score(instructions="How urgent?", criteria=["low", "medium", "high"]),
}
QUESTIONS_JSON = {
    "billing": {"type": "noul", "instructions": "Is this about billing?"},
    "tone": {
        "type": "choice",
        "criteria": {"calm": None, "angry": None},
        "instructions": "What is the tone?",
    },
    "urgency": {
        "type": "score",
        "criteria": ["low", "medium", "high"],
        "instructions": "How urgent?",
    },
}


def _attrs(span: ReadableSpan) -> Dict[str, Any]:
    return dict(span.attributes or {})


def _assert_llm_span(span: ReadableSpan, transport: RecordingTransport) -> None:
    attrs = _attrs(span)
    assert span.status.status_code is StatusCode.OK
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == "typesafe"
    assert SpanAttributes.LLM_SYSTEM not in attrs
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL_NAME] == "jev-latest"
    assert attrs[SpanAttributes.LLM_RESPONSE_MODEL_NAME] == "jev-1.13.0"
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "jev-1.13.0"

    # input.value mirrors the wire request body exactly.
    assert attrs[SpanAttributes.INPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
    assert json.loads(str(attrs[SpanAttributes.INPUT_VALUE])) == transport.requests[-1]
    assert transport.requests[-1] == {
        "state": STATE,
        "model": "jev-latest",
        "questions": QUESTIONS_JSON,
    }

    # questions ride in invocation parameters, like a response schema would.
    invocation_parameters = json.loads(str(attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS]))
    assert invocation_parameters == {"model": "jev-latest", "questions": QUESTIONS_JSON}

    # A System One call is not a chat exchange, so nothing is recorded as messages.
    assert not any(key.startswith(SpanAttributes.LLM_INPUT_MESSAGES) for key in attrs)
    assert not any(key.startswith(SpanAttributes.LLM_OUTPUT_MESSAGES) for key in attrs)

    # output.value mirrors the wire response body.
    assert attrs[SpanAttributes.OUTPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
    assert json.loads(str(attrs[SpanAttributes.OUTPUT_VALUE])) == SYSTEM_ONE_RESPONSE

    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 344
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 65
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 409


def test_oitracer() -> None:
    assert isinstance(TypeSafeAIInstrumentor()._tracer, OITracer)


def test_entrypoint_for_opentelemetry_instrument() -> None:
    (entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="typesafe")
    assert isinstance(entrypoint.load()(), TypeSafeAIInstrumentor)


def test_system_one(
    in_memory_span_exporter: InMemorySpanExporter,
    transport: RecordingTransport,
    client: TypeSafeClient,
) -> None:
    response = client.system_one(STATE, QUESTIONS)
    assert response.choices["tone"].choice == "angry"

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.name == "TypeSafeClient"
    _assert_llm_span(span, transport)


async def test_async_system_one(
    in_memory_span_exporter: InMemorySpanExporter, transport: RecordingTransport
) -> None:
    async with AsyncTypeSafeClient(transport=transport.mock) as async_client:
        response = await async_client.system_one(state=STATE, questions=QUESTIONS)
    assert response.nouls["billing"].noul == pytest.approx(0.98)

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.name == "AsyncTypeSafeClient"
    _assert_llm_span(span, transport)


def test_structured_state_and_raw_dict_questions(
    in_memory_span_exporter: InMemorySpanExporter, transport: RecordingTransport
) -> None:
    state: JSONContent = {"ticket": {"subject": "Charged twice", "messages": [{"text": STATE}]}}
    questions: Dict[str, Question] = {
        "billing": {"type": "noul", "instructions": "Is `ticket` about billing?"}
    }
    client = TypeSafeClient(transport=transport.mock, model="jev-preview")
    client.system_one(state, questions, extra_body={"beam_width": 4})

    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = _attrs(span)
    # Structured state rides in input.value as sent on the wire.
    assert json.loads(str(attrs[SpanAttributes.INPUT_VALUE]))["state"] == state
    # The client-level default model is picked up when no per-call model is given.
    assert attrs[SpanAttributes.LLM_REQUEST_MODEL_NAME] == "jev-preview"
    invocation_parameters = json.loads(str(attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS]))
    assert invocation_parameters == {
        "model": "jev-preview",
        "questions": questions,
        "beam_width": 4,
    }
    assert json.loads(str(attrs[SpanAttributes.INPUT_VALUE])) == transport.requests[-1]


def test_error_sets_span_status(in_memory_span_exporter: InMemorySpanExporter) -> None:
    transport = RecordingTransport(status_code=401, body={"error": "invalid api key"})
    client = TypeSafeClient(transport=transport.mock, retry=RetryPolicy(max_retries=0))
    with pytest.raises(TypeSafeAuthenticationError):
        client.system_one(STATE, QUESTIONS)

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert "TypeSafeAuthenticationError" in (span.status.description or "")
    assert any(event.name == "exception" for event in span.events)
    attrs = _attrs(span)
    # Request-side attributes are still recorded; there is no output.
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert SpanAttributes.INPUT_VALUE in attrs
    assert SpanAttributes.OUTPUT_VALUE not in attrs


def test_suppress_tracing(
    in_memory_span_exporter: InMemorySpanExporter, client: TypeSafeClient
) -> None:
    with suppress_tracing():
        client.system_one(STATE, QUESTIONS)
    assert len(in_memory_span_exporter.get_finished_spans()) == 0
    client.system_one(STATE, QUESTIONS)
    assert len(in_memory_span_exporter.get_finished_spans()) == 1


def test_context_attributes(
    in_memory_span_exporter: InMemorySpanExporter, client: TypeSafeClient
) -> None:
    with using_attributes(
        session_id="session-1",
        user_id="user-1",
        metadata={"env": "test"},
        tags=["a", "b"],
    ):
        client.system_one(STATE, QUESTIONS)

    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = _attrs(span)
    assert attrs[SpanAttributes.SESSION_ID] == "session-1"
    assert attrs[SpanAttributes.USER_ID] == "user-1"
    assert json.loads(str(attrs[SpanAttributes.METADATA])) == {"env": "test"}
    assert list(attrs[SpanAttributes.TAG_TAGS]) == ["a", "b"]


def test_trace_config_hides_inputs_and_outputs(
    in_memory_span_exporter: InMemorySpanExporter,
    client: TypeSafeClient,
    tracer_provider: Any,
) -> None:
    TypeSafeAIInstrumentor().uninstrument()
    TypeSafeAIInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=True, hide_outputs=True),
    )
    client.system_one(STATE, QUESTIONS)

    (span,) = in_memory_span_exporter.get_finished_spans()
    attrs = _attrs(span)
    assert attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert attrs[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    # Non-sensitive attributes survive masking.
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "jev-1.13.0"
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 409


def test_uninstrument(
    in_memory_span_exporter: InMemorySpanExporter, client: TypeSafeClient
) -> None:
    TypeSafeAIInstrumentor().uninstrument()
    client.system_one(STATE, QUESTIONS)
    assert len(in_memory_span_exporter.get_finished_spans()) == 0
    assert not hasattr(TypeSafeClient.system_one, "__wrapped__")
