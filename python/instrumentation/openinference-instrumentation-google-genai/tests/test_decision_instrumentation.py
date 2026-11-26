# type: ignore
# ruff: noqa: E501
"""Tests for tracing the Gemini Decisions API (``client.decisions.create``) as DECISION spans.

Two styles are used, matching the rest of this package:

* ``@pytest.mark.vcr`` happy-path tests recorded against the real API for each question
  type and a mixed request (sync and async).
* ``mock_gemini_api`` tests for cases that are impractical to record: the safety-block 400,
  image redaction, suppress-tracing, context propagation, and ``TraceConfig`` masking.
"""

import json
import os
from typing import Any, Dict, Iterator

import pytest
from google import genai
from httpx2 import Response
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation import (
    REDACTED_VALUE,
    TraceConfig,
    suppress_tracing,
    using_attributes,
)
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from openinference.semconv.trace import SpanAttributes

TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII="

NOUL_QUESTIONS = {"is_refund": {"type": "noul", "instructions": "Is this a refund request?"}}
CHOICE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {"billing": "Charges and refunds", "shipping": "", "other": ""},
    }
}
SCORE_QUESTIONS = {
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"],
    }
}
MIXED_QUESTIONS = {**NOUL_QUESTIONS, **CHOICE_QUESTIONS, **SCORE_QUESTIONS}
INPUT_TEXT = "Hi, I was charged twice for order #4242 and would like my money back."

# A canned response body shaped like the real Gemini Decisions API response, for the
# mock_gemini_api tests. The response carries no "model" field, exactly as the real API.
_DECISION_RESPONSE = json.dumps(
    {
        "answers": {
            "is_refund": {"type": "noul", "noul": 0.97, "confidence": 0.95},
            "department": {
                "type": "choice",
                "choice": "billing",
                "confidence": 0.9,
                "probabilities": {"billing": 0.9, "shipping": 0.05, "other": 0.05},
            },
        },
        "usage": {"total_input_tokens": 213, "total_output_tokens": 0, "total_tokens": 213},
        "prompt_feedback": None,
    }
).encode()


def _mock_decision(mock_gemini_api: Any, response: Response | None = None) -> None:
    mock_gemini_api.post(
        r"/v1beta/decisions.*",
        response
        or Response(200, content=_DECISION_RESPONSE, headers={"content-type": "application/json"}),
    )


def _assert_decision_span(attributes: Dict[str, Any], *, async_: bool = False) -> None:
    assert attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "DECISION"
    assert attributes["decision.system"] == "google"
    assert attributes["decision.provider"] == "google"
    assert attributes["decision.model_name"] == "gemini-decision-1"
    assert attributes["decision.request.model_name"] == "gemini-decision-1"
    # The response does not echo the model, so the response model name is left unset.
    assert "decision.response.model_name" not in attributes
    assert isinstance(attributes["decision.token_count.input"], int)
    assert attributes["decision.token_count.output"] == 0
    assert attributes[SpanAttributes.INPUT_MIME_TYPE] == "application/json"
    assert attributes[SpanAttributes.OUTPUT_MIME_TYPE] == "application/json"
    input_value = json.loads(attributes[SpanAttributes.INPUT_VALUE])
    assert input_value["model"] == "gemini-decision-1"
    assert "questions" in input_value and "input" in input_value
    output_value = json.loads(attributes[SpanAttributes.OUTPUT_VALUE])
    assert "answers" in output_value and "usage" in output_value


# --------------------------------------------------------------------------------------
# VCR happy-path tests (recorded against the real API)
# --------------------------------------------------------------------------------------


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda r: r.headers.clear() or r,
)
@pytest.mark.parametrize(
    "questions",
    [
        pytest.param(NOUL_QUESTIONS, id="noul"),
        pytest.param(CHOICE_QUESTIONS, id="choice"),
        pytest.param(SCORE_QUESTIONS, id="score"),
        pytest.param(MIXED_QUESTIONS, id="mixed"),
    ],
)
def test_decisions_create(
    questions: Dict[str, Any],
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY", "REDACTED"),
        http_options={"api_version": "v1beta"},
    )
    result = client.decisions.create(
        model="gemini-decision-1", input=INPUT_TEXT, questions=questions
    )
    assert result.answers is not None

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "Decisions.create"
    assert span.status.status_code == StatusCode.OK
    _assert_decision_span(dict(span.attributes or {}))


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda r: r.headers.clear() or r,
)
async def test_decisions_create_async(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY", "REDACTED"),
        http_options={"api_version": "v1beta"},
    )
    result = await client.aio.decisions.create(
        model="gemini-decision-1", input=INPUT_TEXT, questions=MIXED_QUESTIONS
    )
    assert result.answers is not None

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "AsyncDecisions.create"
    assert span.status.status_code == StatusCode.OK
    _assert_decision_span(dict(span.attributes or {}), async_=True)


# --------------------------------------------------------------------------------------
# mock_gemini_api tests (no API key / no network)
# --------------------------------------------------------------------------------------


def test_decisions_create_records_span_with_mock(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
    mock_gemini_api: Any,
) -> None:
    _mock_decision(mock_gemini_api)
    client = genai.Client(api_key="fake-key", http_options={"api_version": "v1beta"})
    client.decisions.create(model="gemini-decision-1", input=INPUT_TEXT, questions=MIXED_QUESTIONS)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "Decisions.create"
    assert spans[0].status.status_code == StatusCode.OK
    _assert_decision_span(dict(spans[0].attributes or {}))


def test_decisions_safety_block_sets_error_status(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
    mock_gemini_api: Any,
) -> None:
    error_body = json.dumps(
        {"error": {"code": 400, "message": "The request was blocked.", "status": "safety"}}
    ).encode()
    _mock_decision(
        mock_gemini_api,
        Response(400, content=error_body, headers={"content-type": "application/json"}),
    )
    client = genai.Client(api_key="fake-key", http_options={"api_version": "v1beta"})
    with pytest.raises(Exception):
        client.decisions.create(
            model="gemini-decision-1", input=INPUT_TEXT, questions=NOUL_QUESTIONS
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.ERROR
    assert any(event.name == "exception" for event in span.events)


def test_decisions_suppress_tracing(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
    mock_gemini_api: Any,
) -> None:
    _mock_decision(mock_gemini_api)
    client = genai.Client(api_key="fake-key", http_options={"api_version": "v1beta"})
    with suppress_tracing():
        client.decisions.create(
            model="gemini-decision-1", input=INPUT_TEXT, questions=NOUL_QUESTIONS
        )
    assert len(in_memory_span_exporter.get_finished_spans()) == 0


def test_decisions_context_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
    mock_gemini_api: Any,
) -> None:
    _mock_decision(mock_gemini_api)
    client = genai.Client(api_key="fake-key", http_options={"api_version": "v1beta"})
    with using_attributes(
        session_id="my-session",
        user_id="my-user",
        metadata={"env": "test"},
        tags=["t1", "t2"],
    ):
        client.decisions.create(
            model="gemini-decision-1", input=INPUT_TEXT, questions=NOUL_QUESTIONS
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes[SpanAttributes.SESSION_ID] == "my-session"
    assert attributes[SpanAttributes.USER_ID] == "my-user"
    assert attributes[SpanAttributes.TAG_TAGS] == ("t1", "t2")
    assert json.loads(attributes[SpanAttributes.METADATA]) == {"env": "test"}


@pytest.fixture
def _instrument_with_config(
    tracer_provider: TracerProvider,
) -> Iterator[GoogleGenAIInstrumentor]:
    """Instruments with a caller-chosen TraceConfig, then uninstruments."""
    instrumentor = GoogleGenAIInstrumentor()

    def _instrument(config: TraceConfig) -> None:
        instrumentor.instrument(tracer_provider=tracer_provider, config=config)

    yield _instrument
    instrumentor.uninstrument()


def test_decisions_hide_inputs_and_outputs(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    _instrument_with_config: Any,
    mock_gemini_api: Any,
) -> None:
    _instrument_with_config(TraceConfig(hide_inputs=True, hide_outputs=True))
    _mock_decision(mock_gemini_api)
    client = genai.Client(api_key="fake-key", http_options={"api_version": "v1beta"})
    client.decisions.create(model="gemini-decision-1", input=INPUT_TEXT, questions=NOUL_QUESTIONS)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert attributes[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE


def test_decisions_redacts_image_input(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    _instrument_with_config: Any,
    mock_gemini_api: Any,
) -> None:
    _instrument_with_config(TraceConfig(hide_input_images=True))
    _mock_decision(mock_gemini_api)
    client = genai.Client(api_key="fake-key", http_options={"api_version": "v1beta"})
    client.decisions.create(
        model="gemini-decision-1",
        input=[
            {"type": "text", "text": "Does this image show a cat?"},
            {"type": "image", "mime_type": "image/png", "data": TINY_PNG_BASE64},
        ],
        questions={"is_cat": {"type": "noul", "instructions": "Is it a cat?"}},
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    input_value = dict(spans[0].attributes or {})[SpanAttributes.INPUT_VALUE]
    assert TINY_PNG_BASE64 not in input_value
