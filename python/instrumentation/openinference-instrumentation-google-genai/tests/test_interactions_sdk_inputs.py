import json
import os
from typing import Any, cast

import pytest
from google import genai
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import REDACTED_VALUE, TraceConfig
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from openinference.semconv.trace import SpanAttributes

TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADkl"
    "EQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII="
)

INTERACTIONS_CREATE_CASSETTE = "test_interactions_create"
INTERACTIONS_CREATE_MATCH_ON = [
    "method_case_insensitive",
    "scheme",
    "host",
    "port",
    "path",
]


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "REDACTED"))


@pytest.mark.default_cassette(INTERACTIONS_CREATE_CASSETTE)
@pytest.mark.vcr(match_on=INTERACTIONS_CREATE_MATCH_ON)
def test_interactions_create_accepts_tuple_image_input(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    image = {
        "type": "image",
        "data": TINY_PNG_BASE64,
        "mime_type": "image/png",
    }
    client = _client()

    interaction = client.interactions.create(
        model="gemini-2.5-flash",
        input=cast(Any, (image,)),
    )

    assert interaction is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    input_value = cast(str, dict(spans[0].attributes or {})[SpanAttributes.INPUT_VALUE])
    assert json.loads(input_value) == [image]


@pytest.mark.default_cassette(INTERACTIONS_CREATE_CASSETTE)
@pytest.mark.vcr(match_on=INTERACTIONS_CREATE_MATCH_ON)
def test_interactions_create_accepts_mime_less_image_input(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    image = {
        "type": "image",
        "data": TINY_PNG_BASE64,
    }
    client = _client()

    interaction = client.interactions.create(
        model="gemini-2.5-flash",
        input=cast(Any, [image]),
    )

    assert interaction is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    input_value = cast(str, dict(spans[0].attributes or {})[SpanAttributes.INPUT_VALUE])
    assert json.loads(input_value) == [image]


@pytest.mark.default_cassette(INTERACTIONS_CREATE_CASSETTE)
@pytest.mark.vcr(match_on=INTERACTIONS_CREATE_MATCH_ON)
def test_interactions_create_redacts_image_input(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
) -> None:
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_input_images=True),
    )
    image = {
        "type": "image",
        "data": TINY_PNG_BASE64,
        "mime_type": "image/png",
    }
    interaction_input: list[dict[str, Any]] = [
        {"type": "text", "text": "Describe this image in one sentence."},
        image,
    ]
    client = _client()

    try:
        interaction = client.interactions.create(
            model="gemini-2.5-flash",
            input=cast(Any, interaction_input),
        )
    finally:
        instrumentor.uninstrument()

    assert interaction is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    input_value = cast(str, dict(spans[0].attributes or {})[SpanAttributes.INPUT_VALUE])
    input_payload = cast(list[dict[str, Any]], json.loads(input_value))
    assert input_payload[0]["text"] == "Describe this image in one sentence."
    assert input_payload[1]["data"] == REDACTED_VALUE
    assert input_payload[1]["mime_type"] == "image/png"
