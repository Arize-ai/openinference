import pytest
from guardrails import Guard
from guardrails.validators import TwoWords
from guardrails.utils.llm_response import LLMResponse
from unittest.mock import patch
from openinference.instrumentation.guardrails import GuardrailsInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.resources import Resource

@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()

@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider

@pytest.fixture(autouse=True)
def setup_guardrails_instrumentation(tracer_provider: TracerProvider) -> None:
    GuardrailsInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    GuardrailsInstrumentor().uninstrument()

@patch('guardrails.llm_providers.ArbitraryCallable._invoke_llm', return_value=LLMResponse(output="More Than Two"))
def test_guardrails_instrumentation(mock_invoke_llm, tracer_provider: TracerProvider, in_memory_span_exporter: InMemorySpanExporter):

    # we expect the guard to raise an exception here because the mock LLMResponse has more than two words
    guard = Guard().use(TwoWords, on_fail="exception")
    with pytest.raises(Exception):
        guard(
            llm_api=lambda prompt: "yoo whatever",
            prompt="oh look im a bad person",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) >= 3  # Expecting at least 3 spans from the guardrails module

    expected_span_names = {"invoke_llm", "post_validation", "guard_parse"}
    found_span_names = set(span.name for span in spans)
    assert expected_span_names.issubset(found_span_names), "Missing expected spans"

    for span in spans:
        if span.name == "invoke_llm":
            assert span.attributes["openinference.span.kind"] == "LLM"
            assert span.status.is_ok

        elif span.name == "post_validation":
            assert span.attributes["validator_name"] == "two-words"
            assert span.attributes["validator_on_fail"].name == "EXCEPTION"
            
            # note that validator result should fail because the mock response returns a 3 letter response
            assert span.attributes["validator_result"] == "fail"
            # this may be counter intuitive but the exception from the validator actually occurs in the span for guard_parse, not post_validation
            assert span.status.is_ok, "post_validation span status should be OK"

        elif span.name == "guard_parse":
            assert span.attributes["openinference.span.kind"] == "GUARDRAIL"
            assert "input.value" in span.attributes
            assert not span.status.is_ok, "guard_parse span status should not be OK"
