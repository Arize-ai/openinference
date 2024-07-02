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

@pytest.fixture()
def tracer_provider() -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(InMemorySpanExporter()))
    return tracer_provider

@pytest.fixture(autouse=True)
def setup_guardrails_instrumentation(tracer_provider: TracerProvider) -> None:
    GuardrailsInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    GuardrailsInstrumentor().uninstrument()

@patch('guardrails.llm_providers.ArbitraryCallable._invoke_llm', return_value=LLMResponse(output="hello harrison you so cool"))
def test_guardrails_instrumentation(mock_invoke_llm, tracer_provider: TracerProvider):
    guard = Guard().use(TwoWords())

    response = guard(
        llm_api=lambda prompt: "yoo whatever",
        prompt="oh look im a bad person",
    )

    print(response)
