from nemoguardrails import RailsConfig
from nemoguardrails import LLMRails
import pytest
import os
from typing import Any, Generator, Tuple, cast
import nest_asyncio
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.instrumentation.nemo_guardrails import NemoGuardrailsInstrumentor

nest_asyncio.apply()

@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_nemo_instrumentor(
        tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    GuardrailsInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    GuardrailsInstrumentor().uninstrument()



@pytest.fixture()
def setup_nemo_instrumentation(
        tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    NemoGuardrailsInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    NemoGuardrailsInstrumentor().uninstrument()

def test_rails(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_nemo_instrumentation: Any,
):
    os.environ['OPENAI_API_KEY'] = 'fake_key'
    config = RailsConfig.from_path("./config")
    rails = LLMRails(config)
    response = rails.generate(messages=[{
        "role": "user",
        "content": "Hello!"
    }])
    print(response)