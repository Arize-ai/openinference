from typing import Any, Generator

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import OITracer
from openinference.instrumentation.autogen import AutogenInstrumentor


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
def setup_autogen_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    AutogenInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AutogenInstrumentor().uninstrument()


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(
            group="opentelemetry_instrumentor", name="autogen"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, AutogenInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self, setup_autogen_instrumentation: Any) -> None:
        assert isinstance(AutogenInstrumentor()._tracer, OITracer)


@pytest.mark.asyncio
@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
)
async def test_autogen_chat_agent(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_autogen_instrumentation: AutogenInstrumentor,
) -> None:
    from autogen_core.models import UserMessage
    from autogen_ext.models.anthropic import AnthropicChatCompletionClient

    model_client = AnthropicChatCompletionClient(model="claude-3-7-sonnet-20250219", api_key="sk-")

    result = await model_client.create(
        [UserMessage(content="What is the capital of France?", source="user")]
    )
    print(result)
    await model_client.close()
    spans = in_memory_span_exporter.get_finished_spans()
    print(spans)
