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
    setup_autogen_instrumentation: Any,
) -> None:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.ui import Console
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    # Define a model client with a real API key for recording
    model_client = OpenAIChatCompletionClient(
        model="gpt-3.5-turbo",  # Use a real model for recording
        api_key="sk-proj",  # Use a test key that will be filtered by vcr
    )

    # Define a simple function tool that the agent can use
    def get_weather(city: str) -> str:
        """Get the weather for a given city."""
        return f"The weather in {city} is 73 degrees and Sunny."

    # Define an AssistantAgent with the model, tool, system message, and reflection enabled
    agent = AssistantAgent(
        name="weather_agent",
        model_client=model_client,
        tools=[get_weather],
        system_message="You are a helpful assistant that can check the weather.",
        reflect_on_tool_use=True,
        model_client_stream=True,
    )

    # Run the agent and stream the messages to the console
    result = await agent.run(task="What is the weather in New York?")
    await model_client.close()

    # Verify that spans were created
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0, "Expected spans to be created"
    
    # # Verify the weather tool was called
    # weather_spans = [span for span in spans if span.name == "get_weather"]
    # assert len(weather_spans) > 0, "Expected weather tool to be called"   