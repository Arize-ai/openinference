from typing import Any, Dict, Generator

import pytest
from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.requirements.conditional import (
    ConditionalRequirement,
)
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool
from openinference.instrumentation.beeai import BeeAIInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest
import vcr


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "record_mode": "once",
        "filter_headers": [
            "authorization",
            "api-key",
            "x-api-key",
        ],
        # Match requests on these attributes
        "match_on": ["method", "scheme", "host", "port", "path", "query", "body"],
        # Where to store cassettes
        "cassette_library_dir": "tests/cassettes",
        # Decode compressed responses
        "decode_compressed_response": True,
        # Allow recording of requests
        "allow_playback_repeats": True,

    }


@pytest.fixture(scope="module")
def vcr_cassette_dir(request):
    return "tests/cassettes"


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
async def tracer_provider(
        in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
        tracer_provider: trace_api.TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    BeeAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    BeeAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_instrumentor(
        in_memory_span_exporter: InMemorySpanExporter,
):
    # NOTE: To record new cassettes, set a valid OPENAI_API_KEY in your environment.
    # For replay, a dummy key is fine as long as the cassette matches all requests.
    # os.environ['OPENAI_API_KEY'] = "sk-test-dummy-key"  # Only needed for replay, not for recording

    knowledge_agent = RequirementAgent(
        llm=OpenAIChatModel(model="gpt-4o-mini"),
        tools=[ThinkTool(), WikipediaTool()],
        requirements=[ConditionalRequirement(ThinkTool, force_at_step=1)],
        role="Knowledge Specialist",
        instructions="Provide answers to general questions about the world.",
    )

    weather_agent = RequirementAgent(
        llm=OpenAIChatModel(model="gpt-4o-mini"),
        tools=[OpenMeteoTool()],
        role="Weather Specialist",
        instructions="Provide weather forecast for a given destination.",
    )

    main_agent = RequirementAgent(
        name="MainAgent",
        llm=OpenAIChatModel(model="gpt-4o-mini"),
        # llm=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[
            ThinkTool(),
            HandoffTool(
                knowledge_agent,
                name="KnowledgeLookup",
                description="Consult the Knowledge Agent for general questions.",
            ),
            HandoffTool(
                weather_agent,
                name="WeatherLookup",
                description="Consult the Weather Agent for forecasts.",
            ),
        ],
        requirements=[ConditionalRequirement(ThinkTool, force_at_step=1)],
        # Log all tool calls to the console for easier debugging
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
    )

    question = (
        "If I travel to Rome next weekend, what should I expect in terms of weather, "
        "and also tell me one famous historical landmark there?"
    )
    response = await main_agent.run(question, expected_output="Helpful and clear response.")
    print(response)
    spans = list(in_memory_span_exporter.get_finished_spans())
    spans.sort(key=lambda span: span.start_time or 0)
    assert len(spans) == 18
