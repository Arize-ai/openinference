from typing import Any, Generator

import instructor
import os
import pytest
import vcr
from pydantic import BaseModel
import openai
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.instrumentation.instructor import InstructorInstrumentor

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
def setup_instructor_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    InstructorInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    InstructorInstrumentor().uninstrument()

test_vcr = vcr.VCR(
    serializer='yaml',
    cassette_library_dir='fixtures/cassettes/',
    record_mode='never',
    match_on=['uri', 'method'],
)
class UserInfo(BaseModel):
    name: str
    age: int
async def extract():
    client = instructor.from_openai(openai.AsyncOpenAI())
    return await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": "Create a user"},
        ],
        response_model=UserInfo,
    )

@pytest.mark.asyncio
async def test_async_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_instructor_instrumentation: Any,
):
    os.environ["OPENAI_API_KEY"] = "fake_key"
    with test_vcr.use_cassette('async_instructor_instrumentation.yaml', filter_headers=['authorization']):
        user_info = await extract()
        assert user_info.name == "John Doe"
        assert user_info.age == 25

        spans = in_memory_span_exporter.get_finished_spans()

        # We should have 2 spans for what we consider "TOOL" calling
        assert len(spans) == 2
        for span in spans:
            attributes = dict(span.attributes or dict())
            assert attributes.get("openinference.span.kind") in ["TOOL"]

def test_instructor_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_instructor_instrumentation: Any,
) -> None:
    my_vcr = vcr.VCR(
        serializer='yaml',
        cassette_library_dir='fixtures/cassettes/',
        record_mode='none',
        match_on=['uri', 'method'],
    )
    os.environ["OPENAI_API_KEY"] = "fake_key"
    with test_vcr.use_cassette('instructor_instrumentation.yaml', filter_headers=['authorization']):
        client = instructor.from_openai(openai.OpenAI())
        user_info = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=UserInfo,
            messages=[{"role": "user", "content": "John Doe is 30 years old."}],
        )
        assert user_info.name == "John Doe"
        assert user_info.age == 30

        spans = in_memory_span_exporter.get_finished_spans()

        # We should have 2 spans for what we consider "TOOL" calling
        assert len(spans) == 2
        for span in spans:
            attributes = dict(span.attributes or dict())
            assert attributes.get("openinference.span.kind") in ["TOOL"]
