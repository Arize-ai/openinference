from typing import Any, Generator

import instructor
import pytest
from pydantic import BaseModel
from openai import OpenAI
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

class UserInfo(BaseModel):
    name: str
    age: int
async def extract():
    import openai
    import os
    client = instructor.from_openai(openai.AsyncOpenAI())
    return await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": "Create a user"},
        ],
        response_model=UserInfo,
    )


@pytest.mark.asyncio
async def test_extract(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_instructor_instrumentation: Any,
):
    user = await extract()
    print(user)
    assert user.name == "John Doe"
    assert user.age == 30

def test_instructor_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_instructor_instrumentation: Any,
) -> None:
    # Patch the OpenAI client
    client = instructor.from_openai(OpenAI())

    # Extract structured data from natural language
    user_info = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=UserInfo,
        messages=[{"role": "user", "content": "John Doe is 30 years old."}],
    )

    assert user_info.name == "John Doe"
    assert user_info.age == 30

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0  # Ensure that spans are created

    for span in spans:
        attributes = dict(span.attributes or dict())
        assert attributes.get("openinference.span.kind") in ["LLM", "INSTRUCTOR"]