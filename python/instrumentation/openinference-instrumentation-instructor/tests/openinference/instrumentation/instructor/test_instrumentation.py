import asyncio
import json
import os
from typing import Any, Generator, Optional

import instructor
import openai
import pytest
import vcr  # type: ignore
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points
from pydantic import BaseModel

from openinference.instrumentation import OITracer
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
    serializer="yaml",
    cassette_library_dir="tests/openinference/instrumentation/instructor/fixtures/",
    record_mode="never",
    match_on=["uri", "method"],
)


class UserInfo(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None


async def extract() -> UserInfo:
    client = instructor.from_openai(openai.AsyncOpenAI())
    user_info: UserInfo = await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": "Create a user"},
        ],
        response_model=UserInfo,
    )
    return user_info


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(  # type: ignore[no-untyped-call]
            group="opentelemetry_instrumentor", name="instructor"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, InstructorInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self, setup_instructor_instrumentation: Any) -> None:
        assert isinstance(InstructorInstrumentor()._tracer, OITracer)


@pytest.mark.asyncio
async def test_async_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_instructor_instrumentation: Any,
) -> None:
    os.environ["OPENAI_API_KEY"] = "fake_key"
    with test_vcr.use_cassette(
        "async_instructor_instrumentation.yaml", filter_headers=["authorization"]
    ):
        user_info = await extract()
        assert user_info.name == "John Doe"
        assert user_info.age == 30

        spans = in_memory_span_exporter.get_finished_spans()

        # We should have 2 spans for what we consider "TOOL" calling
        assert len(spans) == 2
        for span in spans:
            attributes = dict(span.attributes or dict())
            if span.name in {"instructor.patch", "instructor.async_patch"}:
                assert attributes.get("llm.model_name") == "gpt-4-turbo-preview"
                assert attributes.get("llm.provider") == "openai"
                assert attributes.get("llm.system") == "openai"
            assert attributes.get("openinference.span.kind") in ["TOOL"]
            assert span.status.status_code == trace_api.StatusCode.OK


@pytest.mark.asyncio
async def test_streaming_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_instructor_instrumentation: Any,
) -> None:
    os.environ["OPENAI_API_KEY"] = "fake_key"

    def run_test() -> Any:
        with test_vcr.use_cassette("streaming.yaml", filter_headers=["authorization"]):
            client = instructor.from_openai(openai.OpenAI())
            user_stream = client.chat.completions.create_partial(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "user", "content": "Create a user"},
                ],
                response_model=UserInfo,
            )
            return user_stream

    user_stream = await asyncio.to_thread(run_test)

    final_user: Optional[UserInfo] = None
    for user in user_stream:
        final_user = user
    assert final_user is not None
    assert final_user.name == "John Doe"
    assert final_user.age == 30

    spans = in_memory_span_exporter.get_finished_spans()

    # We should have 2 spans for what we consider "TOOL" calling
    assert len(spans) == 2
    for span in spans:
        attributes = dict(span.attributes or dict())
        if span.name in {"instructor.patch", "instructor.async_patch"}:
            assert attributes.get("llm.model_name") == "gpt-4-turbo-preview"
            assert attributes.get("llm.provider") == "openai"
            assert attributes.get("llm.system") == "openai"
        assert attributes.get("openinference.span.kind") in ["TOOL"]
        assert span.status.status_code == trace_api.StatusCode.OK


def test_instructor_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_instructor_instrumentation: Any,
) -> None:
    os.environ["OPENAI_API_KEY"] = "fake_key"
    with test_vcr.use_cassette("instructor_instrumentation.yaml", filter_headers=["authorization"]):
        client = instructor.from_openai(openai.OpenAI())
        user_info = client.chat.completions.create(
            model="gpt-3.5-turbo",
            response_model=UserInfo,
            messages=[{"role": "user", "content": "John Doe is 30 years old."}],
            max_retries=1,
        )
        assert user_info.name == "John Doe"
        assert user_info.age == 30

        spans = in_memory_span_exporter.get_finished_spans()

        # We should have 2 spans for what we consider "TOOL" calling
        assert len(spans) == 2
        for span in spans:
            attributes = dict(span.attributes or dict())
            if span.name in {"instructor.patch", "instructor.async_patch"}:
                assert attributes.get("llm.model_name") == "gpt-3.5-turbo"
                assert attributes.get("llm.provider") == "openai"
                assert attributes.get("llm.system") == "openai"
            assert attributes.get("openinference.span.kind") in ["TOOL"]
            assert span.status.status_code == trace_api.StatusCode.OK

            # Validate invocation parameters handling
            invocation_params = attributes.get("llm.invocation_parameters")
            if isinstance(invocation_params, dict):
                # Ensure max_retries key exists
                assert "max_retries" in invocation_params
                # Ensure max_retries is JSON-serializable
                json.dumps(invocation_params)
