import asyncio
from typing import Any, Optional

import instructor
import openai
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel


class UserInfo(BaseModel):
    name: str
    age: int


async def extract() -> UserInfo:
    client = instructor.from_openai(openai.AsyncOpenAI())
    return await client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": "Create a user"},
        ],
        response_model=UserInfo,
    )


@pytest.mark.vcr
async def test_async_instrumentation(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    user_info = await extract()
    assert user_info.name == "John Doe"
    assert user_info.age == 30

    spans = in_memory_span_exporter.get_finished_spans()

    # We should have 2 spans for what we consider "TOOL" calling
    assert len(spans) == 2
    for span in spans:
        attributes = dict(span.attributes or dict())
        assert attributes.get("openinference.span.kind") in ["TOOL"]


@pytest.mark.vcr
async def test_streaming_instrumentation(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def run_test() -> Any:
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
        assert attributes.get("openinference.span.kind") in ["TOOL"]


@pytest.mark.vcr
def test_instructor_instrumentation(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
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
