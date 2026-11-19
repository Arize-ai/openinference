import os
from pathlib import Path
from typing import Any, Dict

import pytest
from google import genai
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import SpanAttributes

# Context caching is not served on the Gemini API's `v1` endpoint, the default from
# google-genai 3.0 on, so cache clients have to opt into `v1beta`.
CACHE_HTTP_OPTIONS = genai.types.HttpOptions(api_version="v1beta")


def normalize_request(r: Any) -> Any:
    r.method = r.method.lower()
    r.headers.clear()
    return r


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_create_cache(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")
    # Initialize the client
    client = genai.Client(api_key=api_key, http_options=CACHE_HTTP_OPTIONS)
    with open(Path(__file__).parent / "fixtures/story.txt") as f:
        story_content = f.read()
    model = "gemini-2.5-flash"
    content = genai.types.Content(
        parts=[genai.types.Part(text=story_content)],
        role="user",
    )
    contents: list[genai.types.ContentUnion] = [content]
    cache_response = client.caches.create(
        model=model,
        config=genai.types.CreateCachedContentConfig(
            display_name="test story cache",
            system_instruction=(
                "You are an expert Agent, and your job is to answer "
                "the user query based on the Context you have access to."
            ),
            contents=contents,
            ttl="300s",
        ),
    )
    assert cache_response is not None
    assert cache_response.display_name == "test story cache"
    cache_response_name = cache_response.name
    assert cache_response_name is not None
    assert cache_response_name.startswith("cachedContents/")
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    expected_attributes: Dict[str, Any] = {
        INPUT_MIME_TYPE: "application/json",
        OUTPUT_MIME_TYPE: "application/json",
        OPENINFERENCE_SPAN_KIND: "CHAIN",
        LLM_TOKEN_COUNT_TOTAL: 1090,
    }
    for key, expected_value in expected_attributes.items():
        assert attributes.pop(key) == expected_value, (
            f"Attribute {key} does not match expected value"
        )
    assert story_content in str(attributes.pop(INPUT_VALUE))
    assert cache_response_name in str(attributes.pop(OUTPUT_VALUE))
    assert attributes.pop("llm.model_name") == "models/gemini-2.5-flash"
    assert not attributes, f"Unexpected attributes found: {attributes}"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=normalize_request,
    before_record_response=lambda _: {**_, "headers": {}},
)
@pytest.mark.asyncio
async def test_create_cache_async(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")
    # Initialize the client
    client = genai.Client(api_key=api_key, http_options=CACHE_HTTP_OPTIONS).aio
    with open(Path(__file__).parent / "fixtures/story.txt") as f:
        story_content = f.read()
    model = "gemini-2.5-flash"
    content = genai.types.Content(
        parts=[genai.types.Part(text=story_content)],
        role="user",
    )
    contents: list[genai.types.ContentUnion] = [content]
    cache_response = await client.caches.create(
        model=model,
        config=genai.types.CreateCachedContentConfig(
            display_name="test story cache",
            system_instruction=(
                "You are an expert Agent, and your job is to answer "
                "the user query based on the Context you have access to."
            ),
            contents=contents,
            ttl="300s",
        ),
    )
    assert cache_response is not None
    assert cache_response.display_name == "test story cache"
    cache_response_name = cache_response.name
    assert cache_response_name is not None
    assert cache_response_name.startswith("cachedContents/")
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    expected_attributes: Dict[str, Any] = {
        INPUT_MIME_TYPE: "application/json",
        OUTPUT_MIME_TYPE: "application/json",
        OPENINFERENCE_SPAN_KIND: "CHAIN",
        LLM_TOKEN_COUNT_TOTAL: 1090,
    }
    for key, expected_value in expected_attributes.items():
        assert attributes.pop(key) == expected_value, (
            f"Attribute {key} does not match expected value"
        )
    assert story_content in str(attributes.pop(INPUT_VALUE))
    assert cache_response_name in str(attributes.pop(OUTPUT_VALUE))
    assert attributes.pop("llm.model_name") == "models/gemini-2.5-flash"
    assert not attributes, f"Unexpected attributes found: {attributes}"


LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
