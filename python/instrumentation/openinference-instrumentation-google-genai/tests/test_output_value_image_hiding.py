import json
from typing import Any, AsyncIterator, Iterator, cast

import pytest
from google.genai import types
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import REDACTED_VALUE, OITracer, TraceConfig
from openinference.instrumentation.google_genai._wrappers import (
    _AsyncGenerateContentStream,
    _AsyncGenerateContentWrapper,
    _SyncGenerateContent,
    _SyncGenerateContentStream,
)
from openinference.semconv.trace import SpanAttributes

_IMAGE = b"image"
# Length of the equivalent data URL, matching TraceConfig.mask semantics.
_IMAGE_URL_LENGTH = len("data:image/png;base64,") + len("aW1hZ2U=")


def _response() -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(text="Here is your image."),
                        types.Part.from_bytes(data=_IMAGE, mime_type="image/png"),
                        types.Part.from_bytes(data=b"\xfb\xff", mime_type="audio/wav"),
                    ],
                )
            )
        ]
    )


def _get_output_parts(in_memory_span_exporter: InMemorySpanExporter) -> list[dict[str, Any]]:
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    output_value = cast(str, attributes[SpanAttributes.OUTPUT_VALUE])
    output_payload = cast(dict[str, Any], json.loads(output_value))
    return cast(list[dict[str, Any]], output_payload["candidates"][0]["content"]["parts"])


_LIMIT_PARAMS = [
    pytest.param(0, REDACTED_VALUE, id="zero-limit"),
    pytest.param(_IMAGE_URL_LENGTH - 1, REDACTED_VALUE, id="over-limit"),
    pytest.param(_IMAGE_URL_LENGTH, "aW1hZ2U=", id="at-limit"),
]


@pytest.mark.parametrize("maximum_length, expected_data", _LIMIT_PARAMS)
def test_generate_content_output_value_respects_base64_image_max_length(
    maximum_length: int,
    expected_data: str,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def generate_content(*, model: str, contents: Any) -> types.GenerateContentResponse:
        return _response()

    config = TraceConfig(base64_image_max_length=maximum_length)
    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
    _SyncGenerateContent(tracer=tracer)(
        generate_content,
        None,
        (),
        {"model": "gemini-2.5-flash-image", "contents": "draw a cat"},
    )

    parts = _get_output_parts(in_memory_span_exporter)
    assert parts[0]["text"] == "Here is your image."
    assert parts[1]["inline_data"] == {"data": expected_data, "mime_type": "image/png"}
    # Non-image media is not subject to the image length limit.
    assert parts[2]["inline_data"] == {"data": "-_8=", "mime_type": "audio/wav"}


@pytest.mark.parametrize("maximum_length, expected_data", _LIMIT_PARAMS)
async def test_async_generate_content_output_value_respects_base64_image_max_length(
    maximum_length: int,
    expected_data: str,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async def generate_content(*, model: str, contents: Any) -> types.GenerateContentResponse:
        return _response()

    config = TraceConfig(base64_image_max_length=maximum_length)
    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
    await _AsyncGenerateContentWrapper(tracer=tracer)(
        generate_content,
        None,
        (),
        {"model": "gemini-2.5-flash-image", "contents": "draw a cat"},
    )

    parts = _get_output_parts(in_memory_span_exporter)
    assert parts[1]["inline_data"]["data"] == expected_data


def test_generate_content_stream_output_value_redacts_oversized_image(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    def generate_content_stream(
        *, model: str, contents: Any
    ) -> Iterator[types.GenerateContentResponse]:
        yield _response()

    config = TraceConfig(base64_image_max_length=_IMAGE_URL_LENGTH - 1)
    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
    stream = _SyncGenerateContentStream(tracer=tracer)(
        generate_content_stream,
        None,
        (),
        {"model": "gemini-2.5-flash-image", "contents": "draw a cat"},
    )
    for _ in stream:
        pass

    parts = _get_output_parts(in_memory_span_exporter)
    assert parts[0]["text"] == "Here is your image."
    assert parts[1]["inline_data"] == {"data": REDACTED_VALUE, "mime_type": "image/png"}


async def test_async_generate_content_stream_output_value_redacts_oversized_image(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    async def chunks() -> AsyncIterator[types.GenerateContentResponse]:
        yield _response()

    async def generate_content_stream(
        *, model: str, contents: Any
    ) -> AsyncIterator[types.GenerateContentResponse]:
        return chunks()

    config = TraceConfig(base64_image_max_length=_IMAGE_URL_LENGTH - 1)
    tracer = OITracer(tracer_provider.get_tracer(__name__), config=config)
    stream = await _AsyncGenerateContentStream(tracer=tracer)(
        generate_content_stream,
        None,
        (),
        {"model": "gemini-2.5-flash-image", "contents": "draw a cat"},
    )
    async for _ in stream:
        pass

    parts = _get_output_parts(in_memory_span_exporter)
    assert parts[1]["inline_data"] == {"data": REDACTED_VALUE, "mime_type": "image/png"}


def test_unredacted_output_value_preserves_pydantic_serialization(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    response = _response()

    def generate_content(*, model: str, contents: Any) -> types.GenerateContentResponse:
        return response

    tracer = OITracer(tracer_provider.get_tracer(__name__), config=TraceConfig())
    _SyncGenerateContent(tracer=tracer)(
        generate_content,
        None,
        (),
        {"model": "gemini-2.5-flash-image", "contents": "draw a cat"},
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    # Nothing exceeded the limit, so output.value is the SDK's own JSON serialization.
    assert attributes[SpanAttributes.OUTPUT_VALUE] == response.model_dump_json(exclude_unset=True)
