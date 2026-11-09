import base64
import json
from typing import Any

import httpx
import pytest
from google import genai
from google.genai import types
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import using_trace_config

_IMAGE_BYTES = b"\x89PNG-secret-image-bytes"
_IMAGE_B64 = base64.b64encode(_IMAGE_BYTES).decode()

_RESPONSE = {
    "candidates": [
        {
            "content": {"role": "model", "parts": [{"text": "reply"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
}


@pytest.fixture
def client() -> genai.Client:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(":streamGenerateContent"):
            return httpx.Response(200, text=f"data: {json.dumps(_RESPONSE)}\n\n")
        return httpx.Response(200, json=_RESPONSE)

    transport = httpx.MockTransport(respond)
    return genai.Client(
        api_key="test-key",
        http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=transport),
            httpx_async_client=httpx.AsyncClient(transport=transport),
        ),
    )


def _contents() -> Any:
    return [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="describe"),
                types.Part.from_bytes(data=_IMAGE_BYTES, mime_type="image/png"),
            ],
        )
    ]


async def _call(client: genai.Client, asynchronous: bool, streaming: bool) -> None:
    kwargs: Any = dict(model="gemini-2.0-flash", contents=_contents())
    if asynchronous:
        if streaming:
            stream = await client.aio.models.generate_content_stream(**kwargs)
            async for _ in stream:
                pass
        else:
            await client.aio.models.generate_content(**kwargs)
    elif streaming:
        for _ in client.models.generate_content_stream(**kwargs):
            pass
    else:
        client.models.generate_content(**kwargs)


def _span_attributes(exporter: InMemorySpanExporter) -> dict[str, Any]:
    (span,) = exporter.get_finished_spans()
    return dict(span.attributes or {})


def _contains_image(attributes: dict[str, Any]) -> bool:
    return any(_IMAGE_B64 in str(value) for value in attributes.values())


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("streaming", [False, True], ids=["non_streaming", "streaming"])
async def test_override_hides_input_images_inside_input_value(
    asynchronous: bool,
    streaming: bool,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_google_genai_instrumentation: None,
) -> None:
    await _call(client, asynchronous, streaming)
    assert _contains_image(_span_attributes(in_memory_span_exporter))
    in_memory_span_exporter.clear()

    with using_trace_config(hide_input_images=True):
        await _call(client, asynchronous, streaming)
    attributes = _span_attributes(in_memory_span_exporter)
    assert "describe" in str(attributes["input.value"])
    assert not _contains_image(attributes)


@pytest.mark.parametrize("streaming", [False, True], ids=["non_streaming", "streaming"])
async def test_override_image_length_limit_reaches_input_value(
    streaming: bool,
    client: genai.Client,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_google_genai_instrumentation: None,
) -> None:
    # The base64 length limit is applied only by the instrumentor when it builds input.value.
    with using_trace_config(base64_image_max_length=1):
        await _call(client, asynchronous=False, streaming=streaming)
    attributes = _span_attributes(in_memory_span_exporter)
    assert _IMAGE_B64 not in str(attributes["input.value"])
