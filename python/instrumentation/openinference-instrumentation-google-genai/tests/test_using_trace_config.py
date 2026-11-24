import base64
import json
from typing import Any

import httpx2
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
def client(mock_gemini_api: Any) -> genai.Client:
    mock_gemini_api.post(
        r":streamGenerateContent$",
        httpx2.Response(
            200,
            text=f"data: {json.dumps(_RESPONSE)}\n\n",
            headers={"content-type": "text/event-stream"},
        ),
    )
    mock_gemini_api.post(r":generateContent$", httpx2.Response(200, json=_RESPONSE))
    return genai.Client(api_key="test-key")


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
