"""
Sends multimodal user messages through Portkey: a text part with an image part (once as a list
and once as a tuple), and a text part with an `input_audio` part. Each call produces one LLM span
whose input message carries the parts as `message.contents`. The audio part is recorded as a
`data:audio/wav;base64,...` URL.

1. Run a local OTLP collector such as Phoenix: `uvx arize-phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set `PORTKEY_API_KEY`, fill in `PROVIDER_SLUG` below and run this example:
   `python multimodal_input.py`
4. View the traces at http://localhost:6006
"""

import base64
import io
import math
import os
import struct
import wave

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from portkey_ai import Portkey

from openinference.instrumentation.portkey import PortkeyInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

PortkeyInstrumentor().instrument(tracer_provider=tracer_provider)

# Slug of a Portkey provider with access to OpenAI models, e.g. "openai".
PROVIDER_SLUG = ""
IMAGE_URL = "https://www.gstatic.com/webp/gallery/1.jpg"


def _tone_wav_base64(seconds: float = 1.0, frequency: float = 440.0) -> str:
    rate = 16000
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(
            b"".join(
                struct.pack("<h", int(8000 * math.sin(2 * math.pi * frequency * i / rate)))
                for i in range(int(rate * seconds))
            )
        )
    return base64.b64encode(buffer.getvalue()).decode()


if __name__ == "__main__":
    client = Portkey(api_key=os.getenv("PORTKEY_API_KEY", ""))

    response = client.chat.completions.create(
        model=f"@{PROVIDER_SLUG}/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                ],
            }
        ],
    )
    print(response.choices[0].message.content)

    # The SDK accepts any iterable of content parts, so a tuple works too.
    response = client.chat.completions.create(
        model=f"@{PROVIDER_SLUG}/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    {"type": "text", "text": "What colors stand out in this image?"},
                    {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                ),
            }
        ],
    )
    print(response.choices[0].message.content)

    response = client.chat.completions.create(
        model=f"@{PROVIDER_SLUG}/gpt-audio-mini",
        modalities=["text"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this sound in one sentence."},
                    {
                        "type": "input_audio",
                        "input_audio": {"data": _tone_wav_base64(), "format": "wav"},
                    },
                ],
            }
        ],
    )
    print(response.choices[0].message.content)
