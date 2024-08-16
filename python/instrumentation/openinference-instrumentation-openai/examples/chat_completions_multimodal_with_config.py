import base64
import os

import openai
import requests
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))


if __name__ == "__main__":
    client = openai.OpenAI()

    img_url = "https://a1cf74336522e87f135f-2f21ace9a6cf0052456644b80fa06d4f.ssl.cf2.rackcdn.com/images/characters/large/800/Homer-Simpson.The-Simpsons.webp"
    img_response = requests.get(img_url)
    if img_response.status_code != 200:
        raise ValueError("Error: Could not retrieve image from URL.")
    base64str = base64.b64encode(img_response.content).decode("utf-8")
    # Set up environment variables:
    os.environ["OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH"] = str(
        10_000
    )  # Base64 encoded images with more than 10k character will appear as __REDACTED__
    os.environ["OPENINFERENCE_HIDE_INPUTS"] = str(True)  # Will hide all inputs
    config = TraceConfig(
        hide_inputs=False,  # Overwrites the environment value setting
        hide_output_text=True,  # The text in output messages will appear as __REDACTED__
        base64_image_max_length=100_000,
    )
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64str}",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    print(response.choices[0].message.content)
