# OpenInference Mistral AI Instrumentation
[![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-mistralai.svg)](https://pypi.python.org/pypi/openinference-instrumentation-mistralai) 

Python autoinstrumentation library for MistralAI's Python SDK.

The traces emitted by this instrumentation are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix)

## Installation

```shell
pip install openinference-instrumentation-mistralai
```

## Quickstart

In this example we will instrument a small program that uses the MistralAI chat completions API and observe the traces via [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

Install packages.

```shell
pip install openinference-instrumentation-mistralai mistralai arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start the phoenix server so that it is ready to collect traces.
The Phoenix server runs entirely on your machine and does not send data over the internet.

```shell
python -m phoenix.server.main serve
```

In a python file, setup the `MistralAIInstrumentor` and configure the tracer to send traces to Phoenix.

```python
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openinference.instrumentation.mistralai import MistralAIInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# Optionally, you can also print the spans to the console.
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider)

MistralAIInstrumentor().instrument()


if __name__ == "__main__":
    client = MistralClient()
    response = client.chat(
        model="mistral-large-latest",
        messages=[
            ChatMessage(
                content="Who won the World Cup in 2018?",
                role="user",
            )
        ],
    )
    print(response.choices[0].message.content)

```

Since we are using MistralAI, we must set the `MISTRAL_API_KEY` environment variable to authenticate with the MistralAI API.

```shell
export MISTRAL_API_KEY=[your_key_here]
```

Now simply run the python file and observe the traces in Phoenix.

```shell
python your_file.py
```

## OCR and Input Image Tracing

The MistralAI instrumentation automatically traces input images and documents passed to the OCR API, following OpenInference semantic conventions. This includes:

### Supported Input Types

- **HTTP Image URLs**: `https://example.com/image.jpg`
- **Base64 Images**: `data:image/jpeg;base64,{base64_data}`  
- **PDF URLs**: `https://example.com/document.pdf`
- **Base64 PDFs**: `data:application/pdf;base64,{base64_data}`

### Trace Attributes

For **image inputs**, the instrumentation creates:
- `input.message_content.type`: `"image"`
- `input.message_content.image.image.url`: The image URL or base64 data URL
- `input.message_content.image.metadata`: JSON metadata including source, encoding type, and MIME type

For **document inputs**, the instrumentation creates:
- `input.message_content.type`: `"document"`  
- `input.document.url`: The document URL or base64 data URL
- `input.document.metadata`: JSON metadata including source, encoding type, and MIME type

### Example Usage

```python
import base64
import os
from mistralai import Mistral
from openinference.instrumentation.mistralai import MistralAIInstrumentor

# Set up instrumentation
MistralAIInstrumentor().instrument()

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# OCR with HTTP image URL
response = client.ocr.process(
    model="mistral-ocr-latest",
    document={
        "type": "image_url",
        "image_url": "https://example.com/receipt.png"
    },
    include_image_base64=True
)

# OCR with base64 image  
with open("image.jpg", "rb") as f:
    base64_image = base64.b64encode(f.read()).decode('utf-8')

response = client.ocr.process(
    model="mistral-ocr-latest", 
    document={
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{base64_image}"
    },
    include_image_base64=True
)
```

### Privacy and Configuration

Input image tracing works seamlessly with [TraceConfig](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration) for:

- **Image size limits**: Control maximum base64 image length with `base64_image_max_length`
- **Privacy controls**: Hide input images with `hide_inputs` or `hide_input_images`
- **MIME type detection**: Automatic detection and proper formatting of image data URLs

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
