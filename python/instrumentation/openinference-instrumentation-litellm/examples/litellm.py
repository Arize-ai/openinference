import os
import phoenix as px
import litellm

# Get the secret key from environment variables
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Launch Phoenix app
session = px.launch_app()

# Import OpenTelemetry components
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.litellm import LiteLLMInstrumentor

# Set up OpenTelemetry tracing
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

# Simple single message completion call
litellm.completion(
    model="gpt-3.5-turbo", 
    messages=[{"content": "What's the capital of China?", "role": "user"}]
)

# Multiple message conversation completion call with added param
litellm.completion(
    model="gpt-3.5-turbo",
    messages=[
        {"content": "Hello, I want to bake a cake", "role": "user"},
        {"content": "Hello, I can pull up some recipes for cakes.", "role": "assistant"},
        {"content": "No actually I want to make a pie", "role": "user"}
    ],
    temperature=0.7
)

# Multiple message conversation acompletion call with added params
await litellm.acompletion(
    model="gpt-3.5-turbo",
    messages=[
        {"content": "Hello, I want to bake a cake", "role": "user"},
        {"content": "Hello, I can pull up some recipes for cakes.", "role": "assistant"},
        {"content": "No actually I want to make a pie", "role": "user"}
    ],
    temperature=0.7,
    max_tokens=20
)

# Completion with retries
litellm.completion_with_retries(
    model="gpt-3.5-turbo",
    messages=[{"content": "What's the highest grossing film ever", "role": "user"}]
)

# Embedding call
litellm.embedding(model='text-embedding-ada-002', input=["good morning from litellm"])

# Asynchronous embedding call
await litellm.aembedding(model='text-embedding-ada-002', input=["good morning from litellm"])

# Image generation call
litellm.image_generation(model='dall-e-2', prompt="cute baby otter")

# Asynchronous image generation call
await litellm.aimage_generation(model='dall-e-2', prompt="cute baby otter")
