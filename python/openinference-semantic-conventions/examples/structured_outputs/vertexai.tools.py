import json
from pathlib import Path

import vertexai
from openinference.instrumentation.vertexai import VertexAIInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Tool,
    ToolConfig,
)

endpoint = "http://127.0.0.1:4317"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
in_memory_span_exporter = InMemorySpanExporter()
tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))

VertexAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Initialize Vertex AI
vertexai.init(location="us-central1")

# Specify a function declaration and parameters for an API request
get_product_sku_func = FunctionDeclaration(
    name="get_product_sku",
    description="Get the available inventory for a Google products, e.g: Pixel phones, "
    "Pixel Watches, Google Home etc",
    # Function parameters are specified in OpenAPI JSON schema format
    parameters={
        "type": "object",
        "properties": {"product_name": {"type": "string", "description": "Product name"}},
    },
)

# Specify another function declaration and parameters for an API request
get_store_location_func = FunctionDeclaration(
    name="get_store_location",
    description="Get the location of the closest store",
    # Function parameters are specified in OpenAPI JSON schema format
    parameters={
        "type": "object",
        "properties": {"location": {"type": "string", "description": "Location"}},
    },
)

# Define a tool that includes the above functions
retail_tool = Tool(
    function_declarations=[
        get_product_sku_func,
        get_store_location_func,
    ],
)

# Define a tool config for the above functions
retail_tool_config = ToolConfig(
    function_calling_config=ToolConfig.FunctionCallingConfig(
        # ANY mode forces the model to predict a function call
        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        # List of functions that can be returned when the mode is ANY.
        # If the list is empty, any declared function can be returned.
        allowed_function_names=["get_product_sku"],
    )
)

model = GenerativeModel(
    model_name="gemini-1.5-pro",
    tools=[retail_tool],
    tool_config=retail_tool_config,
)

response = model.generate_content(
    "Do you have the Pixel 8 Pro 128GB in stock?",
)
pairs = [
    {
        f"REQUEST-{i}": json.loads(span.attributes["input.value"]),
        f"RESPONSE-{i}": json.loads(span.attributes["output.value"]),
    }
    for i, span in enumerate(in_memory_span_exporter.get_finished_spans())
]
with open(Path(__file__).with_suffix(".json"), "w") as f:
    f.write(json.dumps(pairs, indent=2))
