import json

import openai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.openai import OpenAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


client = openai.OpenAI()
response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "system",
            "content": "You are a UI generator AI. Convert the user input into a UI.",
        },
        {"role": "user", "content": "Make a User Profile Form"},
    ],
    text={
        "format": {
            "type": "json_schema",
            "name": "ui",
            "description": "Dynamically generated UI",
            "schema": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "The type of the UI component",
                        "enum": ["div", "button", "header", "section", "field", "form"],
                    },
                    "label": {
                        "type": "string",
                        "description": "The label of the UI component, used for "
                        "buttons or form fields",
                    },
                    "children": {
                        "type": "array",
                        "description": "Nested UI components",
                        "items": {"$ref": "#"},
                    },
                    "attributes": {
                        "type": "array",
                        "description": "Arbitrary attributes for the UI component, "
                        "suitable for any element",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "description": "The name of the attribute, for "
                                    "example onClick or className",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "The value of the attribute",
                                },
                            },
                            "required": ["name", "value"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["type", "label", "children", "attributes"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
)

ui = json.loads(response.output_text)
print(ui)
