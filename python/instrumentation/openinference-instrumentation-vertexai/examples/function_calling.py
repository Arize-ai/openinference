import vertexai
from openinference.instrumentation.vertexai import VertexAIInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool

endpoint = "http://127.0.0.1:4317"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

VertexAIInstrumentor().instrument(tracer_provider=tracer_provider)

vertexai.init(location="us-central1")

# First, create tools that the model is can use to answer your questions.
# Describe a function by specifying its schema (JsonSchema format)
get_current_weather_func = FunctionDeclaration(
    name="get_current_weather",
    description="Get the current weather in a given location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
)
# Tool is a collection of related functions
weather_tool = Tool(function_declarations=[get_current_weather_func])
# Use tools in chat
chat = GenerativeModel("gemini-1.5-flash", tools=[weather_tool]).start_chat()

if __name__ == "__main__":
    # Send a message to the model. The model will respond with a function call.
    for response in chat.send_message("What is the weather like in Boston?", stream=True):
        print(response)
    # Then send a function response to the model. The model will use it to answer.
    for response in chat.send_message(
        Part.from_function_response(
            name="get_current_weather",
            response={"content": {"weather": "super nice"}},
        ),
        stream=True,
    ):
        print(response)
