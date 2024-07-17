# FIXME: This example isn't working due to a bug inside vertexai==1.49.0
import vertexai
from openinference.instrumentation.vertexai import VertexAIInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from vertexai.preview.generative_models import (
    AutomaticFunctionCallingResponder,
    FunctionDeclaration,
    GenerativeModel,
    Tool,
)

endpoint = "http://127.0.0.1:4317"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

VertexAIInstrumentor().instrument(tracer_provider=tracer_provider)

vertexai.init(location="us-central1")


# First, create functions that the model can use to answer your questions.
def get_current_weather(location: str, unit: str = "centigrade"):
    """Gets weather in the specified location.

    Args:
        location: The location for which to get the weather.
        unit: Optional. Temperature unit. Can be Centigrade or Fahrenheit. Defaults to Centigrade.
    """
    return dict(location=location, unit=unit, weather="Super nice.")


# Infer function schema
get_current_weather_func = FunctionDeclaration.from_func(get_current_weather)
# Tool is a collection of related functions
weather_tool = Tool(function_declarations=[get_current_weather_func])
# Use tools in model
model = GenerativeModel("gemini-1.5-flash", tools=[weather_tool])
# Activate automatic function calling
chat = model.start_chat(responder=AutomaticFunctionCallingResponder())

if __name__ == "__main__":
    # Send a message to the model. The model will respond with a function call.
    # The SDK will automatically call the requested function and respond to the model.
    # The model will use the function call response to answer the original question.
    print(chat.send_message("What is the weather like in Boston?"))
