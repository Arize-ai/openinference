from typing import Optional

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from smolagents import (
    LiteLLMModel,
    tool,
)
from smolagents.agents import ToolCallingAgent

from openinference.instrumentation.smolagents import SmolagentsInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)

# Choose which LLM engine to use!
# model = HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct")
# model = TransformersModel(model_id="meta-llama/Llama-3.2-2B-Instruct")

# For anthropic: change model_id below to 'anthropic/claude-3-5-sonnet-20240620'
model = LiteLLMModel(model_id="gpt-4o")


@tool
def get_weather(location: str, celsius: Optional[bool] = False) -> str:
    """
    Get weather in the next days at given location.
    Secretly this tool does not care about the location, it hates the weather everywhere.

    Args:
        location: the location
        celsius: the temperature
    """
    return "The weather is UNGODLY with torrential rains and temperatures below -10°C"


@tool
def get_population(location: str) -> tuple:
    """
    Get Population of the location and location type for the given location.

    Args:
        location: the location
    """
    return f"Population In {location} is 10 million", "City"


if __name__ == "__main__":
    agent = ToolCallingAgent(tools=[get_weather, get_population], model=model)
    print(agent.run("What's the population in Paris?"))
    print(agent.run("What's the weather like in Paris?"))
