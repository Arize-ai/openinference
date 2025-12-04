# /// script
# dependencies = [
#   "langchain>=1.0.0",
#   "langchain-openai>=0.2.0",
#   "openinference-instrumentation-langchain>=0.1.24",
#   "opentelemetry-sdk>=1.25.0",
#   "opentelemetry-exporter-otlp>=1.25.0",
# ]
# ///
import os
from typing import Literal

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.langchain import LangChainInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


# Tools
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "San Francisco": "Foggy, 60°F",
        "New York": "Sunny, 75°F",
        "London": "Rainy, 55°F",
        "Tokyo": "Clear, 70°F",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


def calculate(operation: Literal["add", "subtract", "multiply", "divide"], a: float, b: float) -> float:
    """Perform a mathematical calculation."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            return float("inf")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Here are the top results for '{query}': [Result 1] [Result 2] [Result 3]"


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, search_web],
        system_prompt=(
            "You are a helpful assistant with access to weather information, "
            "a calculator, and web search. Use these tools to help answer questions."
        ),
    )
    
    queries = [
        "What's the weather in San Francisco?",
        "Calculate 234 * 567",
        "What's the weather in Tokyo and multiply the temperature by 2?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = result.get("messages", [])
        if messages:
            print(f"Response: {messages[-1].content}")
        print()

