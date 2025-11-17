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
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.langchain import LangChainInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


# Custom middleware
class LoggingMiddleware(AgentMiddleware):
    """Custom middleware that logs model calls."""
    
    def before_model(self, state: dict[str, Any], runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        print(f"Calling model with {len(messages)} messages")
        return None
    
    def after_model(self, state: dict[str, Any], runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
            print(f"Model wants to call {len(messages[-1].tool_calls)} tool(s)")
        return None


# Tools
def get_current_time() -> str:
    """Get the current time."""
    from datetime import datetime
    return datetime.now().strftime("%I:%M %p")


def get_random_fact() -> str:
    """Get a random interesting fact."""
    facts = [
        "Honey never spoils. Archaeologists have found 3000-year-old honey that's still edible.",
        "A group of flamingos is called a 'flamboyance'.",
        "The shortest war in history lasted only 38-45 minutes.",
        "Bananas are berries, but strawberries aren't.",
        "There are more stars in the universe than grains of sand on Earth.",
    ]
    import random
    return random.choice(facts)


def calculate_fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    agent = create_agent(
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        tools=[get_current_time, get_random_fact, calculate_fibonacci],
        system_prompt=(
            "You are a helpful assistant with access to tools. "
            "Use them to provide accurate and interesting information."
        ),
        middleware=[LoggingMiddleware()],
    )
    
    queries = [
        "What time is it?",
        "Tell me an interesting fact",
        "Calculate the 10th Fibonacci number",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        messages = result.get("messages", [])
        if messages:
            print(f"Response: {messages[-1].content}")
        print()

