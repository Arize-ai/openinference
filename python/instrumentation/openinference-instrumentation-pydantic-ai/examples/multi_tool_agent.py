from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai import Agent, RunContext

from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor

# OpenTelemetry setup
endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
exporter = OTLPSpanExporter(endpoint=endpoint)
trace.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
tracer_provider.add_span_processor(BatchSpanProcessor(exporter))


# Simple dependencies
class Deps:
    def __init__(self, user_id: str):
        self.user_id = user_id


# Create agent that will use multiple tools
agent = Agent(
    "openai:gpt-4.1-nano",
    deps_type=Deps,
    system_prompt="You are a helpful assistant. Use available tools to gather information.",
    instrument=True,
)


@agent.tool
def get_weather(ctx: RunContext[Deps], city: str) -> str:
    """Get weather information for a city."""
    print(f"[TOOL 1] Getting weather for: {city}")
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 65°F",
        "Seattle": "Rainy, 58°F",
    }
    return weather_data.get(city, f"Weather data not available for {city}")


@agent.tool
def get_time(ctx: RunContext[Deps], city: str) -> str:
    """Get current time for a city."""
    print(f"[TOOL 2] Getting time for: {city}")
    time_data = {
        "San Francisco": "10:30 AM PST",
        "New York": "1:30 PM EST",
        "Seattle": "10:30 AM PST",
    }
    return time_data.get(city, f"Time data not available for {city}")


def main():
    deps = Deps(user_id="user_123")
    result = agent.run_sync("What's the weather and time in San Francisco?", deps=deps)
    print("AGENT RESPONSE:")
    print(result.output)
    for i, msg in enumerate(result.all_messages(), 1):
        print(f"\nStep {i}: {msg}")
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if part.part_kind == "tool-call":
                    print(f"  → Tool Call: {part.tool_name}({part.args})")
                elif part.part_kind == "tool-return":
                    print(f"  ← Tool Result: {part.content}")
                elif part.part_kind == "text":
                    print(f"  💬 Text: {part.content[:100]}...")


if __name__ == "__main__":
    main()
