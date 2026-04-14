"""Example: OpenInference tracing for OpenAI Agents RealtimeAgent.

This example shows how to instrument a RealtimeAgent session with
OpenInference tracing so that spans appear in Phoenix or any
OpenTelemetry-compatible backend.

Prerequisites:
    pip install openai-agents openinference-instrumentation-openai-agents
    pip install opentelemetry-exporter-otlp-proto-http

    # Start Phoenix locally:
    pip install arize-phoenix
    python -m phoenix.server.main &

Usage:
    OPENAI_API_KEY=<your-key> python realtime_agent.py

Spans produced for a RealtimeAgent session:
  - RealtimeSession: <agent_name>   [AGENT - root, one per session]
    - Agent: <name>                 [AGENT - one per agent turn]
      - Tool: <tool_name>           [TOOL  - one per tool call]
      - Handoff: A -> B             [TOOL  - one per handoff]
      - Guardrail: <name>           [CHAIN - one per triggered guardrail]
"""

import asyncio

from agents import function_tool
from agents.realtime import RealtimeAgent, RealtimeRunner
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

# ---------------------------------------------------------------------------
# Setup tracing — send spans to Phoenix at localhost:6006
# ---------------------------------------------------------------------------
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)

# ---------------------------------------------------------------------------
# Define tools and agent
# ---------------------------------------------------------------------------


@function_tool
def get_current_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real application, call a weather API here.
    return f"The weather in {city} is sunny and 72°F."


assistant = RealtimeAgent(
    name="weather_assistant",
    instructions=(
        "You are a helpful voice assistant that answers questions about the weather. "
        "Use the get_current_weather tool to look up weather information."
    ),
    tools=[get_current_weather],
)

# ---------------------------------------------------------------------------
# Run a voice session
# ---------------------------------------------------------------------------


async def main() -> None:
    runner = RealtimeRunner(assistant)

    print("Starting realtime session (press Ctrl+C to stop)...")
    print("Traces will appear in Phoenix at http://localhost:6006")
    print()

    async with await runner.run() as session:
        # Send a text message (in a real app you'd stream audio from a microphone)
        await session.send_message("What's the weather in San Francisco?")

        # Iterate session events
        async for event in session:
            if event.type == "agent_start":
                print(f"[agent_start] agent={event.agent.name}")
            elif event.type == "tool_start":
                print(f"[tool_start]  tool={event.tool.name}")
            elif event.type == "tool_end":
                print(f"[tool_end]    tool={event.tool.name}  output={event.output!r}")
            elif event.type == "audio":
                # In a real app, play event.audio to the user's speaker
                print(f"[audio]       {len(event.audio.data)} bytes received")
            elif event.type == "audio_end":
                print("[audio_end]   model finished speaking")
                # Break after the first complete response
                break
            elif event.type == "error":
                print(f"[error]       {event.error}")
                break

    print("\nSession ended. Check Phoenix for traces.")


if __name__ == "__main__":
    asyncio.run(main())
