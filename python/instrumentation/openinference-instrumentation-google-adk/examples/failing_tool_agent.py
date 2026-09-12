"""Span status on failure paths.

Runs an agent whose tool raises, so you can see how the failure is reported in
Phoenix (http://localhost:6006, project ``google-adk-failing-tool``):

- ``execute_tool``  -> ERROR, with the exception recorded as a span event
- ``agent_run`` / ``invocation`` -> ERROR (the exception propagates out of the run)
- ``call_llm``      -> OK (the model call that requested the tool succeeded)

Requires ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY`` and a Phoenix instance on port 6006.
"""

import asyncio

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation import TracerProvider
from openinference.instrumentation.google_adk import GoogleADKInstrumentor
from openinference.semconv.resource import ResourceAttributes

tracer_provider = TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "google-adk-failing-tool"})
)
tracer_provider.add_span_processor(
    SimpleSpanProcessor(OTLPSpanExporter("http://localhost:6006/v1/traces"))
)
GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)


def get_weather(city: str) -> dict[str, str]:
    """Retrieves the current weather report for a specified city.

    Args:
        city: The name of the city for which to retrieve the weather report.
    """
    raise RuntimeError(f"weather service unavailable for {city!r}")


agent = Agent(
    name="weather_agent",
    model="gemini-2.5-flash",
    description="Agent to answer questions about the weather in a city.",
    instruction="Use the get_weather tool to answer questions about the weather.",
    tools=[get_weather],
)

APP_NAME = "failing_tool_app"
USER_ID = "user_1"
SESSION_ID = "session_1"


async def main() -> None:
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    await runner.session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    message = types.Content(role="user", parts=[types.Part(text="What is the weather in Paris?")])
    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=message
        ):
            if event.content and event.content.parts:
                print(event.content.parts[0].model_dump_json(exclude_none=True))
    except RuntimeError as error:
        print(f"Agent run failed as expected: {error}")


if __name__ == "__main__":
    asyncio.run(main())
