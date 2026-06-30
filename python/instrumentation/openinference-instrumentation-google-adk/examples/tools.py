import asyncio

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.google_adk import GoogleADKInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }


agent = Agent(
    name="test_agent",
    model="gemini-2.0-flash",
    description="Agent to answer questions using tools.",
    instruction="You must use the available tools to find an answer.",
    tools=[get_weather],
)


async def main():
    app_name = "test_instrumentation"
    user_id = "test_user"
    session_id = "test_session"
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session_service = runner.session_service
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user", parts=[types.Part(text="What is the weather in New York?")]
        ),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text.strip())


if __name__ == "__main__":
    asyncio.run(main())
