import asyncio

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_adk import GoogleADKInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# Optionally, you can also print the spans to the console.
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

agent = Agent(
    name="Assistant",
    model="gemini-2.0-flash-exp",
    description="A helpful assistant.",
    instruction="You are a helpful assistant.",
)


async def main():
    app_name = "test_instrumentation"
    user_id = "test_user"
    session_id = "test_session"
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    session_service = runner.session_service
    session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user", parts=[types.Part(text="Write a haiku about recursion in programming.")]
        ),
    ):
        if event.is_final_response():
            print(event.content.parts[0].text.strip())


if __name__ == "__main__":
    asyncio.run(main())
