import asyncio

from google.adk.agents import Agent
from google.adk.planners import BuiltInPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation import TracerProvider
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

APP_NAME = "riddle_app"
USER_ID = "12345"
SESSION_ID = "123344"


agent = Agent(
    name="riddle_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a careful logic puzzle solver. Think through the problem "
        "step by step before giving your final answer."
    ),
    planner=BuiltInPlanner(
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=1024,
        ),
    ),
)


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    user_query = (
        "A farmer has 17 sheep. All but 9 die. How many sheep does the "
        "farmer have left? Explain your reasoning before answering."
    )
    print(f"User: {user_query}\n")
    content = types.Content(role="user", parts=[types.Part(text=user_query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    for event in events:
        if event and event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    label = "Thought" if part.thought else "Answer"
                    print(f"Agent [{label}]:\n{part.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
