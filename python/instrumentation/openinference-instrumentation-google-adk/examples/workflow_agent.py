import asyncio

from google.adk.agents import LlmAgent
from google.adk.apps import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import START, Workflow
from google.genai import types
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation import TracerProvider
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

APP_NAME = "workflow_app"
USER_ID = "12345"
SESSION_ID = "123344"


def normalize_topic(node_input: types.Content) -> str:
    text = "".join(part.text or "" for part in node_input.parts or [])
    return text.strip().lower()


haiku_writer = LlmAgent(
    name="haiku_writer",
    model="gemini-2.5-flash",
    mode="single_turn",
    instruction="Write a single haiku about the given topic. Output only the haiku.",
)


def count_words(node_input: str) -> str:
    return f"{node_input}\n\n({len(node_input.split())} words)"


workflow = Workflow(
    name="haiku_workflow",
    edges=[
        (START, normalize_topic),
        (normalize_topic, haiku_writer),
        (haiku_writer, count_words),
    ],
)


async def main():
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    runner = Runner(
        app=App(name=APP_NAME, root_agent=workflow),
        session_service=session_service,
    )
    content = types.Content(role="user", parts=[types.Part(text="  Autumn Rain  ")])
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    print(f"[{event.author}] {part.text}")


if __name__ == "__main__":
    asyncio.run(main())
