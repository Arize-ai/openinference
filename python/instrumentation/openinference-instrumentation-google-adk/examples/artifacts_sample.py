import asyncio
from pathlib import Path

from google.adk.agents import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext, load_artifacts
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

APP_NAME = "story_app"
USER_ID = "12345"
SESSION_ID = "123344"


async def load_local_image_artifact(file_path: str, tool_context: ToolContext) -> str:
    """
    Reads a local image file and registers it as an ADK artifact.
    Args:
        tool_context:
        file_path: The full local path to the image file.
    """
    path = Path(file_path)
    if not path.exists():
        return f"Error: File not found at {file_path}"

    image_bytes = path.read_bytes()

    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    filename = path.name
    await tool_context.save_artifact(filename=filename, artifact=image_part)
    return f"Success! Image '{filename}' is now available as an artifact."


agent = Agent(
    name="poet_agent",
    model="gemini-2.5-flash",
    instruction=(
        "You are a creative poet and visual analyst. "
        "1. First, use 'load_local_image_artifact' to get the file into the system. "
        "2. Then, use 'load_artifacts' to see the image content. "
        "3. Describe the image. "
    ),
    tools=[load_local_image_artifact, load_artifacts],
)


async def main():
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
        artifact_service=artifact_service,
    )
    local_path = "sample.png"
    user_query = f"Please process the image at '{local_path}', describe it."
    print(f"User: {user_query}\n")
    content = types.Content(role="user", parts=[types.Part(text=user_query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    for event in events:
        if event:
            print(f"Agent:\n{event.content}")


if __name__ == "__main__":
    asyncio.run(main())
