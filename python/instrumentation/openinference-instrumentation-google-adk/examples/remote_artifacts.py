import asyncio

import requests
from google.adk.agents import Agent
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext, load_artifacts
from google.genai import types
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation import TracerProvider
from openinference.instrumentation.google_adk import GoogleADKInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)

APP_NAME, USER_ID, SESSION_ID = "story_app", "12345", "123344"


async def load_remote_image(file_url: str, tool_context: ToolContext) -> str:
    """
    Reads a local image file and registers it as an ADK artifact.
    Args:
        tool_context:
        file_url: Remote location of file.
    """
    response = requests.get(file_url)
    image_part = types.Part.from_bytes(data=response.content, mime_type="application/pdf")
    filename = "image_from_url.pdf"
    await tool_context.save_artifact(filename=filename, artifact=image_part)
    return f"Success! Image '{filename}' is now available as an artifact."


poet_agent = Agent(
    name="poet_agent",
    model="gemini-2.0-flash",
    instruction=(
        "You are a creative poet and visual analyst. "
        "1. First, use 'load_remote_image' to get the file into the system. "
        "2. Then, use 'load_artifacts' to see the file content. "
        "3. Describe the file content in detail. "
    ),
    tools=[load_remote_image, load_artifacts],
)


async def run(file_path):
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    runner = Runner(
        agent=poet_agent,
        app_name=APP_NAME,
        session_service=session_service,
        artifact_service=artifact_service,
    )
    user_query = f"Please process the file at '{file_path}', describe it."
    content = types.Content(role="user", parts=[types.Part(text=user_query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)
    for event in events:
        if event:
            print(f"Agent:\n{event.content}")


async def main():
    await run("https://picsum.photos/200/300")
    # await run(
    #     "https://lorempdf.com/140/85/1"
    # )


if __name__ == "__main__":
    asyncio.run(main())
