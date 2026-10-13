"""A LiveKit voice agent auto-instrumented with OpenInference, tracing to Phoenix.

Each reply becomes a conversation.turn span with the user's and the agent's audio, plus
user, assistant, and tool spans.

  phoenix serve   # or set PHOENIX_COLLECTOR_ENDPOINT for a remote Phoenix
  export OPENAI_API_KEY=... LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
  python phoenix_voice_agent.py dev

Then talk to it in the LiveKit Agents Playground, or run `python talk.py`.
"""

import os

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero
from phoenix.otel import register


@function_tool
async def get_weather(context: RunContext[None], city: str) -> str:
    """Get the current weather for a city."""
    return f"It is 18 degrees and cloudy in {city}."


def setup(proc: JobProcess) -> None:
    # Runs once in each job process, before its first session starts.
    register(
        project_name=os.environ.get("PHOENIX_PROJECT_NAME", "livekit-voice-agent"),
        auto_instrument=True,  # Automatically use the installed OpenInference instrumentation
        set_global_tracer_provider=False,  # Keep livekit-agents' own spans out of the project
        batch=True,
    )
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=setup)


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    session: AgentSession[None] = AgentSession(
        stt=openai.STT(model="gpt-4o-mini-transcribe"),
        llm=openai.LLM(model="gpt-4.1-mini"),
        tts=openai.TTS(model="gpt-4o-mini-tts"),
        vad=ctx.proc.userdata["vad"],
    )
    await session.start(
        Agent(
            instructions="You are a friendly voice assistant. Answer briefly.",
            tools=[get_weather],
        ),
        room=ctx.room,
    )


if __name__ == "__main__":
    cli.run_app(server)
