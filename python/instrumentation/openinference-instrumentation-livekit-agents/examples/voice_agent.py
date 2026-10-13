"""A LiveKit voice agent with one tool, traced by the OpenInference instrumentor.

arize_voice_agent.py and phoenix_voice_agent.py build the tracer provider for their
backend and call trace_with() from the server's setup function, which runs once in each
job process. LiveKit pickles setup_fnc and the entrypoint into those processes, so both
must be module-level functions.
"""

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    function_tool,
)
from livekit.plugins import openai, silero
from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.livekit_agents import LiveKitAgentsInstrumentor


@function_tool
async def get_weather(context: RunContext[None], city: str) -> str:
    """Get the current weather for a city."""
    return f"It is 18 degrees and cloudy in {city}."


def trace_with(proc: JobProcess, provider: TracerProvider) -> None:
    LiveKitAgentsInstrumentor().instrument(tracer_provider=provider)
    proc.userdata["tracer_provider"] = provider
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    provider: TracerProvider = ctx.proc.userdata["tracer_provider"]

    async def flush() -> None:
        provider.force_flush()

    ctx.add_shutdown_callback(flush)
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
