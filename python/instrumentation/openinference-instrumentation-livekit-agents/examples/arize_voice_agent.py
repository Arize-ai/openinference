"""Send livekit-agents voice turns, with their audio, to Arize AX.

Set ARIZE_SPACE_ID, ARIZE_API_KEY, OPENAI_API_KEY, and LIVEKIT_URL, LIVEKIT_API_KEY,
LIVEKIT_API_SECRET, then run `python arize_voice_agent.py dev` and talk to the agent from
the LiveKit Agents Playground, or run `python talk.py` in a second terminal.
"""

import os

from arize.otel import register
from livekit.agents import AgentServer, JobProcess, cli
from voice_agent import entrypoint, trace_with


def setup(proc: JobProcess) -> None:
    provider = register(
        space_id=os.environ["ARIZE_SPACE_ID"],
        api_key=os.environ["ARIZE_API_KEY"],
        project_name=os.environ.get("ARIZE_PROJECT_NAME", "livekit-voice-agent"),
        set_global_tracer_provider=False,
    )
    trace_with(proc, provider)


server = AgentServer(setup_fnc=setup)
server.rtc_session()(entrypoint)

if __name__ == "__main__":
    cli.run_app(server)
