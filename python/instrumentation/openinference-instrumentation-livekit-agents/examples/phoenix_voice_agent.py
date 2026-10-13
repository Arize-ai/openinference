"""Send livekit-agents voice turns, with their audio, to Phoenix.

Start Phoenix (`phoenix serve`, or set PHOENIX_COLLECTOR_ENDPOINT for a remote one), set
OPENAI_API_KEY and LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, then run
`python phoenix_voice_agent.py dev` and talk to the agent from the LiveKit Agents
Playground, or run `python talk.py` in a second terminal.
"""

import os

from livekit.agents import AgentServer, JobProcess, cli
from phoenix.otel import register
from voice_agent import entrypoint, trace_with


def setup(proc: JobProcess) -> None:
    provider = register(
        project_name=os.environ.get("PHOENIX_PROJECT_NAME", "livekit-voice-agent"),
        set_global_tracer_provider=False,
        batch=True,
    )
    trace_with(proc, provider)


server = AgentServer(setup_fnc=setup)
server.rtc_session()(entrypoint)

if __name__ == "__main__":
    cli.run_app(server)
