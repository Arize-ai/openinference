"""Trace asynchronous Team human-in-the-loop continuations in local Phoenix.

Prerequisites:
    phoenix serve
    export OPENAI_API_KEY=...
    python team_acontinue_run.py

The example sends traces to http://localhost:6006 and exercises both
``_acontinue_run`` and ``_acontinue_run_stream`` through Team's public API.
"""

import asyncio
from typing import AsyncIterator

from agno.models.openai import OpenAIChat
from agno.run.team import TeamRunOutput
from agno.team import Team
from agno.tools import tool
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.agno import AgnoInstrumentor

PHOENIX_ENDPOINT = "http://localhost:6006/v1/traces"
PROJECT_NAME = "agno-team-acontinue-run"


@tool(requires_confirmation=True)
def reserve_table(restaurant: str, party_size: int) -> str:
    """Reserve a restaurant table after the user confirms the booking."""
    return f"Reserved a table at {restaurant} for {party_size}."


def build_team() -> Team:
    return Team(
        name="Reservation Team",
        model=OpenAIChat(id="gpt-4o-mini"),
        members=[],
        tools=[reserve_table],
        instructions=[
            "When asked to reserve a table, call reserve_table exactly once.",
            "Do not ask follow-up questions when the restaurant and party size are provided.",
        ],
    )


def confirm_requirements(run_output: TeamRunOutput) -> None:
    requirements = run_output.requirements or []
    if not requirements:
        raise RuntimeError("Expected the team run to pause for confirmation.")
    for requirement in requirements:
        requirement.confirm()


async def final_stream_output(stream: AsyncIterator[object]) -> TeamRunOutput:
    final_output = None
    async for event in stream:
        if isinstance(event, TeamRunOutput):
            final_output = event
    if final_output is None:
        raise RuntimeError("The continuation stream did not yield its final TeamRunOutput.")
    return final_output


async def main() -> None:
    tracer_provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": PROJECT_NAME})
    )
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_ENDPOINT))
    )
    AgnoInstrumentor().instrument(tracer_provider=tracer_provider)

    team = build_team()

    paused = await team.arun(
        "Reserve a table at The Kitchen for 2 people.",
        session_id="team-acontinue-async",
    )
    confirm_requirements(paused)
    completed = await team.acontinue_run(paused)
    print(f"non-streaming continuation: {completed.content}")

    paused_stream = await team.arun(
        "Reserve a table at Tavernetta for 4 people.",
        session_id="team-acontinue-stream",
    )
    confirm_requirements(paused_stream)
    stream = team.acontinue_run(paused_stream, stream=True, yield_run_output=True)
    completed_stream = await final_stream_output(stream)
    print(f"streaming continuation: {completed_stream.content}")

    tracer_provider.force_flush()
    print(f"View project {PROJECT_NAME!r} at http://localhost:6006")


if __name__ == "__main__":
    asyncio.run(main())
