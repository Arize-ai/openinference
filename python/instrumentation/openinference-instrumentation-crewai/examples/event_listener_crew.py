"""
CrewAI event-listener instrumentation example.

Demonstrates ``CrewAIInstrumentor().instrument(..., use_event_listener=True)``,
which hooks into CrewAI's event bus instead of wrapper-based instrumentation.
This mode is intended for event-bus-native CrewAI execution such as AMP /
low-code usage. Standard Python CrewAI applications should continue to use the
default wrapper-based instrumentation path.
"""

import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.tools import BaseTool  # type: ignore[import-untyped, unused-ignore]
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic import BaseModel, Field

from openinference.instrumentation.crewai import CrewAIInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"


class SearchToolSchema(BaseModel):
    query: str = Field(..., description="The search query to look up")


class SearchTool(BaseTool):  # type: ignore[misc, unused-ignore]
    name: str = "search_tool"
    description: str = "Search for information on a given topic."
    args_schema: type[BaseModel] = SearchToolSchema

    def _run(self, query: str, **kwargs: Any) -> str:
        return (
            f"Search results for '{query}':\n"
            "1. Significant progress in the field this year\n"
            "2. Industry leaders cite it as a top priority\n"
        )


def build_crew() -> Crew:
    search_tool = SearchTool()

    researcher = Agent(
        role="Research Analyst",
        goal="Find the latest trends in AI",
        backstory="You are a senior research analyst at a tech think tank.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
    )

    writer = Agent(
        role="Content Writer",
        goal="Write a short summary of the research findings",
        backstory="You turn complex research into clear, concise prose.",
        verbose=True,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            "Use the search_tool to find the latest AI trends. "
            "Summarize key findings in bullet points."
        ),
        expected_output="Bullet-point summary of AI trends",
        agent=researcher,
    )

    writing_task = Task(
        description="Write a two-paragraph summary based on the research findings.",
        expected_output="Two-paragraph summary",
        agent=writer,
        context=[research_task],
    )

    return Crew(
        name="EventListenerDemoCrew",
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        verbose=True,
        process=Process.sequential,
    )


def main() -> None:
    crew = build_crew()
    result = crew.kickoff()
    print(result)


if __name__ == "__main__":
    CrewAIInstrumentor().instrument(
        tracer_provider=tracer_provider,
        use_event_listener=True,
    )
    main()
