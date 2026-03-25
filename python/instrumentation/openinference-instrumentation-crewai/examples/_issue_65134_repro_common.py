from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from crewai import LLM, Agent, Crew, Process, Task
from crewai.memory.storage.kickoff_task_outputs_storage import KickoffTaskOutputsSQLiteStorage
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.crewai import CrewAIInstrumentor

EXAMPLES_DIR = Path(__file__).resolve().parent
STORAGE_DIR = EXAMPLES_DIR / ".crewai_storage"
KICKOFF_OUTPUTS_DB = STORAGE_DIR / "latest_kickoff_task_outputs.db"
PROJECT_NAME = "crewai-issue-65134-repro"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "INSERT_OPENAI_API_KEY_HERE")
DEFAULT_TOPIC = os.getenv("CREWAI_REPRO_TOPIC", "AI")


def _is_placeholder(value: str) -> bool:
    return value.startswith("INSERT_") or value.startswith("YOUR_")


def require_value(name: str, value: str) -> str:
    if not value or _is_placeholder(value):
        raise RuntimeError(f"Set {name} in the script or environment before running this example.")
    return value


def configure_runtime() -> str:
    api_key = require_value("OPENAI_API_KEY", OPENAI_API_KEY)
    os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
    os.environ["OPENAI_API_KEY"] = api_key
    return api_key


def add_console_exporter(tracer_provider: Any) -> None:
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))


def configure_local_storage(crew: Crew) -> Crew:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    crew._task_output_handler.storage = KickoffTaskOutputsSQLiteStorage(
        db_path=str(KICKOFF_OUTPUTS_DB)
    )
    return crew


def build_repro_crew(*, openai_api_key: str) -> Crew:
    llm = LLM(model=OPENAI_MODEL, api_key=openai_api_key, temperature=0)

    writer = Agent(
        role="Tech Content Strategist",
        goal="Craft compelling content on tech advancements",
        backstory="You are a great at creating insightful articles.",
        verbose=True,
        allow_delegation=True,
        llm=llm,
    )

    task = Task(
        description="Develop a short blog post that highlights recent AI advancements.",
        expected_output="Short blog post of 3 sentences",
        agent=writer,
    )

    return configure_local_storage(
        Crew(
            name="Issue65134ReproCrew",
            agents=[writer],
            tasks=[task],
            verbose=False,
            process=Process.sequential,
        )
    )


def run_repro(*, tracer_provider: Any, topic: str = DEFAULT_TOPIC) -> Any:
    instrumentor = CrewAIInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        result = build_repro_crew(openai_api_key=configure_runtime()).kickoff(
            inputs={"topic": topic}
        )
        print("\nCrew kickoff completed.\n")
        print(result)
        tracer_provider.force_flush()
        return result
    finally:
        instrumentor.uninstrument()
        tracer_provider.shutdown()
