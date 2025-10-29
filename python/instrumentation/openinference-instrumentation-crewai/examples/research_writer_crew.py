"""
CrewAI Multi-Agent Research Crew Example

Demonstrates CrewAI instrumentation with a comprehensive multi-agent workflow.
Features 5 specialized agents collaborating on research, analysis, and content creation.
"""

import os
from pathlib import Path
from typing import Optional

from crewai import Agent, Crew, Process, Task
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.crewai import CrewAIInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Disable CrewAI's built-in telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Make sure to set the OPENAI_API_KEY environment variable
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Store in project directory
project_root = Path(__file__).parent
storage_dir = project_root / "crewai_storage"


def create_research_writer_crew(crew_name: Optional[str] = None) -> Crew:
    """
    Create a multi-agent research crew.

    Demonstrates multiple agents collaborating in a realistic workflow.

    Args:
        crew_name: Optional name to set for the crew

    Returns:
        Configured CrewAI crew ready for execution
    """

    # Research Analyst - conducts initial research
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments in AI and data science",
        backstory="""You work at a leading tech think tank.
        Your expertise lies in identifying emerging trends.
        You have a knack for dissecting complex data and presenting actionable insights.""",
        verbose=False,
        allow_delegation=False,
    )

    # Data Scientist - analyzes research findings
    data_scientist = Agent(
        role="AI Data Scientist",
        goal="Analyze technical aspects and provide data-driven insights",
        backstory="""You are an expert data scientist specializing in AI/ML technologies.
        You provide technical depth and statistical analysis to research projects.""",
        verbose=False,
        allow_delegation=False,
    )

    # Content Writer - creates engaging content
    writer = Agent(
        role="Tech Content Strategist",
        goal="Craft compelling content on tech advancements",
        backstory="""You are a renowned Content Strategist, known for your insightful articles.
        You transform complex concepts into compelling narratives.""",
        verbose=False,
        allow_delegation=True,
    )

    # Task 1: Research latest developments
    research_task = Task(
        description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
        Identify key trends, breakthrough technologies, and potential industry impacts.""",
        expected_output="Full analysis report in bullet points",
        agent=researcher,
    )

    # Task 2: Analyze research data
    analysis_task = Task(
        description="""Analyze the research findings for technical insights and market implications.
        Provide statistical analysis and data-driven conclusions.""",
        expected_output="Technical analysis report with insights",
        agent=data_scientist,
        context=[research_task],
    )

    # Task 3: Create engaging content
    writing_task = Task(
        description="""Using the insights provided, develop an engaging blog post
        that highlights the most significant AI advancements.
        Your post should be informative yet accessible, catering to a tech-savvy audience.""",
        expected_output="Full blog post of at least 4 paragraphs",
        agent=writer,
        context=[analysis_task],
    )

    # Create crew with agents and tasks
    crew = Crew(
        name="Demo Crew",
        agents=[researcher, data_scientist, writer],
        tasks=[research_task, analysis_task, writing_task],
        verbose=False,
        process=Process.sequential,
        memory=True,
        long_term_memory=LongTermMemory(
            storage=LTMSQLiteStorage(
                db_path=f"{storage_dir}/memory.db"
            )
        ),
    )

    # Note: crew.key is auto-generated and read-only

    return crew


def run_collaborative_crew():
    """
    Run a comprehensive crew to demonstrate multi-agent collaboration.
    """
    try:
        # Create comprehensive multi-agent crew
        tech_research_crew = create_research_writer_crew("TechResearchCrew")
        tech_research_crew.kickoff()
        print("✅ Multi-agent research crew completed")

    except Exception as e:
        print(f"⚠️  Crew execution failed: {type(e).__name__}")


def run_multiple_crews():
    """
    Run multiple specialized crews to demonstrate different research focuses.
    """
    try:
        # Crew 1: General Tech Research
        tech_crew = create_research_writer_crew("TechResearchCrew")
        tech_crew.kickoff()

        # Crew 2: AI Specialized Focus
        ai_crew = create_research_writer_crew("AISpecializedCrew")
        # Modify tasks for AI focus
        ai_crew.tasks[0].description = """Research specifically AI and machine learning trends.
        Focus on LLMs, computer vision, and AI automation tools."""
        ai_crew.kickoff()

        print("✅ Multiple research crews completed")

    except Exception as e:
        print(f"⚠️  Crew execution failed: {type(e).__name__}")


def main():
    """Run the CrewAI instrumentation demonstration."""
    print("CrewAI Instrumentation Demo - Multi-Agent Research Workflow")
    print("Check Phoenix UI at http://localhost:6006 for trace visualization\n")

    # Run comprehensive multi-agent crew
    run_collaborative_crew()

    # Run multiple crews with different specializations
    run_multiple_crews()


if __name__ == "__main__":
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    main()
