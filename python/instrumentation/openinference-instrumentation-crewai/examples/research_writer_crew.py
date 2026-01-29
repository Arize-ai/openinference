"""
CrewAI Multi-Agent Research Crew Example

Demonstrates CrewAI instrumentation with a comprehensive multi-agent workflow.
Features 5 specialized agents collaborating on research, analysis, and content creation.
"""

import os
from pathlib import Path
from typing import Any, Optional

from crewai import Agent, Crew, Process, Task
from crewai.memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
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

# Disable CrewAI's built-in telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Make sure to set the OPENAI_API_KEY environment variable
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Store in project directory
project_root = Path(__file__).parent
storage_dir = project_root / "crewai_storage"


class SearchToolSchema(BaseModel):
    query: str = Field(..., description="The search query to look up")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class SearchTool(BaseTool):  # type: ignore[misc, unused-ignore]
    """Custom tool to search for information."""

    name: str = "search_tool"
    description: str = "Search for information on a given topic. Returns relevant results."
    args_schema: type[BaseModel] = SearchToolSchema

    def _run(self, query: str, max_results: int = 5, **kwargs: Any) -> str:
        """Execute the search."""
        return f"""Search results for '{query}':

        1. Recent advancements in {query} show significant progress in 2024
        2. Industry leaders report {query} as a top priority
        3. Research papers on {query} increased by 45% this year
        4. Key applications of {query} in enterprise environments
        5. Future trends and predictions for {query}

        (Showing top {max_results} results)"""


class DataAnalysisToolSchema(BaseModel):
    data: str = Field(..., description="The data or text to analyze")
    analysis_type: str = Field(
        default="summary", description="Type of analysis: 'summary', 'trends', or 'insights'"
    )


class DataAnalysisTool(BaseTool):  # type: ignore[misc, unused-ignore]
    """Custom tool to analyze data and extract insights."""

    name: str = "data_analysis_tool"
    description: str = "Analyze data to extract trends, patterns, and insights."
    args_schema: type[BaseModel] = DataAnalysisToolSchema

    def _run(self, data: str, analysis_type: str = "summary", **kwargs: Any) -> str:
        """Perform data analysis."""
        word_count = len(data.split())
        return f"""Data Analysis ({analysis_type}):

        - Total words analyzed: {word_count}
        - Key themes identified: AI, Machine Learning, Innovation
        - Sentiment: Positive (85%)
        - Trend direction: Upward
        - Confidence score: High
        - Recommendation: Strong potential for further research"""


class ContentFormatterToolSchema(BaseModel):
    content: str = Field(..., description="The content to format")
    format_type: str = Field(
        default="blog", description="Output format: 'blog', 'report', or 'summary'"
    )


class ContentFormatterTool(BaseTool):  # type: ignore[misc, unused-ignore]
    """Custom tool to format content in different styles."""

    name: str = "content_formatter_tool"
    description: str = "Format content into different styles (blog post, report, summary)."
    args_schema: type[BaseModel] = ContentFormatterToolSchema

    def _run(self, content: str, format_type: str = "blog", **kwargs: Any) -> str:
        """Format the content."""
        if format_type == "blog":
            return f"""# Blog Post

{content}

---
*Published by AI Research Team*
*Tags: #AI #Technology #Innovation*
"""
        elif format_type == "report":
            return f"""RESEARCH REPORT
===============

{content}

CONCLUSIONS
-----------
The findings indicate significant progress in the field.

RECOMMENDATIONS
---------------
Further investigation is recommended.
"""
        else:
            return f"Summary: {content[:200]}..."


def create_research_writer_crew(crew_name: Optional[str] = None) -> Crew:
    """
    Create a multi-agent research crew with custom tools.

    Demonstrates multiple agents collaborating with tools in a realistic workflow.

    Args:
        crew_name: Optional name to set for the crew

    Returns:
        Configured CrewAI crew ready for execution
    """

    # Initialize custom tools
    search_tool = SearchTool()
    data_analysis_tool = DataAnalysisTool()
    content_formatter_tool = ContentFormatterTool()

    # Research Analyst - conducts initial research using search tool
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments in AI and data science",
        backstory="""You work at a leading tech think tank.
        Your expertise lies in identifying emerging trends.
        You have a knack for dissecting complex data and presenting actionable insights.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
    )

    # Data Scientist - analyzes research findings using data analysis tool
    data_scientist = Agent(
        role="AI Data Scientist",
        goal="Analyze technical aspects and provide data-driven insights",
        backstory="""You are an expert data scientist specializing in AI/ML technologies.
        You provide technical depth and statistical analysis to research projects.""",
        verbose=True,
        allow_delegation=False,
        tools=[data_analysis_tool],
    )

    # Content Writer - creates engaging content using content formatter tool
    writer = Agent(
        role="Tech Content Strategist",
        goal="Craft compelling content on tech advancements",
        backstory="""You are a renowned Content Strategist, known for your insightful articles.
        You transform complex concepts into compelling narratives.""",
        verbose=True,
        allow_delegation=True,
        tools=[content_formatter_tool],
    )

    # Task 1: Research latest developments using search tool
    research_task = Task(
        description="""Use the search_tool to conduct a comprehensive analysis of the latest
        advancements in AI in 2024. Search for 'AI advancements 2024' and identify key trends,
        breakthrough technologies, and potential industry impacts.""",
        expected_output="Full analysis report in bullet points based on search results",
        agent=researcher,
    )

    # Task 2: Analyze research data using data analysis tool
    analysis_task = Task(
        description="""Use the data_analysis_tool to analyze the research findings for technical
        insights and market implications. Perform a 'trends' analysis on the research data.
        Provide statistical analysis and data-driven conclusions.""",
        expected_output="Technical analysis report with insights from the analysis tool",
        agent=data_scientist,
        context=[research_task],
    )

    # Task 3: Create engaging content using content formatter tool
    writing_task = Task(
        description="""Using the insights provided, use the content_formatter tool to develop
        an engaging blog post that highlights the most significant AI advancements.
        Format the output as a 'blog' post. Your content should be informative yet accessible,
        catering to a tech-savvy audience.""",
        expected_output="Full blog post formatted using the content_formatter tool",
        agent=writer,
        context=[analysis_task],
    )

    # Create crew with agents and tasks
    crew = Crew(
        name=crew_name or "Demo Crew",
        agents=[researcher, data_scientist, writer],
        tasks=[research_task, analysis_task, writing_task],
        verbose=True,
        process=Process.sequential,
        memory=True,
        long_term_memory=LongTermMemory(
            storage=LTMSQLiteStorage(db_path=f"{storage_dir}/memory.db")
        ),
    )

    # Note: crew.key is auto-generated and read-only

    return crew


def run_collaborative_crew():
    """
    Run a comprehensive crew to demonstrate multi-agent collaboration with tools.
    """
    try:
        # Create comprehensive multi-agent crew with custom tools
        tech_research_crew = create_research_writer_crew("TechResearchCrew")
        result = tech_research_crew.kickoff()
        print("\n" + "=" * 80)
        print("✅ Multi-agent research crew completed")
        print("=" * 80)
        print(f"\nFinal Output:\n{result}")

    except Exception as e:
        print(f"⚠️  Crew execution failed: {type(e).__name__}: {e}")


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
        ai_crew.tasks[0].description = """Use the search_tool to research specifically
        AI and machine learning trends. Focus on LLMs, computer vision, and AI automation tools.
        Search for 'Large Language Models 2024' and 'Computer Vision advancements'."""
        ai_crew.kickoff()

        print("\n" + "=" * 80)
        print("✅ Multiple research crews completed")
        print("=" * 80)

    except Exception as e:
        print(f"⚠️  Crew execution failed: {type(e).__name__}: {e}")


def main():
    """Run the CrewAI instrumentation demonstration."""
    print("=" * 80)
    print("CrewAI Instrumentation Demo - Multi-Agent Research Workflow")
    print("=" * 80)
    print("Check Phoenix UI at http://localhost:6006 for trace visualization\n")

    # Run comprehensive multi-agent crew
    run_collaborative_crew()

    # Run multiple crews with different specializations
    run_multiple_crews()


if __name__ == "__main__":
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    main()
