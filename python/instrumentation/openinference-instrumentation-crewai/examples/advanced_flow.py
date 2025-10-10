"""
CrewAI Flows - Advanced Example
===============================

This example demonstrates an advanced CrewAI flow that integrates:
- Crew with multiple Agents and Tasks
- Structured Pydantic outputs with multi-step flow
- Runs the flow and exports traces to Phoenix + Console

The goal is to perform market research for a given product, summarize findings,
and display structured and human-readable results.
"""

import os
from typing import Any, Dict, List, Optional

from crewai import Agent, Crew, Task
from crewai.flow.flow import Flow, listen, start
from crewai_tools import SerperDevTool
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)  # type: ignore[import-not-found]
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from pydantic import BaseModel, Field

from openinference.instrumentation.crewai import CrewAIInstrumentor

# OpenTelemetry Configuration
endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

# Disable CrewAI's built-in telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

# Make sure to set the OPENAI_API_KEY environment variable
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Make sure to set the SERPER_API_KEY environment variable
os.environ["SERPER_API_KEY"] = "YOUR_API_KEY"


class MarketAnalysis(BaseModel):
    """Structured schema for market research results."""

    key_trends: List[str] = Field(description="List of identified market trends")
    market_size: str = Field(description="Estimated market size")
    competitors: List[str] = Field(description="Major competitors in the market")


class MarketResearchState(BaseModel):
    """Shared state across flow steps."""

    product: str = ""
    analysis: MarketAnalysis | None = None
    summary: str | None = None


class AdvancedFlow(Flow[MarketResearchState]):
    """
    An advanced flow that orchestrates market research using a Crew
    with multiple Agents and Tasks.
    """

    @start()
    def initialize_research(self) -> Dict[str, Any]:
        """Initialize the market research process."""
        print(f"🟢 Starting market research for product: {self.state.product}")
        return {"product": self.state.product}

    @listen(initialize_research)
    async def analyze_market(self, product: str) -> Dict[str, Any]:
        """Perform market research using a Crew with Agents and Tasks."""

        # Define Agents
        analyst = Agent(
            role="Market Research Analyst",
            goal=f"Research and analyze the market for {product}",
            backstory="""You are an experienced analyst skilled.
            You help in identifying trends and competitors.""",
            tools=[SerperDevTool()],
            verbose=True,
        )

        manager = Agent(
            role="Market Research Manager",
            goal="Summarize research findings clearly and concisely.",
            backstory="You oversee market research projects and ensure well-presented insights.",
            verbose=True,
        )

        # Define Tasks
        analysis_task = Task(
            description=(
                f"Research the market for {product}. Include:\n"
                "1. Key market trends\n"
                "2. Market size\n"
                "3. Major competitors\n"
                "Respond using the provided structured schema."
            ),
            expected_output="Structured JSON matching the MarketAnalysis model.",
            agent=analyst,
            output_pydantic=MarketAnalysis,
        )

        summary_task = Task(
            description=(
                "Summarize the research findings into a concise paragraph suitable "
                "for an executive briefing."
            ),
            expected_output="A short, well-written summary paragraph of the market research.",
            agent=manager,
            depends_on=[analysis_task],
        )

        # Create Crew
        crew = Crew(
            name="Market Research Crew",
            agents=[analyst, manager],
            tasks=[analysis_task, summary_task],
            verbose=True,
        )

        print("🚀 Kicking off Market Research Crew execution...\n")
        results = await crew.kickoff_async()

        # CrewOutput contains TaskOutput objects
        analysis_result = None
        for task_output in results.tasks_output:
            if hasattr(task_output, "pydantic") and task_output.pydantic:
                analysis_result = task_output.pydantic

        if not analysis_result:
            print("⚠️ No structured output returned — Using fallback.")
            analysis_result = MarketAnalysis(
                key_trends=[
                    "Increased adoption of conversational AI in enterprises",
                    "Integration of AI chatbots with CRM platforms",
                    "Growing use of voice-based assistants"
                ],
                market_size="Estimated at $3.5B in 2025, growing 25% annually",
                competitors=["Drift", "Intercom", "Ada", "Kore.ai"]
            )

        return {"analysis": analysis_result}

    @listen(analyze_market)
    def present_results(self, analysis: Dict[str, Any], **kwargs):
        """Display market research results."""
        print("\n📊 Market Analysis Results")
        print("==========================")

        market_analysis = analysis.get("analysis") if isinstance(analysis, dict) else analysis

        if isinstance(market_analysis, MarketAnalysis):
            print("\nKey Market Trends:")
            for trend in market_analysis.key_trends:
                print(f"- {trend}")

            print(f"\nMarket Size: {market_analysis.market_size}")

            print("\nMajor Competitors:")
            for competitor in market_analysis.competitors:
                print(f"- {competitor}")
        else:
            print("❌ No structured analysis available.")
            print("Raw Analysis:", analysis)


def create_advanced_flow(flow_name: Optional[str] = None) -> Flow:
    """
    Creates and returns an instance of an advanced flow.

    Args:
        flow_name: Optional name to set for the flow

    Returns:
        Configured CrewAI flow ready for execution
    """
    flow = AdvancedFlow()
    if flow_name:
        flow.name = flow_name
    return flow


def run_advanced_flow():
    """
    Executes the advanced flow and handles any runtime exceptions.
    """
    try:
        flow = create_advanced_flow("Advanced Flow Example")
        flow.kickoff(inputs={"product": "AI-powered Chatbots"})
        print("✅ Flow execution completed successfully.")
    except Exception as e:
        print(f"⚠️ Flow execution failed: {type(e).__name__}")


def main():
    """Run the CrewAI instrumentation demonstration."""
    print("CrewAI Instrumentation Demo - Advanced Flow")
    print("Check Phoenix UI at http://localhost:6006 for trace visualization\n")
    run_advanced_flow()


if __name__ == "__main__":
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    main()
