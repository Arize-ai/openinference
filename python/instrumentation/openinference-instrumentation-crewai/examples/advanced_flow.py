"""
CrewAI Flows - Advanced Example (Crew + Agents + Tasks)

Demonstrates CrewAI instrumentation with an advanced flow.
"""

import asyncio
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


class MarketResearchFlow(Flow[MarketResearchState]):
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
            role="Research Manager",
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
            output_json_schema=MarketAnalysis.schema(),  # structure validation
        )

        summary_task = Task(
            description=(
                "Summarize the research findings into a concise paragraph suitable "
                "for an executive briefing."
            ),
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

        # Extract results
        analysis_result = results.get(analysis_task.description)
        summary_result = results.get(summary_task.description)

        if analysis_result:
            try:
                structured = MarketAnalysis.parse_obj(analysis_result)
                print("✅ Structured Market Analysis captured.")
                self.state.analysis = structured
            except Exception:
                print("⚠️ Could not parse structured output. Using raw text instead.")
                self.state.analysis = None

        if summary_result:
            self.state.summary = summary_result

        return {"analysis": self.state.analysis, "summary": self.state.summary}

    @listen(analyze_market)
    def present_results(self, analysis: MarketAnalysis | None, summary: str | None) -> None:
        """Print final results from the research."""
        print("\n📊 Market Analysis Results")
        print("==========================")

        if analysis:
            print("\nKey Market Trends:")
            for trend in analysis.key_trends:
                print(f"- {trend}")

            print(f"\nMarket Size: {analysis.market_size}")

            print("\nMajor Competitors:")
            for competitor in analysis.competitors:
                print(f"- {competitor}")
        else:
            print("⚠️ No structured analysis data available.")

        if summary:
            print("\n🧭 Summary:")
            print(summary)
        else:
            print("⚠️ No summary provided.")


def create_advanced_flow(flow_name: Optional[str] = None) -> Flow:
    """
    Creates and returns an instance of an advanced flow.

    Args:
        flow_name: Optional name to set for the flow

    Returns:
        Configured CrewAI flow ready for execution
    """
    flow = MarketResearchFlow()
    if flow_name:
        flow.name = flow_name
    return flow


async def run_advanced_flow():
    """
    Executes the advanced flow and handles any runtime exceptions.
    """
    try:
        flow = create_advanced_flow("AdvancedFlowExample")
        await flow.kickoff_async(inputs={"product": "AI-powered chatbots"})
        print("✅ Flow execution completed successfully")
    except Exception as e:
        print(f"⚠️ Flow execution failed: {type(e).__name__}")


def main():
    """Run the CrewAI instrumentation demonstration."""
    print("CrewAI Instrumentation Demo - Advanced Flow")
    print("Check Phoenix UI at http://localhost:6006 for trace visualization\n")

    # Run the flow
    asyncio.run(run_advanced_flow())


if __name__ == "__main__":
    CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)
    main()
