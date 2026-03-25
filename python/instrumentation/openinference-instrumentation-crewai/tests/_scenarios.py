import os
import uuid
from typing import Any

from crewai import LLM, Agent, Crew, Task
from crewai.crews import CrewOutput
from crewai.flow.flow import Flow, listen, start  # type: ignore[import-untyped, unused-ignore]
from crewai.tools import BaseTool  # type: ignore[import-untyped, unused-ignore]
from pydantic import BaseModel, Field


class MockScrapeWebsiteToolSchema(BaseModel):
    url: str = Field(..., description="The website URL to scrape")


class MockScrapeWebsiteTool(BaseTool):  # type: ignore[misc, unused-ignore]
    """Mock tool to replace ScrapeWebsiteTool and avoid chromadb dependency."""

    name: str = "scrape_website"
    description: str = "Scrape text content from a website URL"
    args_schema: type[BaseModel] = MockScrapeWebsiteToolSchema

    def _run(self, url: str = "http://quotes.toscrape.com/", **kwargs: Any) -> str:
        return (
            '"The world as we have created it is a process of our thinking. '
            'It cannot be changed without changing our thinking." by Albert Einstein'
        )


def _get_test_llm() -> LLM:
    openai_api_key = os.getenv("OPENAI_API_KEY", "sk-test")
    return LLM(model="gpt-4.1-nano", api_key=openai_api_key, temperature=0)


def kickoff_crew() -> CrewOutput:
    """Run a representative Crew with tool and LLM activity."""
    url = "http://quotes.toscrape.com/"
    llm = _get_test_llm()

    scraper_agent = Agent(
        role="Website Scraper",
        goal="Scrape content from URL",
        backstory="You extract text from websites",
        tools=[MockScrapeWebsiteTool()],
        allow_delegation=False,
        llm=llm,
        max_iter=2,
        max_retry_limit=0,
        max_execution_time=120,
        verbose=True,
    )
    analyzer_agent = Agent(
        role="Content Analyzer",
        goal="Extract quotes from text",
        backstory="You extract quotes from text",
        allow_delegation=False,
        llm=llm,
        tools=[],
        max_iter=2,
        max_retry_limit=0,
        verbose=True,
    )

    scrape_task = Task(
        description=f"Call the scrape_website tool to fetch text from {url} and return the result.",
        expected_output="Text content from the website.",
        agent=scraper_agent,
        name="scrape-task",
    )
    analyze_task = Task(
        description="Extract the first quote from the content.",
        expected_output="Quote with author.",
        agent=analyzer_agent,
        context=[scrape_task],
        name="analyze-task",
    )

    crew = Crew(
        agents=[scraper_agent, analyzer_agent],
        tasks=[scrape_task, analyze_task],
    )

    crew_output = crew.kickoff(inputs={"id": str(uuid.uuid4())})
    assert isinstance(crew_output, CrewOutput)
    assert crew_output.raw
    return crew_output


def kickoff_flow() -> str:
    """Run a minimal Flow with two CHAIN nodes."""

    class SimpleFlow(Flow[Any]):  # type: ignore[misc, unused-ignore]
        @start()  # type: ignore[misc, unused-ignore]
        def step_one(self) -> str:
            return "Step One Output"

        @listen(step_one)  # type: ignore[misc, unused-ignore]
        def step_two(self, step_one_output: str) -> str:
            return f"Step Two Received: {step_one_output}"

    result = SimpleFlow().kickoff(inputs={"id": str(uuid.uuid4())})
    assert isinstance(result, str)
    assert result == "Step Two Received: Step One Output"
    return result


def kickoff_flow_with_crew() -> CrewOutput:
    """Run a Flow step that executes a Crew."""
    llm = _get_test_llm()

    class CrewFlow(Flow[Any]):  # type: ignore[misc, unused-ignore]
        @start()  # type: ignore[misc, unused-ignore]
        def run_crew(self) -> CrewOutput:
            scraper = Agent(
                role="Website Scraper",
                goal="Scrape content from URL",
                backstory="You extract text from websites",
                tools=[MockScrapeWebsiteTool()],
                allow_delegation=False,
                llm=llm,
                max_iter=2,
                max_retry_limit=0,
            )
            task = Task(
                description=(
                    "Call the scrape_website tool to fetch text from "
                    "http://quotes.toscrape.com/ and return the result."
                ),
                expected_output="Text content from the website.",
                agent=scraper,
            )
            crew = Crew(agents=[scraper], tasks=[task])
            crew_output = crew.kickoff()
            assert isinstance(crew_output, CrewOutput)
            return crew_output

    result = CrewFlow().kickoff()
    assert isinstance(result, CrewOutput)
    return result


def kickoff_agent(prompt: str = "What is 2+2?") -> Any:
    """Run a standalone Agent outside a Crew."""
    agent = Agent(
        role="Helpful Assistant",
        goal="Answer questions clearly and concisely",
        backstory="You are a helpful assistant.",
        allow_delegation=False,
        llm=_get_test_llm(),
    )
    result = agent.kickoff(prompt)
    assert result is not None
    return result
