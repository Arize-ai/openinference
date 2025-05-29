# Import open-telemetry dependencies
from arize.otel import register
from crewai import Agent, Crew, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv

from openinference.instrumentation.crewai import CrewAIInstrumentor

# Setup OTEL via our convenience function
# For Dev
tracer_provider = register(
    space_id="U3BhY2U6NDE3Oll3UzQ=",  # in app space settings page
    api_key="0101fb144245b35b97c",  # in app space settings page
    project_name="crew-ai-test",  # name this to whatever you would like
    endpoint="https://devotlp.arize.com/v1",
)


CrewAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Load environment variables
load_dotenv()

# Initialize tool
search_tool = SerperDevTool()

# Define Agent
researcher = Agent(
    role="AI Research Analyst",
    goal="Research and analyze AI trends and developments",
    backstory="Expert in researching and analyzing AI trends, with a focus on providing clear insights.",
    tools=[search_tool],
    verbose=True,
)

# Define Tasks
task1 = Task(
    description="Research and summarize the latest developments in Large Language Models (LLMs) from the past month.",
    expected_output="A detailed summary of 3 key LLM developments with their potential impact.",
    agent=researcher,
)

task2 = Task(
    description="Research and analyze recent breakthroughs in AI image generation technology.",
    expected_output="A comprehensive analysis of 3 major breakthroughs in AI image generation with their implications.",
    agent=researcher,
)

# Create Crew
crew = Crew(agents=[researcher], tasks=[task1, task2], verbose=True)

# Run it
if __name__ == "__main__":
    result = crew.kickoff()
    print("\nFinal Result:\n", result)
