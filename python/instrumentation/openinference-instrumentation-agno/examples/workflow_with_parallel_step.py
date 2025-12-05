"""
This example shows how to instrument your agno agent with OpenInference
and send traces to Arize Phoenix.

Install dependencies:
pip install openai opentelemetry-sdk opentelemetry-exporter-otlp
pip install openinference-instrumentation-agno
"""

import os

from phoenix.otel import register

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.workflow import Step, Workflow
from agno.workflow.parallel import Parallel


os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={os.getenv('ARIZE_PHOENIX_API_KEY')}"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = ""

# configure the Phoenix tracer
tracer_provider = register(
    project_name="default",  # Default is 'default'
    auto_instrument=True,  # Automatically use the installed OpenInference instrumentation
)


# Create agents
researcher = Agent(name="Researcher", tools=[HackerNewsTools(), DuckDuckGoTools()])
writer = Agent(name="Writer")
reviewer = Agent(name="Reviewer")

# Create individual steps
research_hn_step = Step(name="Research HackerNews", agent=researcher)
research_web_step = Step(name="Research Web", agent=researcher)
write_step = Step(name="Write Article", agent=writer)
review_step = Step(name="Review Article", agent=reviewer)

# Create workflow with direct execution
workflow = Workflow(
    name="Content Creation Pipeline",
    steps=[
        Parallel(research_hn_step, research_web_step, name="Research Phase"),
        write_step,
        review_step,
    ],
)

workflow.print_response(
    "Write about the latest AI developments",
    stream=True,
)
