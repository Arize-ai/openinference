"""
This example shows how to instrument your agno agent with OpenInference and send traces to Langfuse.

1. Install dependencies: pip install openai langfuse opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-agno
2. Either self-host or sign up for an account at https://us.cloud.langfuse.com
3. Set your Langfuse API key as an environment variables:
  - export LANGFUSE_PUBLIC_KEY=<your-key>
  - export LANGFUSE_SECRET_KEY=<your-key>
"""

import asyncio
import base64
import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.yfinance import YFinanceTools
from openinference.instrumentation.agno import AgnoInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

LANGFUSE_AUTH = base64.b64encode(
    f"{os.getenv('LANGFUSE_PUBLIC_KEY')}:{os.getenv('LANGFUSE_SECRET_KEY')}".encode()
).decode()
# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
#     "https://us.cloud.langfuse.com/api/public/otel"  # 🇺🇸 US data region
# )
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
    "https://cloud.langfuse.com/api/public/otel"  # 🇪🇺 EU data region
)
# os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]="http://localhost:3000/api/public/otel" # 🏠 Local deployment (>= v3.22.0)

os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {LANGFUSE_AUTH}"


tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))

# Start instrumenting agno
AgnoInstrumentor().instrument(tracer_provider=tracer_provider)

from textwrap import dedent

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow

# Define agents
web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[DuckDuckGoTools()],
    role="Search the web for the latest news and trends",
)
hackernews_agent = Agent(
    name="Hackernews Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[HackerNewsTools()],
    role="Extract key insights and content from Hackernews posts",
)

writer_agent = Agent(
    name="Writer Agent",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions="Write a blog post on the topic",
)


def prepare_input_for_web_search(step_input: StepInput) -> StepOutput:
    topic = step_input.input
    return StepOutput(
        content=dedent(f"""\
	I'm writing a blog post on the topic
	<topic>
	{topic}
	</topic>

	Search the web for atleast 10 articles\
	""")
    )


def prepare_input_for_writer(step_input: StepInput) -> StepOutput:
    topic = step_input.input
    research_team_output = step_input.previous_step_content

    return StepOutput(
        content=dedent(f"""\
	I'm writing a blog post on the topic:
	<topic>
	{topic}
	</topic>

	Here is information from the web:
	<research_results>
	{research_team_output}
	<research_results>\
	""")
    )


# Define research team for complex analysis
research_team = Team(
    name="Research Team",
    members=[hackernews_agent, web_agent],
    instructions="Research tech topics from Hackernews and the web",
)


# Create and use workflow
if __name__ == "__main__":
    content_creation_workflow = Workflow(
        name="Blog Post Workflow",
        description="Automated blog post creation from Hackernews and the web",
        db=SqliteDb(
            session_table="workflow_session",
            db_file="tmp/workflow.db",
        ),
        steps=[
            prepare_input_for_web_search,
            research_team,
            prepare_input_for_writer,
            writer_agent,
        ],
    )

    content_creation_workflow.print_response(
        input="Compare machine learning algorithms for image classification?",
        stream=True,
        stream_events=True,
    )
