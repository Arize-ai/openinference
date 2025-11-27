"""
This example shows how to instrument your agno agent with OpenInference and send traces to Langfuse.

1. Install dependencies: pip install openai langfuse opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-agno
2. Either self-host or sign up for an account at https://us.cloud.langfuse.com
3. Set your Langfuse API key as an environment variables:
  - export LANGFUSE_PUBLIC_KEY=<your-key>
  - export LANGFUSE_SECRET_KEY=<your-key>
"""

import base64
import os

from agno.agent import Agent
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.workflow import Step, Workflow
from agno.workflow.parallel import Parallel
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
