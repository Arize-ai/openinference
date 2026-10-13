"""
Trace the AG2 quickstart: a single `ConversableAgent` driven by `run()`.

This mirrors the agent in the AG2 basic concepts guide
(https://docs.ag2.ai/latest/docs/user-guide/basic-concepts/conversable-agent/). `run()`
returns a response you iterate with `process()`, and produces an AGENT span for the run
with a nested AGENT span for each reply.

1. Run Phoenix locally: `pip install arize-phoenix && phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set your API key: `export OPENAI_API_KEY=<your-key>`
4. Run this example: `python conversable_agent_run.py`
5. View the traces at http://localhost:6006 under the `ag2-conversable-agent` project.
"""

import os

from autogen import ConversableAgent, LLMConfig
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.ag2 import AG2Instrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "ag2-conversable-agent"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
AG2Instrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

llm_config = LLMConfig(
    {
        "api_type": "openai",
        "model": "gpt-4o-mini",
        "api_key": os.environ["OPENAI_API_KEY"],
        "temperature": 0.2,
    }
)


def main() -> None:
    finance_agent = ConversableAgent(
        name="finance_agent",
        system_message=(
            "You are a financial assistant who helps analyze financial data and "
            "transactions. Keep answers to a few sentences."
        ),
        llm_config=llm_config,
    )

    response = finance_agent.run(
        message="Can you explain what makes a transaction suspicious?",
        max_turns=1,
        user_input=False,
    )
    response.process()

    print("\nView the traces at http://localhost:6006")


if __name__ == "__main__":
    main()
