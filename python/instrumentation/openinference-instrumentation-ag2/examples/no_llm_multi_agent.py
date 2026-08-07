"""
Trace a two-agent AG2 conversation and a tool call without an LLM API key.

The agents reply with canned messages, so this runs offline and is the quickest way to
confirm spans are reaching Phoenix. It produces an AGENT span for the chat, a nested
AGENT span for the reply, and a TOOL span for the function call.

1. Run Phoenix locally: `pip install arize-phoenix && phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Run this example: `python no_llm_multi_agent.py`
4. View the traces at http://localhost:6006 under the `ag2-no-llm-multi-agent` project.
"""

import json

from autogen import ConversableAgent
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.ag2 import AG2Instrumentor
from openinference.semconv.resource import ResourceAttributes

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "ag2-no-llm-multi-agent"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
AG2Instrumentor().instrument(tracer_provider=tracer_provider)


def get_weather(city: str) -> str:
    return f"It is 72F and sunny in {city}."


def main() -> None:
    researcher = ConversableAgent(
        "researcher",
        llm_config=False,
        human_input_mode="NEVER",
        default_auto_reply="Portland is a great pick for a weekend trip.",
    )
    planner = ConversableAgent("planner", llm_config=False, human_input_mode="NEVER")
    planner.register_function({"get_weather": get_weather})

    chat = planner.initiate_chat(
        researcher,
        message="Which city should we visit this weekend?",
        max_turns=1,
        silent=True,
    )
    print("researcher:", chat.chat_history[-1]["content"])

    # With no LLM to request the call, invoke the tool directly to emit a TOOL span.
    _, result = planner.execute_function(
        {"name": "get_weather", "arguments": json.dumps({"city": "Portland"})},
        call_id="call-1",
    )
    print("get_weather:", result["content"])
    print("\nView the traces at http://localhost:6006")


if __name__ == "__main__":
    main()
