"""
Trace an AG2 conversation and a tool call without an LLM API key.

The agent replies by running a tool instead of calling a model, so this runs offline and
is the quickest way to confirm spans are reaching Phoenix. Executing the tool from
inside the reply is what an LLM-driven tool call does too, so the trace has the same
shape: an AGENT span for the chat, a nested AGENT span for the reply, and a TOOL span
under that.

1. Run Phoenix locally: `pip install arize-phoenix && phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Run this example: `python no_llm_multi_agent.py`
4. View the traces at http://localhost:6006 under the `ag2-no-llm-multi-agent` project.
"""

import json
from typing import Any

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


def reply_with_weather(
    agent: ConversableAgent,
    messages: list[dict[str, Any]] | None = None,
    sender: Any = None,
    config: Any = None,
) -> tuple[bool, str]:
    _, result = agent.execute_function(
        {"name": "get_weather", "arguments": json.dumps({"city": "Portland"})},
        call_id="call-1",
    )
    return True, str(result["content"])


def main() -> None:
    weather_agent = ConversableAgent("weather_agent", llm_config=False, human_input_mode="NEVER")
    weather_agent.register_function({"get_weather": get_weather})
    weather_agent.register_reply([ConversableAgent, None], reply_with_weather, position=0)

    user_proxy = ConversableAgent("user_proxy", llm_config=False, human_input_mode="NEVER")

    chat = user_proxy.initiate_chat(
        weather_agent,
        message="What is the weather in Portland?",
        max_turns=1,
        silent=True,
    )
    print("weather_agent:", chat.chat_history[-1]["content"])
    print("\nView the traces at http://localhost:6006")


if __name__ == "__main__":
    main()
