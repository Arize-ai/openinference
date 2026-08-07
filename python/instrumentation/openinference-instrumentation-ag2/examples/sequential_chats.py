"""
Trace a sequence of AG2 chats that carry context forward.

`initiate_chats` runs a queue of chats in order, passing each chat's summary into the
next as carryover, which is the sequential orchestration pattern from the AG2 guide.
Each chat in the queue gets its own AGENT span, so the trace shows the whole pipeline.

1. Run Phoenix locally: `pip install arize-phoenix && phoenix serve`
2. Install dependencies: `pip install -r requirements.txt`
3. Set your API key: `export OPENAI_API_KEY=<your-key>`
4. Run this example: `python sequential_chats.py`
5. View the traces at http://localhost:6006 under the `ag2-sequential-chats` project.
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
    resource=Resource({ResourceAttributes.PROJECT_NAME: "ag2-sequential-chats"})
)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
AG2Instrumentor().instrument(tracer_provider=tracer_provider)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

llm_config = LLMConfig(
    {"api_type": "openai", "model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]}
)


def main() -> None:
    researcher = ConversableAgent(
        name="researcher",
        system_message="List the key facts about the topic in three short bullets.",
        llm_config=llm_config,
    )
    writer = ConversableAgent(
        name="writer",
        system_message="Turn the research you are given into a two-sentence summary.",
        llm_config=llm_config,
    )
    editor = ConversableAgent(
        name="editor",
        system_message="Tighten the summary you are given into a single sentence.",
        llm_config=llm_config,
    )
    coordinator = ConversableAgent(name="coordinator", human_input_mode="NEVER")

    # Each chat's summary is carried into the next chat in the queue.
    results = coordinator.initiate_chats(
        [
            {
                "recipient": researcher,
                "message": "Research the benefits of tracing LLM applications.",
                "max_turns": 1,
                "summary_method": "last_msg",
            },
            {
                "recipient": writer,
                "message": "Write the summary.",
                "max_turns": 1,
                "summary_method": "last_msg",
            },
            {
                "recipient": editor,
                "message": "Edit it down.",
                "max_turns": 1,
                "summary_method": "last_msg",
            },
        ]
    )

    print("\nfinal:", results[-1].summary)
    print("\nView the traces at http://localhost:6006")


if __name__ == "__main__":
    main()
