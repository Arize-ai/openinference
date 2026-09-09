"""Trace the OpenAI Agents SDK's hosted file search and web search tools.

The agent gets a `FileSearchTool` over a throwaway vector store (created on the fly from
a small FAQ document) and a `WebSearchTool`. A first turn asks a question that is only
answerable from the FAQ; a second turn replays the first turn's items (including the
`file_search_call` and `web_search_call` items) as input and asks a follow-up that needs
the web. Both hosted tool calls appear on the LLM spans as `tool_call.*` attributes.

Prerequisites:
    pip install -r examples/requirements.txt
    export OPENAI_API_KEY=...
    phoenix serve                    # http://localhost:6006

Run:
    python examples/hosted_search_tools.py

Environment variables:
    SEARCH_MODEL       model to use (default: gpt-5.4)
    PHOENIX_PROJECT    Phoenix project name (default: hosted-search-tools)

The vector store and its file are deleted when the script exits.
"""

import io
import os
import time

from agents import Agent, FileSearchTool, Runner, WebSearchTool
from openai import OpenAI
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

FAQ = """\
Aurora Widget Co. internal FAQ

Q: What is the warranty period for the Aurora X200 widget?
A: The Aurora X200 ships with a 27-month limited warranty.

Q: Who is the support contact for enterprise customers?
A: Enterprise customers should email enterprise-support@example.com.
"""


def create_vector_store(client: OpenAI) -> str:
    store = client.vector_stores.create(name="openinference-hosted-search-example")
    client.vector_stores.files.upload_and_poll(
        vector_store_id=store.id,
        file=("aurora_faq.txt", io.BytesIO(FAQ.encode())),
    )
    # upload_and_poll returns once the file is processed, but give the index a moment.
    time.sleep(2)
    return store.id


def delete_vector_store(client: OpenAI, vector_store_id: str) -> None:
    for f in client.vector_stores.files.list(vector_store_id=vector_store_id):
        client.files.delete(f.id)
    client.vector_stores.delete(vector_store_id)


def main():
    provider = TracerProvider(
        resource=Resource.create(
            {"openinference.project.name": os.getenv("PHOENIX_PROJECT", "hosted-search-tools")}
        )
    )
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter("http://localhost:6006/v1/traces"))
    )
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)

    client = OpenAI()
    vector_store_id = create_vector_store(client)
    try:
        agent = Agent(
            name="Research assistant",
            model=os.getenv("SEARCH_MODEL", "gpt-5.4"),
            instructions=(
                "Use file search for questions about Aurora Widget Co. products. "
                "Use web search for anything about the outside world. Answer in one sentence."
            ),
            tools=[
                FileSearchTool(vector_store_ids=[vector_store_id], max_num_results=3),
                WebSearchTool(),
            ],
        )
        first = Runner.run_sync(
            agent, "What is the warranty period for the Aurora X200 widget?", max_turns=4
        )
        print(f"Turn 1: {first.final_output}")

        # Replay turn 1 (user message, file_search_call, assistant answer) as input, then
        # ask something that requires the web so the second turn issues a web_search_call.
        follow_up_input = first.to_input_list() + [
            {
                "role": "user",
                "content": "Now search the web: what is the current population of Iceland?",
            }
        ]
        second = Runner.run_sync(agent, follow_up_input, max_turns=4)
        print(f"Turn 2: {second.final_output}")
    finally:
        delete_vector_store(client, vector_store_id)
        provider.force_flush()
        provider.shutdown()


if __name__ == "__main__":
    main()
