from typing import Optional

from haystack.components.agents import Agent
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.components.converters.html import HTMLToDocument
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.core.pipeline import Pipeline
from haystack.dataclasses import ChatMessage, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import tool
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.haystack import HaystackInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

HaystackInstrumentor().instrument(tracer_provider=tracer_provider)

document_store = InMemoryDocumentStore()  # create a document store or an SQL database


@tool
def add_database_tool(name: str, surname: str, job_title: Optional[str], other: Optional[str]):
    """Use this tool to add names to the database with information about them"""
    document_store.write_documents(
        [Document(content=name + " " + surname + " " + (job_title or ""), meta={"other": other})]
    )
    return


database_assistant = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=[add_database_tool],
    system_prompt="You are a database assistant. Extract all people’s names and relevant details"
    " from the given context (using only the provided text), add them to the "
    "knowledge base automatically, and return a brief summary of the added entries.",
    exit_conditions=["text"],
    max_agent_steps=100,
    raise_on_tool_invocation_failure=False,
)

builder = ChatPromptBuilder(
    template=[
        ChatMessage.from_user("""
    {% for doc in docs %}
    {{ doc.content|default|truncate(25000) }}
    {% endfor %}
    """)
    ],
    required_variables=["docs"],
)

extraction_agent = Pipeline()
extraction_agent.add_component("fetcher", LinkContentFetcher())
extraction_agent.add_component("converter", HTMLToDocument())
extraction_agent.add_component("builder", builder)

extraction_agent.add_component("database_agent", database_assistant)
extraction_agent.connect("fetcher.streams", "converter.sources")
extraction_agent.connect("converter.documents", "builder.docs")
extraction_agent.connect("builder", "database_agent")

agent_output = extraction_agent.run(
    {"fetcher": {"urls": ["https://en.wikipedia.org/wiki/Deepset"]}}
)

print(agent_output["database_agent"]["messages"][-1].text)
