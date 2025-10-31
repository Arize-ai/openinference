from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.haystack import HaystackInstrumentor

# Configure HaystackInstrumentor with Phoenix endpoint
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

HaystackInstrumentor().instrument(tracer_provider=tracer_provider)


def search_documents(query: str, user_context: str) -> dict:
    """Search documents using query and user context."""
    return {"results": [f"Found results for '{query}' (user: {user_context})"]}


# Create tool that reads from state
search_tool = Tool(
    name="search",
    description="Search documents",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}, "user_context": {"type": "string"}},
        "required": ["query"],
    },
    function=search_documents,
    inputs_from_state={"user_name": "user_context"},
    # Maps state's "user_name" to the tool’s input parameter “user_context”
)

# Define agent with state schema including user_name
agent = Agent(
    chat_generator=OpenAIChatGenerator(),
    tools=[search_tool],
    state_schema={"user_name": {"type": str}, "search_results": {"type": list}},
)

# Initialize agent with user context
result = agent.run(
    messages=[ChatMessage.from_user("Search for Python tutorials")], user_name="Alice"
)
