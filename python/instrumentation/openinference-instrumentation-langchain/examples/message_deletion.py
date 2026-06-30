"""
Example showing how to delete messages from a LangGraph conversation using RemoveMessage.
This demonstrates both manual and programmatic message deletion.
"""

from typing import Literal, cast

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, RemoveMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.langchain import LangChainInstrumentor

# Set up OpenTelemetry
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

# Instrument LangChain
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# Initialize components
memory = MemorySaver()
model = ChatAnthropic(model_name="claude-3-haiku-20240307")


def should_continue(state: MessagesState) -> Literal["delete_messages"] | Literal["end"]:
    """Determine whether to delete messages or end."""
    return "delete_messages" if len(state["messages"]) > 3 else "end"


def call_model(state: MessagesState):
    """Call the model and return its response."""
    response = model.invoke(state["messages"])
    return {"messages": response}


def delete_messages(state: MessagesState):
    """Delete all but the last 3 messages."""
    print("Deleting messages...")
    messages = state["messages"]
    if len(messages) > 3:
        # Only create RemoveMessage for messages that have an ID
        return {
            "messages": [RemoveMessage(id=str(m.id)) for m in messages[:-3] if m.id is not None]
        }
    return {}


def create_workflow():
    """Create and return the workflow graph."""
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("delete_messages", delete_messages)

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "delete_messages": "delete_messages",
            "end": END,
        },
    )
    workflow.add_edge("delete_messages", END)

    return workflow.compile(checkpointer=memory)


def main():
    """Run the example."""
    app = create_workflow()
    config = cast(RunnableConfig, {"configurable": {"thread_id": "example-1"}})

    # First message
    print("\nSending first message...")
    input_message = HumanMessage(content="Hi! I'm Alice.")
    for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
        print("\nMessages after first interaction:")
        for msg in event["messages"]:
            print(f"{msg.type}: {msg.content}")

    # Second message
    print("\nSending second message...")
    input_message = HumanMessage(content="What's my name?")
    for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
        print("\nMessages after second interaction:")
        for msg in event["messages"]:
            print(f"{msg.type}: {msg.content}")

    # Third message (this should trigger deletion of older messages)
    print("\nSending third message...")
    input_message = HumanMessage(content="Am I Bob or Alice?")
    for event in app.stream({"messages": [input_message]}, config, stream_mode="values"):
        print("\nMessages after third interaction (with deletion):")
        for msg in event["messages"]:
            print(f"{msg.type}: {msg.content}")


if __name__ == "__main__":
    main()
