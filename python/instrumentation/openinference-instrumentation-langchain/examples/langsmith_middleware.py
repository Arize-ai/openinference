"""Run a simple weather agent with traces in both LangSmith and Phoenix.

Set OPENAI_API_KEY and LANGSMITH_API_KEY environment variables before running.
Start Phoenix with `uvx arize-phoenix serve`, then run this example. Traces appear
in the `langchain-langsmith-middleware` project in both applications.

The model uses OpenAI, and the weather tool returns sample data. With LangSmith
tracing enabled, LLM spans stay under `model` and tool spans stay under `tools`
in Phoenix, even when model and tool middleware are installed.
"""

import os

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain_openai import ChatOpenAI
from langsmith import Client, tracing_context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.semconv.resource import ResourceAttributes


def get_weather(city: str) -> str:
    """Get sample weather for a city."""
    return f"The weather in {city} is sunny."


def main(project_name: str = "langchain-langsmith-middleware") -> None:
    for name in ("OPENAI_API_KEY", "LANGSMITH_API_KEY"):
        if not os.environ.get(name):
            raise SystemExit(f"Please set the {name} environment variable")
    os.environ["LANGSMITH_TRACING"] = "true"

    # Export OpenInference spans to local Phoenix.
    endpoint = "http://127.0.0.1:6006/v1/traces"
    tracer_provider = trace_sdk.TracerProvider(
        resource=Resource({ResourceAttributes.PROJECT_NAME: project_name})
    )
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    langsmith_client = Client()
    try:
        agent = create_agent(
            model=ChatOpenAI(model="gpt-4o-mini", temperature=0),
            tools=[get_weather],
            system_prompt="You are a helpful assistant. Use the weather tool to answer questions.",
            # These exercise wrap_model_call and wrap_tool_call, which LangSmith traces.
            middleware=[ModelRetryMiddleware(), ToolRetryMiddleware()],
        )
        query = "What's the weather in Paris?"
        print(f"Query: {query}\n")
        # LangSmith uses its own client while OpenInference exports the same agent to Phoenix.
        with tracing_context(project_name=project_name, client=langsmith_client):
            with tracer_provider.get_tracer(__name__).start_as_current_span("request"):
                result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        print(f"Response:\n{result['messages'][-1].content}")
        print(f"\nView project '{project_name}' in Phoenix (http://localhost:6006) and LangSmith.")
    finally:
        tracer_provider.shutdown()
        langsmith_client.flush()


if __name__ == "__main__":
    main()
