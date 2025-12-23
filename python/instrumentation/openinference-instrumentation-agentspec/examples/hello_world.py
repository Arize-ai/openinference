from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pyagentspec.adapters.langgraph import AgentSpecLoader
from pyagentspec.agent import Agent
from pyagentspec.llms import OpenAiConfig

from openinference.instrumentation.agentspec import AgentSpecInstrumentor

agent = Agent(
    name="assistant",
    description="An general purpose agent without tools",
    llm_config=OpenAiConfig(name="openai-gpt-5-mini", model_id="gpt-5-mini"),
    system_prompt="You are a helpful assistant. Help the user answering politely.",
)

langgraph_agent = AgentSpecLoader().load_component(agent)

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AgentSpecInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)

while True:
    user_input = input("USER  >>> ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = langgraph_agent.invoke(
        input={"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "1"}},
    )
    print("AGENT >>>", response["messages"][-1].content.strip())

AgentSpecInstrumentor().uninstrument()
