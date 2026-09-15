from phoenix.otel import register
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

tracer_provider = register(batch=True, project_name="hello-world-app")
AgentSpecInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)

messages = []
while True:
    user_input = input("USER  >>> ")
    if user_input.lower() in ["exit", "quit"]:
        break
    messages.append({"role": "user", "content": user_input})
    response = langgraph_agent.invoke(
        input={"messages": messages},
        config={"configurable": {"thread_id": "1"}},
    )
    agent_answer = response["messages"][-1].content.strip()
    print("AGENT >>>", agent_answer)
    messages.append({"role": "assistant", "content": agent_answer})

AgentSpecInstrumentor().uninstrument()
