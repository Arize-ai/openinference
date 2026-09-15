from phoenix.otel import register
from pyagentspec.adapters.langgraph import AgentSpecLoader
from pyagentspec.agent import Agent
from pyagentspec.llms import OpenAiConfig
from pyagentspec.property import FloatProperty
from pyagentspec.tools import ServerTool

from openinference.instrumentation.agentspec import AgentSpecInstrumentor

tools = [
    ServerTool(
        name="sum",
        description="Sum two numbers",
        inputs=[FloatProperty(title="a"), FloatProperty(title="b")],
        outputs=[FloatProperty(title="result")],
    ),
    ServerTool(
        name="subtract",
        description="Subtract two numbers",
        inputs=[FloatProperty(title="a"), FloatProperty(title="b")],
        outputs=[FloatProperty(title="result")],
    ),
    ServerTool(
        name="multiply",
        description="Multiply two numbers",
        inputs=[FloatProperty(title="a"), FloatProperty(title="b")],
        outputs=[FloatProperty(title="result")],
    ),
    ServerTool(
        name="divide",
        description="Divide two numbers",
        inputs=[FloatProperty(title="a"), FloatProperty(title="b")],
        outputs=[FloatProperty(title="result")],
    ),
]

agent = Agent(
    name="calculator_agent",
    description="An agent that provides assistance with tool use.",
    llm_config=OpenAiConfig(name="openai-gpt-5-mini", model_id="gpt-5-mini"),
    system_prompt=(
        "You are a helpful calculator agent.\n"
        "Your duty is to compute the result of the given operation using tools, "
        "and to output the result.\n"
        "It's important that you reply with the result only.\n"
    ),
    tools=tools,
)

tool_registry = {
    "sum": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
}
langgraph_agent = AgentSpecLoader(tool_registry=tool_registry).load_component(agent)

tracer_provider = register(batch=True, project_name="calculator-app")

with AgentSpecInstrumentor().instrument_context(
    skip_dep_check=True, tracer_provider=tracer_provider
):
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
