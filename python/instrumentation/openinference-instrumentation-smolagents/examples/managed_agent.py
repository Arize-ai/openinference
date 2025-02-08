from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    OpenAIServerModel,
    ToolCallingAgent,
)

model = OpenAIServerModel(model_id="gpt-4o")
agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    max_steps=3,
    name="search",
    description=(
        "This is an agent that can do web search. "
        "When solving a task, ask him directly first, he gives good answers. "
        "Then you can double check."
    ),
)
manager_agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()],
    model=model,
    managed_agents=[agent],
)
manager_agent.run(
    "How many seconds would it take for a leopard at full speed to run through Pont des Arts? "
    "ASK YOUR MANAGED AGENT FOR LEOPARD SPEED FIRST"
)
