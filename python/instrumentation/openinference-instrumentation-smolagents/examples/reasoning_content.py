from smolagents import LiteLLMModel
from smolagents.agents import CodeAgent

model_params = {"thinking": {"type": "enabled", "budget_tokens": 4000}}

model = LiteLLMModel(model_id="anthropic/claude-3-7-sonnet-20250219", **model_params)

agent = CodeAgent(model=model, add_base_tools=False)

print(agent.run("What's the weather like in Paris?"))
