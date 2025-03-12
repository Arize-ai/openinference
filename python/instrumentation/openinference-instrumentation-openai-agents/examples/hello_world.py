from agents import Agent, Runner
from phoenix.otel import register

from openinference.instrumentation.openai import OpenAIInstrumentor

tracer_priver = register(project_name="openai-agents-hello-world")

OpenAIInstrumentor().instrument(tracer_provider=tracer_priver)

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
