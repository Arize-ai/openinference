from phoenix.otel import register
from smolagents import OpenAIServerModel

register(auto_instrument=True)

model = OpenAIServerModel(model_id="gpt-4o")
output = model(messages=[{"role": "user", "content": "hello world"}])
print(output.content)
