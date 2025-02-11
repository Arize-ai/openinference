from smolagents import OpenAIServerModel

model = OpenAIServerModel(model_id="gpt-4o")
output = model(messages=[{"role": "user", "content": "hello world"}])
print(output.content)
