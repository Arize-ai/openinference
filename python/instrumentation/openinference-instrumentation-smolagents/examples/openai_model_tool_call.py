from smolagents import OpenAIServerModel
from smolagents.tools import Tool


class GetWeatherTool(Tool):
    name = "get_weather"
    description = "Get the weather for a given city"
    inputs = {"location": {"type": "string", "description": "The city to get the weather for"}}
    output_type = "string"

    def forward(self, location: str) -> str:
        return "sunny"


model = OpenAIServerModel(model_id="gpt-4o")
output = model(
    messages=[
        {
            "role": "user",
            "content": "What is the weather in Paris?",
        }
    ],
    tools_to_call_from=[GetWeatherTool()],
)
print(output.tool_calls[0].function)
