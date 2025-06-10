from smolagents.tools import Tool


class GetWeatherTool(Tool):
    name = "get_weather"
    description = "Get the weather for a given city"
    inputs = {"location": {"type": "string", "description": "The city to get the weather for"}}
    output_type = "string"

    def forward(self, location: str) -> str:
        return "sunny"


get_weather_tool = GetWeatherTool()
assert get_weather_tool("Paris") == "sunny"
assert get_weather_tool(location="Paris") == "sunny"
