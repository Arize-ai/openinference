import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

AutogenAgentChatInstrumentor().instrument(tracer_provider=tracer_provider)


async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        model="gpt-3.5-turbo",
    )

    def get_weather(city: str) -> str:
        """Get the weather for a given city."""
        return f"The weather in {city} is 73 degrees and Sunny."

    # Define an AssistantAgent with the model, tool, system message, and reflection enabled
    agent = AssistantAgent(
        name="weather_agent",
        model_client=model_client,
        tools=[get_weather],
        system_message="You are a helpful assistant that can check the weather.",
        reflect_on_tool_use=True,
        model_client_stream=True,
    )

    result = await agent.run(task="What is the weather in New York?")
    await model_client.close()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
