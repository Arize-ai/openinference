# OpenInference Autogen-Agentchat Instrumentation

[![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-autogen-agentchat.svg)](https://pypi.python.org/pypi/openinference-instrumentation-autogen-agentchat)

OpenTelelemetry instrumentation for Autogen AgentChat, enabling tracing of agent interactions and conversations.

The traces emitted by this instrumentation are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix)

## Installation

```shell
pip install openinference-instrumentation-autogen-agentchat
```

## Quickstart

In this example we will instrument a simple Autogen AgentChat application and observe the traces via [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

Install required packages.

```shell
pip install openinference-instrumentation-autogen-agentchat autogen-agentchat arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start the phoenix server so that it is ready to collect traces.
The Phoenix server runs entirely on your machine and does not send data over the internet.

```shell
phoenix serve
```

Here's a simple example using a single assistant agent:

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor

# Set up the tracer provider
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

# Instrument AutogenAgentChat
AutogenAgentChatInstrumentor().instrument(tracer_provider=tracer_provider)

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-3.5-turbo",
    )

    def get_weather(city: str) -> str:
        """Get the weather for a given city."""
        return f"The weather in {city} is 73 degrees and Sunny."

    # Create an assistant agent with tools
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
```

For a more complex example using multiple agents in a team:

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.autogen_agentchat import AutogenAgentChatInstrumentor

# Set up the tracer provider
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

# Instrument AutogenAgentChat
AutogenAgentChatInstrumentor().instrument(tracer_provider=tracer_provider)

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4",
    )

    # Create two agents: a primary and a critic
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message="""
        Provide constructive feedback.
        Respond with 'APPROVE' when your feedbacks are addressed.
        """,
    )

    # Termination condition: stop when the critic says "APPROVE"
    text_termination = TextMentionTermination("APPROVE")

    # Create a team with both agents
    team = RoundRobinGroupChat(
        [primary_agent, critic_agent],
        termination_condition=text_termination
    )

    # Run the team on a task
    result = await team.run(task="Write a short poem about the fall season.")
    await model_client.close()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

Since we are using OpenAI, we must set the `OPENAI_API_KEY` environment variable to authenticate with the OpenAI API.

```shell
export OPENAI_API_KEY=[your_key_here]
```

Now simply run the python file and observe the traces in Phoenix.

```shell
python your_file.py
```

## More Info

- [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
- [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
- [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
