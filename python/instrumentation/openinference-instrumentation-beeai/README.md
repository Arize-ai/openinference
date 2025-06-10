# OpenInference Instrumentation for BeeAI

This module provides **automatic instrumentation** for [BeeAI framework](https://github.com/i-am-bee/beeai-framework/tree/main/python). It integrates seamlessly with the [@opentelemetry/sdk-trace-node](https://github.com/open-telemetry/opentelemetry-js/tree/main/packages/opentelemetry-sdk-trace-node) to collect and export telemetry data.

## Installation

```shell
pip install openinference-instrumentation-beeai
```

## Quickstart

This quickstart shows you how to instrument your guardrailed LLM application

Install required packages.

```shell
pip install beeai-framework arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start Phoenix in the background as a collector. By default, it listens on http://localhost:6006. You can visit the app via a browser at the same address. (Phoenix does not send data over the internet. It only operates locally on your machine.)

```
python -m phoenix.server.main serve
```

Set up **BeeAIInstrumentor** to trace your crew and send the traces to Phoenix at the endpoint defined below. 
The `openinference_setup.py` file. 

```python
import logging

from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.beeai import BeeAIInstrumentor

logging.basicConfig(level=logging.DEBUG)


def setup_observability(endpoint: str = "http://localhost:6006/v1/traces") -> None:
    """
    Sets up OpenTelemetry with OTLP HTTP exporter and instruments the beeai framework.
    """
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace_api.set_tracer_provider(tracer_provider)

    BeeAIInstrumentor().instrument()

```

Set up a simple ReActAgent to get the current weather in Las Vegas. 

```python
import asyncio
import sys
import traceback

from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import TokenMemory
from beeai_framework.tools.search import DuckDuckGoSearchTool, WikipediaTool
from beeai_framework.tools.tool import AnyTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from openinference_setup import setup_observability

setup_observability()

llm = ChatModel.from_name(
    "ollama:granite3.1-dense:8b",
    ChatModelParameters(temperature=0),
)

tools: list[AnyTool] = [
    WikipediaTool(),
    OpenMeteoTool(),
    DuckDuckGoSearchTool(),
]

agent = ReActAgent(llm=llm, tools=tools, memory=TokenMemory(llm))

prompt = "What's the current weather in Las Vegas?"


async def main() -> None:
    response = await agent.run(
        prompt=prompt,
        execution=AgentExecutionConfig(
            max_retries_per_step=3, total_max_retries=10, max_iterations=20
        ),
    )

    print("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
