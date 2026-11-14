# OpenInference LangChain Instrumentation

Python auto-instrumentation library for LangChain.

These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).

[![pypi](https://badge.fury.io/py/openinference-instrumentation-langchain.svg)](https://pypi.org/project/openinference-instrumentation-langchain/)

## Compatibility

This instrumentation works with:
- **LangChain 1.x** (`langchain>=1.0.0`): Modern agent framework built on LangGraph
- **LangChain Classic** (`langchain-classic>=1.0.0`): Legacy chains and tools (formerly `langchain 0.x`)
- All LangChain partner packages (`langchain-openai`, `langchain-anthropic`, `langchain-google-vertexai`, etc.)

The instrumentation hooks into `langchain-core`, which is the shared foundation used by all LangChain packages.

## Installation

### For LangChain 1.x (Recommended for New Projects)

```shell
pip install openinference-instrumentation-langchain langchain langchain-openai
```

### For LangChain Classic (Legacy Applications)

```shell
pip install openinference-instrumentation-langchain langchain-classic langchain-openai
```

### For Both (Migration Scenarios)

```shell
pip install openinference-instrumentation-langchain langchain langchain-classic langchain-openai
```

## Quickstart

### Example with LangChain 1.x (New Agent Framework)

Install packages needed for this demonstration.

```shell
pip install openinference-instrumentation-langchain langchain langchain-openai arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start the Phoenix app in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address.

The Phoenix app does not send data over the internet. It only operates locally on your machine.

```shell
python -m phoenix.server.main serve
```

The following Python code sets up the `LangChainInstrumentor` to trace `langchain` and send the traces to Phoenix at the endpoint shown below.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

LangChainInstrumentor().instrument()
```

To demonstrate tracing, we'll create a simple agent. First, configure your OpenAI credentials.

```python
import os

os.environ["OPENAI_API_KEY"] = "<your openai key>"
```

Now we can create an agent and run it.

```python
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny!"

model = ChatOpenAI(model="gpt-4")
agent = create_agent(model, tools=[get_weather])
result = agent.invoke({"messages": [{"role": "user", "content": "What's the weather in Paris?"}]})
print(result)
```

### Example with LangChain Classic (Legacy Chains)

For legacy applications using LangChain Classic:

```python
from langchain_classic.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

# ... (same instrumentation setup as above)

prompt_template = "Tell me a {adjective} joke"
prompt = PromptTemplate(input_variables=["adjective"], template=prompt_template)
llm = LLMChain(llm=OpenAI(), prompt=prompt, metadata={"category": "jokes"})
completion = llm.predict(adjective="funny", metadata={"variant": "funny"})
print(completion)
```

Visit the Phoenix app at `http://localhost:6006` to see the traces.

## Nesting Spans From Other Instrumentors

By default, the span of a LangChain run is not made current in the OpenTelemetry context, so a span created inside a run by another instrumentor, such as an `httpx` request made by a tool, doesn't nest under it. To opt in, pass `attach_run_span_to_context=True`:

```python
LangChainInstrumentor().instrument(attach_run_span_to_context=True)
```

This is off by default because LangChain reports the start and the end of a run through separate callbacks, with nothing to guarantee the second one arrives. Be aware that:

- Tracer callbacks run on the event loop during async runs, instead of in a thread pool.
- In an async call to a text-completion LLM (not a chat model), your other callback handlers don't receive `on_llm_start` unless they are inline too. When any handler is inline, LangChain sends `on_llm_start` for such a call only to the inline handlers. LangChain's own LangSmith tracer is inline, so enabling LangSmith tracing has the same effect.
- While a streaming run is suspended between chunks, its span is also current in the code consuming the stream.
- A run that LangChain never reports as ended keeps its span current in the context it started in, and that span is never exported, so spans started later in that context are orphaned. For example, LangChain doesn't report the end of a chat model or tool call cancelled by `asyncio.timeout`. Keep such a call in its own task so that the context ends with it: `await asyncio.wait_for(asyncio.create_task(model.ainvoke(messages)), timeout)`.
- A chat model or LLM call with several prompts starts all of their runs at once, so none of them is made current, and spans created during the call nest under the enclosing span.

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [More info on OpenInference and Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)