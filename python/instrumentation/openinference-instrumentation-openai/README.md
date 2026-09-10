# OpenInference OpenAI Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-openai.svg)](https://pypi.org/project/openinference-instrumentation-openai/)

Python auto-instrumentation library for OpenAI's python SDK.

The traces emitted by this instrumentation are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).

## Installation

```shell
pip install openinference-instrumentation-openai
```

## SDK compatibility

The supported SDK range is `openai>=2.8.0`, matching the `instruments` extra
and the OpenTelemetry instrumentation dependency check. OpenAI 1.x and earlier
2.x releases are outside this range. Install the SDK with the instrumentor using:

```shell
pip install 'openinference-instrumentation-openai[instruments]'
```

CI tests OpenAI 2.8.0, OpenAI 3.0.0, and the latest available SDK on Python 3.10
and 3.14. There is no SDK upper bound; the latest lane is a required compatibility
check, including when a new major version becomes available.

OpenAI 3 uses `httpx2` as its default HTTP transport. `OpenAIInstrumentor` wraps
the SDK request methods and supports this transport without application changes.
Chat Completions, Completions, Embeddings, and Responses are tested with sync and
async clients, including streaming where the API supports it. OpenInference spans
are independent of HTTP transport spans: `opentelemetry-instrumentation-httpx`
does not instrument the SDK's default `httpx2` client in normal application use.

## Development and testing

From the repository root, run the three SDK lanes (formatting, lint, types, and tests):

```shell
uvx --with tox-uv tox -c python/tox.ini run -e py310-ci-openai,py310-ci-openai-v3,py310-ci-openai-latest
```

Use `py314` in place of `py310` to run the other CI Python version. The baseline
SDK is pinned in `test-requirements.txt`; the `v3` and `latest` overrides live in
`python/tox.ini` and follow the repository's dependency release-age policy.

The pytest-only `_httpx2_compat` plugin loads before HTTP mocking plugins. When
`httpx2` is installed, it aliases `httpx` and `httpcore` to their version 2 modules,
so RESPX and VCR intercept the SDK's native transport. This does not substitute a
legacy HTTP client into the SDK. The alias also lets the test suite exercise HTTP
child spans; application instrumentation does not load this plugin.

Tests block network access and replay existing VCR cassettes by default. Missing
mocks or cassette entries fail instead of contacting OpenAI or Azure. To deliberately
record a cassette, run the selected test with `--record-mode=once` and valid provider
credentials. This enables network access for VCR-marked tests. Request and response
headers are stripped; review recorded bodies before committing them.

## Quickstart

In this example we will instrument a small program that uses OpenAI and observe the traces via [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

Install packages.

```shell
pip install openinference-instrumentation-openai "openai>=2.8.0" arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start the phoenix server so that it is ready to collect traces.
The Phoenix server runs entirely on your machine and does not send data over the internet.

```shell
python -m phoenix.server.main serve
```

In a python file, setup the `OpenAIInstrumentor` and configure the tracer to send traces to Phoenix.

```python
import openai
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# Optionally, you can also print the spans to the console.
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


if __name__ == "__main__":
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Write a haiku."}],
        max_tokens=20,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in response:
        if chunk.choices and (content := chunk.choices[0].delta.content):
            print(content, end="")
```

Since we are using OpenAI, we must set the `OPENAI_API_KEY` environment variable to authenticate with the OpenAI API.

```shell
export OPENAI_API_KEY=your-api-key
```

Now simply run the python file and observe the traces in Phoenix.

```shell
python your_file.py
```

## FAQ
**Q: How to get token counts when streaming?**

**A:** To get token counts when streaming, install `openai>=2.8.0` and set `stream_options={"include_usage": True}` when calling `create`. See the example shown above. For more info, see [here](https://community.openai.com/t/usage-stats-now-available-when-using-streaming-with-the-chat-completions-api-or-completions-api/738156).

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [More info on OpenInference and Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
