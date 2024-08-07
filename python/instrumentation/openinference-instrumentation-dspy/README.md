# OpenInference DSPy Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-dspy.svg)](https://pypi.org/project/openinference-instrumentation-dspy/)

Python auto-instrumentation library for DSPy.

These traces are fully OpenTelemetry-compatible and can be sent to an OpenTelemetry collector for viewing, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix).


## Installation

```shell
pip install openinference-instrumentation-dspy
```

## Quickstart

This quickstart shows you how to instrument your DSPy application. It is adapted from the [DSPy quickstart](https://dspy-docs.vercel.app/docs/quick-start/minimal-example).

Install required packages.

```shell
pip install openinference-instrumentation-dspy dspy-ai arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start Phoenix in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address. (Phoenix does not send data over the internet. It only operates locally on your machine.)

```shell
python -m phoenix.server.main serve
```

Set up `DSPyInstrumentor` to trace your DSPy application and sends the traces to Phoenix at the endpoint defined below.

```python
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
trace_api.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

DSPyInstrumentor().instrument()
```

Import `dspy` and configure your language model.

```python
import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric

turbo = dspy.OpenAI(model='gpt-3.5-turbo-instruct', max_tokens=250)
dspy.settings.configure(lm=turbo)
gms8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gms8k.train[:10], gms8k.dev[:10]
```

Define a custom program that utilizes the `ChainOfThought` module to perform step-by-step reasoning to generate answers.

```python
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)
```

Optimize your program using the `BootstrapFewShotWithRandomSearch` teleprompter.

```python
from dspy.teleprompt import BootstrapFewShot

config = dict(max_bootstrapped_demos=4, max_labeled_demos=4)
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=gsm8k_trainset, valset=gsm8k_devset)
```

Evaluate performance on the dev dataset.

```python
from dspy.evaluate import Evaluate

evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=0)
evaluate(optimized_cot)
```

Visit the Phoenix app at `http://localhost:6006` to see your traces.

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)