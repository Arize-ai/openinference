# OpenInference Swarm Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-swarm.svg)](https://pypi.org/project/openinference-instrumentation-swarm/)

Python auto-instrumentation library for Swarm.

These traces are fully OpenTelemetry-compatible and can be sent to an OpenTelemetry collector for viewing, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix).


## Installation

This package is not published and needs to be installed from source.

```shell
pip install openinference-instrumentation-swarm
```

## Quickstart

TODO

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)