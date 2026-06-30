# OpenInference MCP Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-mcp.svg)](https://pypi.org/project/openinference-instrumentation-mcp/)

Python auto-instrumentation library for MCP's python SDK. Currently, it only enables context propagation so that the span active
when making an MCP tool call can be connected to those generated when executing it. It does not generate any telemetry.

## Installation

```shell
pip install openinference-instrumentation-mcp
```

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
