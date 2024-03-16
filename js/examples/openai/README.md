# Overview

This example shows how to use [@arizeai/openinference-instrumentation-openai](https://github.com/Arize-ai/openinference/tree/main/js/packages/openinference-instrumentation-openai) to instrument a simple Node.js application with OpenAI

Our example will export spans data simultaneously on `Console` and [arize-phoenix](https://github.com/Arize-ai/phoenix), however you can run your code anywhere and can use any exporter that OpenTelemetry supports.

## Installation

```shell
npm install
```

## Run

Make sure you have `OPENAI_API_KEY` set as an environment variable, then run:

```shell
npm run build
npm run start
```
