# OpenTelemetry Instrumentation in Bee-Agent-Framework

This document provides an overview of the OpenTelemetry instrumentation setup in the BeeAI.
The implementation is designed to [create telemetry spans](https://opentelemetry.io/docs/languages/js/instrumentation/#create-spans) for observability.

## Overview

This is a simple example of setting up BeeAI auto instrumentation in a node application.

## Instrumentation

Checkout the [instrumentation.ts](./instrumentation.ts) file to see how to auto-instrument Beeai and export spans to a locally running [Phoenix](https://github.com/Arize-ai/phoenix) server.

### Environment Variable

You can set what keys will be disabled in the open telemetry spans.

```bash
# Ignore sensitive keys from collected events data
export INSTRUMENTATION_IGNORED_KEYS="apiToken,accessToken"
```

## Instructions

> Please use node version >= 20 to run this example.

1. (Optional) In order to see spans in [Phoenix](https://github.com/Arize-ai/phoenix), begin running a Phoenix server. This can be done in one command using docker.

```
docker run -p 6006:6006 -i -t arizephoenix/phoenix
```

or via the command line:

```
brew install i-am-bee/beeai/arize-phoenix
brew services start arize-phoenix
```

see https://docs.beeai.dev/observability/agents-traceability for more detail.

2. To run this example, be sure that you have installed ollama with the llama3.1 model downloaded. Beeai documentation is available at [BeeAI Framework for TypeScript](https://github.com/i-am-bee/beeai-framework/tree/main/typescript)
3. Go to the root of the JS directory of this repo and run the following commands to install dependencies and build the latest packages in the repo. This ensures any dependencies within the repo are at their latest versions.

```shell
nvm use 20
```

```shell
pnpm -r install
```

```shell
pnpm -r prebuild
```

```shell
pnpm -r build
```

4. Run the following command from the `openinference-instrumentation-beeai` package directory

| Module            | command                                       |
| ----------------- | --------------------------------------------- |
| ReAct Agent       | `node dist/examples/run-react-agent.js`       |
| ToolCalling Agent | `node dist/examples/run-toolcalling-agent.js` |
| LLM               | `node dist/examples/run-llm.js`               |
| Tool              | `node dist/examples/run-tool.js`              |

You should see your spans exported in your console. If you've set up a locally running Phoenix server, open http://localhost:6006 in your browser and navigate to the default project to explore the collected traces.
