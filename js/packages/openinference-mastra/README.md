# OpenInference Mastra

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-mastra.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-mastra)

This package provides a set of utilities to ingest [Mastra](https://github.com/mastra-ai/mastra) spans into platforms like [Arize](https://arize.com/) and [Phoenix](https://phoenix.arize.com/).

## Installation

```shell
npm install --save @arizeai/openinference-mastra
```

A typical Mastra project will already have OpenTelemetry and related packages installed, so you will likely not need to install any additional packages.

## Usage

`@arizeai/openinference-mastra` provides a set of utilities to help you ingest Mastra spans into platforms and works in conjunction with Mastra's OpenTelemetry support. To get started, you will need to add OpenTelemetry support to your Mastra project according to the [Mastra Observability guide](https://mastra.ai/en/reference/observability/providers), or, follow along with the rest of this README.

To process your Mastra spans add an `OpenInferenceOTLPTraceExporter` to your `telemetry` configuration within your `Mastra` instance.

```shell
# Set the Phoenix collector endpoint and API key in your environment
export PHOENIX_COLLECTOR_ENDPOINT="https://localhost:6006/v1/traces"
export PHOENIX_API_KEY="your-api-key"
```

```typescript
import { Mastra } from "@mastra/core";
import { OpenInferenceOTLPTraceExporter } from "@arizeai/openinference-mastra";
 
export const mastra = new Mastra({
  // ... other config
  telemetry: {
    serviceName: "openinference-mastra-agent", // you can rename this to whatever you want to appear in the Phoenix UI
    enabled: true,
    export: {
      type: "custom",
      exporter: new OpenInferenceOTLPTraceExporter({
        collectorEndpoint: process.env.PHOENIX_COLLECTOR_ENDPOINT,
        apiKey: process.env.PHOENIX_API_KEY,
      }),
    },
  },
});
```

For general details on Mastra's OpenTelemetry support see the [Mastra Observability guide](https://mastra.ai/en/docs/observability/tracing).

## Examples

TODO: Add examples