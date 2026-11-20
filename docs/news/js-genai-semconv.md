---
title: OpenTelemetry GenAI conventions for JavaScript
description: The JavaScript instrumentations can now emit OpenTelemetry GenAI gen_ai.* attributes alongside their OpenInference attributes, behind one flag. OpenAI, Anthropic and Bedrock are covered by tests.
pubDate: 2026-11-20
authors: [mikeldking]
tags: [javascript, spec, opentelemetry]
releases:
  - "@arizeai/openinference-core@2.9.0"
  - "@arizeai/openinference-instrumentation-openai@4.7.0"
  - "@arizeai/openinference-instrumentation-anthropic@1.0.5"
  - "@arizeai/openinference-instrumentation-bedrock@0.5.11"
---

OpenInference for JavaScript can now emit OpenTelemetry
[GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
`gen_ai.*` attributes next to its OpenInference attributes, following the same
[specification](../../spec/configuration/genai-semconv.md) as Python. The
export is **experimental** and off by default. One flag turns it on:

```bash
OPENINFERENCE_ENABLE_GENAI_SEMCONV=true node app.js
```

Or set `traceConfig: { enableGenAISemconv: true }` on an instrumentation or an
`OITracer`. The conversion lives in `@arizeai/openinference-core`, so every
instrumentation that depends on core 2.9.0 picks it up.

- **What you get:** `gen_ai.operation.name`, the provider, request and
  response models, request parameters, token usage, finish reasons, and the
  messages and tool definitions as JSON.
- **Masking still applies:** attributes are derived after data masking, so
  hidden content stays hidden.
- **Nothing is overwritten:** `gen_ai.*` attributes you set yourself are kept.

The OpenAI, Anthropic and Bedrock instrumentations now have tests that pin
these attributes, including streaming and tool calls. Checking JavaScript
spans against the GenAI registry with Weaver, as Python does, is tracked in
[#3114](https://github.com/Arize-ai/openinference/issues/3114).

The full mapping is in the
[core docs](../../js/packages/openinference-core/docs/trace-config-and-masking.md#genai-semantic-conventions-export).
