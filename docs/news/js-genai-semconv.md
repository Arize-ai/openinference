---
title: OpenTelemetry GenAI conventions for JavaScript
description: Set one flag and the JavaScript instrumentations also emit OpenTelemetry GenAI gen_ai.* attributes, right next to the OpenInference ones.
pubDate: 2026-11-20
authors: [mikeldking]
tags: [javascript, spec, opentelemetry]
releases:
  - "@arizeai/openinference-core@2.9.0"
  - "@arizeai/openinference-instrumentation-openai@4.7.0"
  - "@arizeai/openinference-instrumentation-anthropic@1.0.5"
  - "@arizeai/openinference-instrumentation-bedrock@0.5.11"
---

If your backend expects the OpenTelemetry
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/),
you no longer have to choose. The JavaScript instrumentations can now emit
`gen_ai.*` attributes on the same spans as their OpenInference attributes.
Python already could; JavaScript has caught up.

Set one environment variable:

```bash
OPENINFERENCE_ENABLE_GENAI_SEMCONV=true node app.js
```

Or, if you'd rather do it in code, pass
`traceConfig: { enableGenAISemconv: true }` to your instrumentation.

That's it. Your spans keep everything they had and also get the model,
provider, token usage, messages and tool calls under their `gen_ai.*` names.
If you mask inputs or outputs, they stay masked there too.

It works with any instrumentation on `@arizeai/openinference-core` 2.9.0 or
later. It's experimental and off by default, so try it and
[tell us](https://github.com/Arize-ai/openinference/issues) what breaks.

The details are in the [specification](../../spec/configuration/genai-semconv.md).
