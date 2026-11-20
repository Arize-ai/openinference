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

OpenInference for JavaScript now speaks the OpenTelemetry
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
Set one flag and every span also carries `gen_ai.*` attributes, next to the
OpenInference attributes it already had. Python shipped this first; JavaScript
now follows the same [specification](../../spec/configuration/genai-semconv.md).

The export is **experimental** and off by default.

## Turning it on

```bash
OPENINFERENCE_ENABLE_GENAI_SEMCONV=true node app.js
```

Or in code, on the trace config of an instrumentation or an `OITracer`:

```typescript
const instrumentation = new OpenAIInstrumentation({
  traceConfig: { enableGenAISemconv: true },
});
```

The conversion lives in `OITracer`, in `@arizeai/openinference-core`, so every
instrumentation built on it picks the flag up once it depends on core 2.9.0.
No instrumentation needed a code change to emit the attributes.

## What a span gets

The `gen_ai.*` attributes are derived when the span ends, from the
OpenInference attributes set on it. A chat completion with a tool call comes
out as:

```json
{
  "gen_ai.operation.name": "chat",
  "gen_ai.provider.name": "openai",
  "gen_ai.request.model": "gpt-4o",
  "gen_ai.request.temperature": 0.2,
  "gen_ai.request.max_tokens": 256,
  "gen_ai.response.model": "gpt-4o-2024-08-06",
  "gen_ai.response.id": "chatcmpl-8hhqZDFTRD0vzExhqWnMLE7viVl7E",
  "gen_ai.response.finish_reasons": ["tool_call"],
  "gen_ai.usage.input_tokens": 70,
  "gen_ai.usage.output_tokens": 7,
  "gen_ai.usage.cache_read.input_tokens": 32
}
```

along with `gen_ai.input.messages`, `gen_ai.output.messages` and
`gen_ai.tool.definitions` as JSON. Tool, retriever, embedding and agent spans
are converted too.

Three things hold for every span:

- `gen_ai.*` attributes you set yourself are never overwritten.
- Derivation runs after data masking, so hidden content stays hidden: a
  redacted message is redacted in `gen_ai.input.messages` as well.
- Provider vocabularies are normalized. `max_tokens`, `maxTokens` and
  `maxOutputTokens` all become `gen_ai.request.max_tokens`; `tool_use`,
  `tool_calls` and `function_call` all become the finish reason `tool_call`.

The full mapping is in the
[core docs](../../js/packages/openinference-core/docs/trace-config-and-masking.md#genai-semantic-conventions-export).

## Covered providers

Three instrumentations now have tests that pin the `gen_ai.*` attributes of
their spans, with the flag on, off and combined with masking:

| Instrumentation | Covered                                                                                                  |
| --------------- | -------------------------------------------------------------------------------------------------------- |
| OpenAI          | Chat completions (plain, streamed, multi-turn with tool results), Responses API, embeddings, legacy completions, Azure OpenAI |
| Anthropic       | Messages with tool use, streamed messages                                                                |
| Bedrock         | `InvokeModel`, `InvokeModelWithResponseStream`, `Converse`, `ConverseStream`                             |

To make the OpenAI mapping exact, its spans now record
`llm.request.model_name` and `llm.response.model_name`, which become
`gen_ai.request.model` and `gen_ai.response.model`. `llm.model_name` is
unchanged.

These tests check the attributes against expected values. Checking JavaScript
spans against the GenAI registry with
[Weaver](https://github.com/open-telemetry/weaver), as the Python harness
does, is tracked in
[#3114](https://github.com/Arize-ai/openinference/issues/3114).

## Outside an instrumentation

The conversion is exported for span processors and other code that doesn't
create spans through an `OITracer`:

```typescript
import { getGenAIAttributes } from "@arizeai/openinference-core";

getGenAIAttributes({
  "openinference.span.kind": "LLM",
  "llm.model_name": "gpt-4o",
  "llm.provider": "openai",
});
// {
//   "gen_ai.operation.name": "chat",
//   "gen_ai.provider.name": "openai",
//   "gen_ai.request.model": "gpt-4o",
// }
```

## A caveat

The derived attributes count against the span's attribute limit, 128 by
default. A span already at the limit when it ends gets no `gen_ai.*`
attributes, which can happen on LLM spans with long conversations. Raise
`OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT` for those workloads.
