---
"@arizeai/openinference-instrumentation-anthropic": patch
---

Stop `@anthropic-ai/sdk` >= 0.134.0 from adding a second span to each traced Messages call. The SDK now starts a span of its own for every API call, so `messages.create()`, `messages.stream()` and `countTokens()` (stable and beta) each exported an OpenInference LLM span and an SDK `anthropic.messages.*` span, and the `messages.stream()` span was not under the LLM span. A traced call now hands the SDK a non-recording span with the LLM span's context, so only the LLM span is recorded, the request still carries the trace context, and HTTP instrumentation spans nest under the LLM span. SDK calls that are not traced keep their spans.
