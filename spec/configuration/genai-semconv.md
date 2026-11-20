# GenAI Semantic Conventions Export

**Experimental.** `OPENINFERENCE_ENABLE_GENAI_SEMCONV` (bool, default `false`; Python `TraceConfig(enable_genai_semconv=True)`, JavaScript `traceConfig: { enableGenAISemconv: true }`)
makes an instrumentation also emit OTel [GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) `gen_ai.*`
attributes, derived at span end from the span's OpenInference attributes.

- Existing `gen_ai.*` attributes MUST NOT be overwritten.
- Derivation runs after [data masking](./data-masking.md), so hidden content stays hidden.
- The mapping is defined by the Python reference,
  [`_genai_conversion.py`](https://github.com/Arize-ai/openinference/blob/main/python/openinference-instrumentation/src/openinference/instrumentation/_genai_conversion.py).

## Status

| Language | Flag | Weaver conformance |
| --- | --- | --- |
| Python | ✅ | ✅ Anthropic, OpenAI, Google GenAI |
| JavaScript | ✅ spans created through `OITracer` (`@arizeai/openinference-core`) | 🗓️ [#3114](https://github.com/Arize-ai/openinference/issues/3114) |
| Java | 🗓️ [#2983](https://github.com/Arize-ai/openinference/issues/2983) | 🗓️ [#3114](https://github.com/Arize-ai/openinference/issues/3114) |
| Go | 🗓️ [#2983](https://github.com/Arize-ai/openinference/issues/2983) | 🗓️ [#3114](https://github.com/Arize-ai/openinference/issues/3114) |

🗓️ = planned; the flag is ignored.

## Conformance

```bash
uv run python/openinference-instrumentation/scripts/conformance/run.py
```

Runs each provider against a mock server with the flag on and checks the spans with [Weaver](https://github.com/open-telemetry/weaver) `registry live-check`
against the GenAI semconv registry. Non-registry OpenInference attributes are expected; only `gen_ai.*` is judged.
See the [harness README](https://github.com/Arize-ai/openinference/blob/main/python/openinference-instrumentation/scripts/conformance/README.md).
New language implementations SHOULD add an equivalent harness.
