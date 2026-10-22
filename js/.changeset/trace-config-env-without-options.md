---
"@arizeai/openinference-core": patch
---

Read the `OPENINFERENCE_*` environment variables when no `traceConfig` is passed. `generateTraceConfig()` returned the hard-coded defaults for an undefined config, so `OPENINFERENCE_HIDE_INPUTS=true` and the other masking variables had no effect on any instrumentation constructed without a `traceConfig`.
