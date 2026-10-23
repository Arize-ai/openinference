---
"@arizeai/openinference-vercel": minor
---

Support OpenTelemetry JS SDK 3.x. The span processors now build on `@opentelemetry/sdk-trace` instead of `@opentelemetry/sdk-trace-base`, which SDK 3.0 removed. `sdk-trace-base` was never a declared dependency, so with `@opentelemetry/api` 1.10 the package either failed to load (`ERR_MODULE_NOT_FOUND`) or failed to install (ERESOLVE). `@opentelemetry/sdk-trace` is now a peer dependency (`^2.9.0 || ^3.0.0`), so the processors work on both the SDK 2.x and 3.x lines. On SDK 2.x, install `@opentelemetry/sdk-trace@^2` alongside your other 2.x packages. `@opentelemetry/core` now also accepts 3.x. Span attributes with complex values, which SDK 3.x allows, are no longer passed to the GenAI attribute conversion, which only reads primitive values.
