---
"@arizeai/openinference-instrumentation-openai": patch
"@arizeai/openinference-instrumentation-anthropic": patch
---

Scope the double-patch guard to the module object so both the CJS and ESM builds of a dual-package SDK can be patched in the same process. Previously the module-global `_isOpenInferencePatched` flag made whichever build was patched first silently block `patch()`/`manuallyInstrument()` for the other build (#3557). The guard is now a `WeakSet` keyed on the patched class, which needs no write to the module and therefore also keeps protecting immutable modules (Deno, webpack) — the case the global flag existed for. `isPatched()` behavior is unchanged.
