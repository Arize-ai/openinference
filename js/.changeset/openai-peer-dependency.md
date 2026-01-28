---
"@arizeai/openinference-instrumentation-openai": patch
---

Add openai as peer dependency to fix strict dependency management issues with pnpm and bazel. The openai SDK is required at runtime for instanceof checks, not just for types.
