---
"@arizeai/openinference-instrumentation-typesafe": minor
"@arizeai/openinference-semantic-conventions": minor
"@arizeai/openinference-instrumentation-openai": patch
---

Add TypeSafe AI SDK instrumentation with one LLM span per systemOne call, structured JSON input/output payloads, question confidence metadata, token usage, context propagation, and configurable masking. Preserve the SDK's APIPromise interface and support both ESM and CommonJS. Add TypeSafe provider and system values to the semantic conventions and recognize the TypeSafe API hostname in provider inference.
