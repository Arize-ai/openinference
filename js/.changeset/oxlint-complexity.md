---
"@arizeai/openinference-core": patch
"@arizeai/openinference-genai": patch
"@arizeai/openinference-instrumentation-anthropic": patch
"@arizeai/openinference-instrumentation-bedrock-agent-runtime": patch
"@arizeai/openinference-instrumentation-bedrock": patch
"@arizeai/openinference-instrumentation-beeai": patch
"@arizeai/openinference-instrumentation-langchain-v0": patch
"@arizeai/openinference-instrumentation-langchain": patch
"@arizeai/openinference-instrumentation-openai-agents": patch
"@arizeai/openinference-instrumentation-openai": patch
"@arizeai/openinference-vercel": patch
---

Split over-complex functions into focused helpers and make implicit returns explicit (enforce `eslint/complexity`). Also hardens bedrock-agent-runtime tool-call extraction against a `function: null` payload that previously threw. No other behavior changes.
