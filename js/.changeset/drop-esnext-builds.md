---
"@arizeai/openinference-core": patch
"@arizeai/openinference-instrumentation-bedrock-agent-runtime": patch
"@arizeai/openinference-instrumentation-bedrock": patch
"@arizeai/openinference-instrumentation-beeai": patch
"@arizeai/openinference-instrumentation-langchain-v0": patch
"@arizeai/openinference-instrumentation-langchain": patch
"@arizeai/openinference-instrumentation-mcp": patch
"@arizeai/openinference-instrumentation-openai": patch
"@arizeai/openinference-semantic-conventions": patch
---

Stop building and publishing the unused `dist/esnext` output and the non-standard `esnext` package field. Entry points resolved through `exports`, `main` and `module` are unchanged.
