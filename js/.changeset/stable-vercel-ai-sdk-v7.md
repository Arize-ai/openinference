---
"@arizeai/openinference-vercel": major
---

Add support for stable Vercel AI SDK v7 telemetry through `@ai-sdk/otel`. This release updates the Vercel span processor to convert AI SDK v7 GenAI semantic convention spans into idiomatic OpenInference AGENT, CHAIN, LLM, TOOL, EMBEDDING, and RERANKER spans, including model/provider metadata, token counts, cache-token details, runtime context metadata, tool definitions, tool calls, tool results, and agent names.

This also updates the package to target the stable AI SDK v7 package set and Node.js 22 or newer. AI SDK v6 users should remain on the latest v2 release of `@arizeai/openinference-vercel`.
