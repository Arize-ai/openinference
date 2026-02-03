---
"@arizeai/openinference-vercel": major
"@arizeai/openinference-genai": patch
---

feat(openinference-vercel): BREAKING - Add AI SDK v6 support using openinference-genai

This is a breaking change that refactors openinference-vercel to:

1. **Require AI SDK v6+**: This version only supports AI SDK v6. For AI SDK v5, use openinference-vercel v2.x.

2. **Leverage openinference-genai**: The package now uses `@arizeai/openinference-genai` as the primary converter for standard `gen_ai.*` OpenTelemetry GenAI semantic conventions that AI SDK v6 emits.

3. **Vercel-specific fallback handling**: Vercel-specific `ai.*` attributes are now processed as supplements to the standard `gen_ai.*` attributes, including:
   - Span kind determination from `operation.name`
   - Embeddings (`ai.value`, `ai.embedding`, etc.)
   - Tool calls (`ai.toolCall.*`)
   - Metadata (`ai.telemetry.metadata.*`)
   - Streaming metrics (`ai.response.msToFirstChunk`, etc.)
   - Input/output messages from `ai.prompt.messages` and `ai.response.toolCalls`

4. **New attribute support**: Added support for AI SDK v6 specific attributes including streaming metrics which are stored as metadata.

**Migration Guide:**

- Update AI SDK to v6+: `pnpm install ai@^6.0.0`
- Update openinference-vercel: `pnpm install @arizeai/openinference-vercel@^3.0.0`
- No code changes required - the API remains the same
