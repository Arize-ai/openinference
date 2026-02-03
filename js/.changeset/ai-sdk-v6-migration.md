---
"@arizeai/openinference-vercel": minor
---

feat(openinference-vercel): add AI SDK v6 telemetry support

This release improves compatibility with AI SDK v6 telemetry while keeping best-effort compatibility with older AI SDK versions.

Key behavior:

- Prefer standard `gen_ai.*` attributes (OTel GenAI semantic conventions) when present
- Fall back to Vercel-specific `ai.*` attributes for data not available in `gen_ai.*` and for older SDK versions

Vercel-specific `ai.*` processing includes:

- Span kind determination from `operation.name`
- Embeddings (`ai.value`, `ai.embedding`, etc.)
- Tool calls (`ai.toolCall.*`)
- Metadata (`ai.telemetry.metadata.*`)
- Streaming metrics (`ai.response.msToFirstChunk`, etc.)
- Input/output messages from `ai.prompt.messages` and `ai.response.toolCalls`

Additional improvements:

- Root AI SDK spans now have a status set (`OK`/`ERROR`) based on the overall invocation result.

Notes:

- AI SDK telemetry is experimental; older versions are supported on a best-effort basis.

**Migration Guide:**

- If you are on AI SDK v6: no code changes required.
- If you are on older AI SDK versions: no code changes required; compatibility is best-effort.
