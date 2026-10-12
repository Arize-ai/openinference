---
"@arizeai/openinference-instrumentation-anthropic": minor
---

Support the current Anthropic Messages API (stable and beta, `@anthropic-ai/sdk` 0.133)

- Streams are observed through the caller's own iterator instead of a tee, so a mid-stream error no longer causes an unhandled rejection, `break` still aborts the request, and the span ends OK, unset (left early or aborted) or ERROR. The streamed `output.value` is now the rebuilt message as JSON, like the non-streaming path.
- Attribute extraction can no longer throw into the caller.
- Server and MCP tool calls are recorded as tool calls; compaction blocks as text; every other content block (tool results, MCP listings, container uploads, unknown types) as its type and JSON. Citations, MCP server names, `stop_details`, cache miss reasons, the container id, context management, input transformations, iteration usage and server tool usage go to span metadata, merged with the context metadata.
- The top-level `system` prompt is recorded as the first input message; images record URLs, file ids and base64 data URIs; `llm.tools` records `tool.name`, `tool.description` and inline `tool_addition` tools.
- Thinking tokens are recorded as `llm.token_count.completion_details.reasoning`.
- `llm.provider` is `aws`, `google` or `azure` for the Bedrock, Vertex and Foundry clients.
- `messages.countTokens` / `beta.messages.countTokens` are traced, and `beta.messages.toolRunner` runs under a CHAIN span parenting its LLM spans and a TOOL span per tool run.
