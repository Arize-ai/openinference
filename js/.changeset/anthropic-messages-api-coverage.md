---
"@arizeai/openinference-instrumentation-anthropic": major
---

Support the current Anthropic Messages API (stable and beta, `@anthropic-ai/sdk` 0.133)

Behavior changes to note:

- A streamed response's `output.value` is now the message rebuilt from the stream as JSON (`output.mime_type` `application/json`, previously the concatenated text as `text/plain`), the same as for a non-streaming response.
- The top-level `system` prompt is now recorded as input message 0, so the conversation's messages start at index 1 when a system prompt is set. It is no longer part of `llm.invocation_parameters`, which `hideInputs` does not hide.

Changes:

- Streams are observed through the caller's own iterator instead of a tee, so a mid-stream error no longer causes an unhandled rejection, `break` still aborts the request, nothing is buffered, and the span ends OK, unset (left early or aborted) or ERROR. Streamed tool input is parsed once per block.
- Attribute extraction can no longer throw into the caller.
- Server and MCP tool calls are recorded as tool calls; compaction blocks as text; every other content block (tool results, documents, MCP listings, container uploads, unknown types) as its type and JSON, with embedded base64 data replaced by a placeholder and nested images hidden by `hideInputImages`.
- Citations, MCP server names, tool call callers and toolset names, `stop_details`, cache miss reasons, the container id, context management edits, input transformations, iteration usage and server tool usage go to span metadata, merged with the context metadata. Citations are left out when inputs, outputs, input messages or output messages are hidden, and a citation's `cited_text` when input text or output text are hidden, and a refusal's `explanation` when outputs, output messages or output text are hidden.
- Images record URLs and base64 data URIs (Files API images are recorded as their JSON); `llm.tools` records `tool.name` and `tool.description`, and applies the request's `tool_addition` and `tool_removal` blocks.
- Thinking tokens are recorded as `llm.token_count.completion_details.reasoning`.
- `llm.provider` is `aws`, `google` or `azure` for the Bedrock, Vertex and Foundry clients.
- `messages.countTokens` / `beta.messages.countTokens` are traced, and each run of `beta.messages.toolRunner` runs under a `beta.messages.tool_runner` CHAIN span (the naming proposed for the Python instrumentation in PR #3531) parenting its LLM spans and a TOOL span per tool run, including tools added with `addTools()` or `setMessagesParams()`.
