# OpenInference attribute reference

Grouped by where they appear. `N`, `M` are zero-based indexes. "JSON" means a JSON string.
Complete spec: https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md

## Every span

| Attribute | Type | Notes |
| --- | --- | --- |
| `openinference.span.kind` | string | required; one of the span kinds |
| `input.value` / `output.value` | string | text or JSON payload |
| `input.mime_type` / `output.mime_type` | string | `text/plain` or `application/json` |
| `session.id` | string | context attribute |
| `user.id` | string | context attribute |
| `metadata` | JSON | context attribute, arbitrary key/values |
| `tag.tags` | list[string] | context attribute |
| `exception.type` / `exception.message` / `exception.stacktrace` | string | via OTel `record_exception` |
| `exception.escaped` | bool | |

## LLM spans

| Attribute | Type | Notes |
| --- | --- | --- |
| `llm.model_name` | string | model that answered; falls back to requested |
| `llm.request.model_name` / `llm.response.model_name` | string | only when the provider reports both |
| `llm.provider` | string | host: `openai`, `anthropic`, `azure`, `google`, `aws`, `cohere`, `mistralai`, `xai`, `deepseek`, `groq`, `fireworks`, `together`, `ollama`, `perplexity`, `cerebras`, `moonshot`, `meta`, `zai`, `minimax` |
| `llm.system` | string | model family: `openai`, `anthropic`, `vertexai`, `cohere`, `mistralai`, `xai`, `deepseek`, `amazon`, `meta`, `ai21` |
| `llm.invocation_parameters` | JSON | temperature, max_tokens, etc.; no messages |
| `llm.input_messages.N.message.role` | string | `system`, `user`, `assistant`, `tool` |
| `llm.input_messages.N.message.content` | string | plain text content |
| `llm.input_messages.N.message.contents.M.message_content.type` | string | `text`, `image`, `audio`, `reasoning`, `tool_use` for multimodal or ordered parts |
| `llm.input_messages.N.message.contents.M.message_content.text` | string | |
| `llm.input_messages.N.message.contents.M.message_content.image.image.url` | string | URL or base64 data URI |
| `llm.input_messages.N.message.tool_call_id` | string | on `tool` role messages, matches `tool_call.id` |
| `llm.input_messages.N.message.name` | string | tool name on `tool` role messages |
| `llm.output_messages.N.message.role` / `.content` | string | same shape as input |
| `llm.output_messages.N.message.tool_calls.M.tool_call.id` | string | |
| `llm.output_messages.N.message.tool_calls.M.tool_call.function.name` | string | |
| `llm.output_messages.N.message.tool_calls.M.tool_call.function.arguments` | JSON | |
| `llm.tools.N.tool.json_schema` | JSON | full tool definition advertised to the model |
| `llm.tools.N.tool.name` / `.description` | string | |
| `llm.token_count.prompt` / `.completion` / `.total` | int | prompt includes cache tokens |
| `llm.token_count.prompt_details.cache_read` / `.cache_write` / `.audio` | int | sub-counts of prompt |
| `llm.token_count.completion_details.reasoning` / `.audio` | int | sub-counts of completion |
| `llm.cost.prompt` / `.completion` / `.total` | float | USD |
| `llm.finish_reason` | string | `stop`, `length`, `tool_calls` |
| `llm.prompts.N.prompt.text` / `llm.choices.N.completion.text` | string | legacy completions API |
| `llm.prompt_template.template` / `.variables` (JSON) / `.version` | string | context attribute |

## TOOL spans

| Attribute | Type | Notes |
| --- | --- | --- |
| `tool.name` | string | |
| `tool.description` | string | |
| `tool.parameters` | JSON | JSON schema of the tool input |
| `tool.id` | string | matches the `tool_call.id` that invoked it |

## RETRIEVER and RERANKER spans

| Attribute | Type | Notes |
| --- | --- | --- |
| `retrieval.documents.N.document.id` | string | |
| `retrieval.documents.N.document.content` | string | |
| `retrieval.documents.N.document.score` | float | |
| `retrieval.documents.N.document.metadata` | JSON | |
| `reranker.query` | string | |
| `reranker.model_name` | string | |
| `reranker.top_k` | int | |
| `reranker.input_documents.N.document.*` / `reranker.output_documents.N.document.*` | | same document fields |

## EMBEDDING spans

| Attribute | Type | Notes |
| --- | --- | --- |
| `embedding.model_name` | string | do not set `llm.model_name`, `llm.provider`, `llm.system` |
| `embedding.invocation_parameters` | JSON | |
| `embedding.embeddings.N.embedding.text` | string | |
| `embedding.embeddings.N.embedding.vector` | list[float] | |

## AGENT and graph spans

| Attribute | Type | Notes |
| --- | --- | --- |
| `agent.name` | string | |
| `graph.node.id` | string | node in the execution graph |
| `graph.node.name` | string | display name |
| `graph.node.parent_id` | string | unset or empty for the root node |

## Media and prompt provenance

| Attribute | Type | Notes |
| --- | --- | --- |
| `image.url` | string | |
| `audio.url` / `audio.mime_type` / `audio.transcript` | string | |
| `prompt.vendor` / `prompt.id` / `prompt.url` | string | where a managed prompt came from |

## Annotations and evaluations

Feedback attached to a span: `annotations.N.annotation.{name,label,score,explanation,annotator_kind,identifier,metadata}`
or the synonym `evaluations.N.evaluation.*`. Prefix with `trace.` or `session.` to widen the target.
`annotator_kind` is `HUMAN`, `LLM`, or `CODE`.

## Resource

| Attribute | Type | Notes |
| --- | --- | --- |
| `openinference.project.name` | string | groups traces into a project; set on the Resource |
