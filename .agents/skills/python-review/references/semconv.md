# Semantic conventions

All judgment. Check every attribute name against
`python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py`, and
match the conventions to what the library does. Not every span needs every attribute.

## Helpers

Instrumentors build attributes with the helpers in
`python/openinference-instrumentation/src/openinference/instrumentation/_attributes.py`
(`get_input_attributes`, `get_output_attributes`, `get_llm_attributes`,
`get_embedding_attributes`, `get_retriever_attributes`, `get_tool_attributes`, and others),
all exported from `openinference.instrumentation`. Hand-built keys are where typos and
flattening bugs come from.

## Every span

- `OPENINFERENCE_SPAN_KIND`
- `INPUT_VALUE` with `INPUT_MIME_TYPE`, and `OUTPUT_VALUE` with `OUTPUT_MIME_TYPE`.
  Use `application/json` for structured data and `text/plain` for strings. A value
  without its MIME type is High.

## Indexed keys

Lists flatten under an index: `llm.input_messages.<i>.message.role`,
`llm.output_messages.<i>.message.tool_calls.<j>.tool_call.function.name`,
`embedding.embeddings.<i>.embedding.text`, `retrieval.documents.<i>.document.id`. A flat
key where an indexed one belongs is High, because Phoenix will not render it.

## LLM spans

Span kind `LLM`, with `LLM_MODEL_NAME`, `LLM_PROVIDER`, `LLM_SYSTEM` when known,
`LLM_INVOCATION_PARAMETERS` (JSON), `LLM_INPUT_MESSAGES`, `LLM_OUTPUT_MESSAGES`,
`LLM_TOOLS` for tool definitions sent with the request (`llm.tools.<i>.tool.json_schema`),
`LLM_TOKEN_COUNT_PROMPT`, `LLM_TOKEN_COUNT_COMPLETION`, `LLM_TOKEN_COUNT_TOTAL`, and token
details such as `LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ` and
`LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING` when the response reports them.

## Tool calls

It depends on who runs the tool.
- **LLM client libraries** (OpenAI, Anthropic, and similar) never run tools. The LLM span
  records definitions in `LLM_TOOLS` and the model's calls on the output messages. These
  packages create no `TOOL` spans, so never flag a missing one.
- **Frameworks that execute tools** create one `TOOL` span per call with `TOOL_NAME`,
  `TOOL_DESCRIPTION` when available, and `TOOL_PARAMETERS`.

## Embedding and retrieval spans

`EMBEDDING` spans carry `EMBEDDING_MODEL_NAME` and per-item `EMBEDDING_TEXT` and
`EMBEDDING_VECTOR` under `EMBEDDING_EMBEDDINGS`. `RETRIEVER` spans carry per-document
`DOCUMENT_ID`, `DOCUMENT_CONTENT`, `DOCUMENT_SCORE`, and `DOCUMENT_METADATA` under
`RETRIEVAL_DOCUMENTS`.

## Span kinds in a hierarchy

Agent and orchestration frameworks nest `CHAIN` for workflows, `AGENT` for agent runs,
`TOOL` for tool calls, and `LLM` for model calls. Common shapes are `CHAIN -> LLM`,
`CHAIN -> AGENT -> TOOL`, `CHAIN -> AGENT -> LLM`, and `CHAIN -> RETRIEVER -> EMBEDDING`.
A wrong kind changes how Phoenix renders the trace. Prove a kind claim with `phoenix-verify`.
