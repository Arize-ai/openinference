---
name: openinference
description: >
  Core OpenInference concepts shared by every language: span kinds, semantic-convention
  attributes and flattening rules, context attributes (session, user, metadata, tags,
  prompt template), resource attributes (project name), privacy masking via TraceConfig /
  OPENINFERENCE_HIDE_* env vars, and suppressing tracing. Use when tracing LLM, agent, or
  RAG applications with OpenTelemetry, choosing a span kind or attribute, or reviewing spans
  destined for Phoenix or Arize. Pair with openinference-instrument-py or
  openinference-instrument-js for language-specific code.
---

# OpenInference

OpenTelemetry semantic conventions for LLM applications. Spans are ordinary OTel spans
plus `openinference.span.kind` and a fixed attribute vocabulary. Any OTel backend accepts
them; Phoenix and Arize render them natively.

## Rules

1. Every span carries `openinference.span.kind`. The request root is usually `CHAIN` or `AGENT`.
2. Prefer auto-instrumentors for libraries. Add manual spans only for your own logic.
3. Attribute values are primitives or flat lists of primitives. JSON-stringify anything
   else (`metadata`, `llm.invocation_parameters`, tool arguments, document metadata).
4. Lists of objects flatten to `prefix.<index>.suffix`, zero-based:
   `llm.input_messages.0.message.role`.
5. Import attribute keys from the semantic-conventions package; never type them by hand.
6. Mask sensitive data with trace config, not ad hoc string scrubbing.

## Span kinds

| Kind | Use for |
| --- | --- |
| `LLM` | one chat or completion call |
| `EMBEDDING` | one embedding call |
| `CHAIN` | request root or glue between steps |
| `AGENT` | reasoning loop that drives LLM and tools |
| `TOOL` | a function run on the model's behalf |
| `RETRIEVER` | fetching documents from a store |
| `RERANKER` | re-scoring retrieved documents |
| `GUARDRAIL` | safety check on input or output |
| `EVALUATOR` | scoring a model output |
| `PROMPT` | rendering a prompt template |

## Attributes most often set

- Any kind: `input.value`, `input.mime_type`, `output.value`, `output.mime_type`
  (`text/plain` or `application/json`).
- `LLM`: `llm.model_name`, `llm.provider`, `llm.system`, `llm.invocation_parameters` (JSON),
  `llm.input_messages.N.message.{role,content}`, `llm.output_messages.N.message.{role,content}`,
  `llm.output_messages.N.message.tool_calls.M.tool_call.{id,function.name,function.arguments}`,
  `llm.token_count.{prompt,completion,total}`, `llm.tools.N.tool.json_schema`.
- `TOOL`: `tool.name`, `tool.description`, `tool.parameters` (JSON schema).
- `RETRIEVER`: `retrieval.documents.N.document.{id,content,score,metadata}`.
- `EMBEDDING`: `embedding.model_name`, `embedding.embeddings.N.embedding.{text,vector}`.
- `AGENT`: `agent.name`, `graph.node.{id,name,parent_id}`.

Full table by span kind: [references/attributes.md](references/attributes.md).
Spec: https://github.com/Arize-ai/openinference/tree/main/spec

## Context attributes

Attach once to the OTel Context at the request boundary. Every span created inside,
automatic or manual, copies them: `session.id`, `user.id`, `metadata` (JSON),
`tag.tags` (list of strings), `llm.prompt_template.{template,variables,version}`.
A session id groups a multi-turn conversation and must stay stable across turns.

## Resource attributes

`openinference.project.name` on the OTel Resource routes traces to a project in Phoenix
or Arize. Set it when building the TracerProvider, never per span. Keep OTel's standard
`service.name` alongside it.

## Privacy masking

Configure with env vars, or in code via TraceConfig. Precedence: code, then env, then
default. Hidden values are replaced by the string `__REDACTED__`. Masking is applied by
the `OITracer` wrapper, so spans from a raw OTel tracer are never masked.

`OPENINFERENCE_HIDE_INPUTS`, `_HIDE_OUTPUTS`, `_HIDE_INPUT_MESSAGES`, `_HIDE_OUTPUT_MESSAGES`,
`_HIDE_INPUT_TEXT`, `_HIDE_OUTPUT_TEXT`, `_HIDE_INPUT_IMAGES`, `_HIDE_LLM_INVOCATION_PARAMETERS`,
`_HIDE_LLM_TOOLS`, `_HIDE_EMBEDDINGS_VECTORS`, `_HIDE_EMBEDDINGS_TEXT`, `_HIDE_PROMPTS`,
`_HIDE_CHOICES` (all booleans, default false); `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH`
(int, default 32000). `HIDE_INPUTS` also hides input messages and tool definitions;
`HIDE_OUTPUTS` also hides output messages.

## Suppressing tracing

OTel's suppress-instrumentation context flag pauses every OpenInference instrumentor for
the enclosed code. Use it around evaluations, health checks, and internal LLM calls whose
spans would be noise. `uninstrument()` on an instrumentor disables it permanently.
