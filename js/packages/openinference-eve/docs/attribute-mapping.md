# Attribute Mapping

## Overview

Eve injects `eve.*` attributes onto every span in a trace. This processor maps
them to [OpenInference semantic conventions](https://arize-ai.github.io/openinference/spec/).

## `eve.*` → OpenInference Mapping

| Eve attribute | OpenInference attribute | Notes |
|---|---|---|
| `eve.session.id` | `session.id` | Direct mapping; used by Phoenix to group turns into a session |
| `eve.version` | `metadata.eve.version` | Framework version |
| `eve.environment` | `metadata.eve.environment` | `"development"`, `"production"`, etc. |
| `eve.turn.id` | `metadata.eve.turn.id` | Unique turn identifier |
| `eve.turn.sequence` | `metadata.eve.turn.sequence` | 0-based turn counter within a session |
| `eve.step.index` | `metadata.eve.step.index` | 0-based step counter within a turn |
| `eve.channel.kind` | `metadata.eve.channel.kind` | Channel type, e.g. `"channel:terminal"` |

Any `eve.*` attribute not in the table above is also mapped to `metadata.eve.*`
using the same pattern.

## Span Kind Mapping

| `operation.name` prefix | `openinference.span.kind` |
|---|---|
| `ai.eve.turn` | `AGENT` |

All other span kinds (AGENT for `ai.streamText`, LLM for `ai.streamText.doStream`,
TOOL for `ai.toolCall`) are set by the inherited Vercel AI SDK processor.

## Vercel AI SDK Attribute Mapping (Inherited)

The following mappings are handled by `@arizeai/openinference-vercel` and are
applied to every span regardless of Eve:

**LLM span (`ai.streamText.doStream`):**

| Vercel / GenAI attribute | OpenInference attribute |
|---|---|
| `gen_ai.request.model` / `ai.model.id` | `llm.model_name` |
| `gen_ai.system` / `ai.model.provider` | `llm.provider` |
| `gen_ai.usage.input_tokens` / `ai.usage.promptTokens` | `llm.token_count.prompt` |
| `gen_ai.usage.output_tokens` / `ai.usage.completionTokens` | `llm.token_count.completion` |
| `ai.prompt.messages` | `llm.input_messages` (parsed from JSON) |
| `ai.response.toolCalls` | `llm.output_messages` (tool calls converted to messages) |
| `ai.response.text` | `output.value` |

**Tool span (`ai.toolCall`):**

| Vercel attribute | OpenInference attribute |
|---|---|
| `ai.toolCall.name` | `tool.name` |
| `ai.toolCall.args` | `input.value` |
| `ai.toolCall.result` | `output.value` |

## Attribute Precedence

`addEveAttributesToSpan` only sets `openinference.span.kind` if it is not
already present on the span. This means:

1. If an Eve span already has `openinference.span.kind` set (e.g., manually by
   the application), that value is preserved.
2. The Vercel processor's existing guard ("if already set, return early") also
   respects the value set by the Eve processor for `ai.eve.turn` spans.

## Custom Attributes from `runtimeContext`

Attributes set via Eve's `step.started` event `runtimeContext` are added to
the span as plain OTel attributes. They are preserved as-is (not prefixed) and
appear alongside the `eve.*` attributes in Phoenix.

Example: returning `{ "app.user_id": "u-42" }` from `step.started` results in
`app.user_id = "u-42"` on the model call span.
