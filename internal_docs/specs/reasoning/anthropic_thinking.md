# Anthropic Messages API: extended thinking block variants

Practical JSON examples for every shape involved in Anthropic's "extended thinking" (reasoning) surface on `POST /v1/messages` (or `messages.create(...)` in the Python SDK). The shapes follow the TypedDict / `BaseModel` names under `anthropic/types/`, which mirror the HTTP schema. Use a section's example as a starting point and drop fields you do not need.

Permalinks below use commit `e8e6f6692632b5fdbea5df1e44cdbd0193fac521` from [anthropics/anthropic-sdk-python](https://github.com/anthropics/anthropic-sdk-python) (v0.101.0). Behavior claims are cross-checked against the [extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) and [streaming](https://platform.claude.com/docs/en/api/messages-streaming) docs pages.

---

## 1. Definitions

**Extended thinking** — the model produces internal reasoning before its visible answer. Three surfaces: **request config** (`ThinkingConfigParam`) — opt in, set a budget, choose visibility; **response blocks** (`ThinkingBlock`, `RedactedThinkingBlock`) inside `Message.content`; **streaming deltas** (`thinking_delta`, `signature_delta`) inside `content_block_delta` envelopes during `stream=true`.

**Signature** — an opaque, base64-encoded token appended to every `ThinkingBlock`. It authenticates the reasoning content server-side. Signature pass-through is **load-bearing** — when building a follow-up turn you must echo every `thinking` block back verbatim (`thinking` **and** `signature`) and keep `redacted_thinking` blocks intact; the API rejects the request (or silently disables thinking) if the `signature` is dropped or mutated.

**Budget tokens** — the model's reasoning budget in tokens, not a hard output ceiling. Must be ≥ 1024 and strictly less than `max_tokens` (except during interleaved thinking, where it is a per-turn total that may exceed `max_tokens`).

**Redacted thinking** — reasoning content suppressed by Anthropic's safety classifiers. The model still reasoned; the content is replaced by an opaque `data` blob. Not an error — echo it back unchanged on subsequent turns.

**Display mode** — controls visibility of thinking text in the response. `"summarized"` (default on Sonnet/Opus 4.6 and earlier Claude 4) shows the text; `"omitted"` (default on Opus 4.7 and Mythos Preview) hides it but still emits a `signature` for multi-turn continuity.

**Full request-config union** (same order as the SDK):

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/thinking_config_param.py#L11-L13)

```python
ThinkingConfigParam: TypeAlias = Union[
    ThinkingConfigEnabledParam,
    ThinkingConfigDisabledParam,
    ThinkingConfigAdaptiveParam,
]
```

**Full response content-block union** (thinking-related members highlighted):

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/content_block.py#L19-L34)

```python
ContentBlock: TypeAlias = Annotated[
    Union[
        TextBlock,
        ThinkingBlock,          # type: "thinking"
        RedactedThinkingBlock,  # type: "redacted_thinking"
        ToolUseBlock,
        ServerToolUseBlock,
        WebSearchToolResultBlock,
        WebFetchToolResultBlock,
        CodeExecutionToolResultBlock,
        BashCodeExecutionToolResultBlock,
        TextEditorCodeExecutionToolResultBlock,
        ToolSearchToolResultBlock,
        ContainerUploadBlock,
    ],
    PropertyInfo(discriminator="type"),
]
```

**Full streaming delta union:**

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/raw_content_block_delta.py#L13-L15)

```python
RawContentBlockDelta = Union[TextDelta, InputJSONDelta, CitationsDelta, ThinkingDelta, SignatureDelta]
```

---

## 2. Input Params

Shapes sent on the **request** side of `POST /v1/messages`.

### `ThinkingConfigEnabledParam`

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/thinking_config_enabled_param.py#L11)

Turn extended thinking on with a fixed token budget. `budget_tokens` must be **≥ 1024** and **strictly less than `max_tokens`**. `display` is optional.

```json
{
  "type": "enabled",
  "budget_tokens": 8000,
  "display": "summarized"
}
```

Minimal:

```json
{
  "type": "enabled",
  "budget_tokens": 8000
}
```

Hide thinking text but keep the signature for multi-turn continuity:

```json
{
  "type": "enabled",
  "budget_tokens": 8000,
  "display": "omitted"
}
```

**Insight:** `budget_tokens` is the model's reasoning budget, not a hard output ceiling. The `budget_tokens < max_tokens` constraint is server-enforced — violation returns HTTP 400 (and blocks the `max_tokens: 0` cache-prewarming pattern). `display: "summarized"` shows thinking text, but **billed tokens cover the full thinking**, not just what appears in `content[].thinking`. `display: "omitted"` hides the text but still emits a `signature`; no `thinking_delta` events are emitted in the stream when this mode is active. Thinking tokens are billed like output tokens whether shown or omitted.

---

### `ThinkingConfigDisabledParam`

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/thinking_config_disabled_param.py#L9)

Explicitly turn extended thinking off. Equivalent to omitting `thinking` from the request on models where thinking is opt-in.

```json
{
  "type": "disabled"
}
```

**Insight:** Use this when a code path **toggles** thinking — passing `disabled` is clearer than conditionally deleting the key and makes intent visible in logs and traces.

---

### `ThinkingConfigAdaptiveParam`

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/thinking_config_adaptive_param.py#L10)

Let the model decide whether to use thinking, and how much, per request. No fixed budget — the model self-selects based on the prompt's difficulty.

```json
{
  "type": "adaptive",
  "display": "summarized"
}
```

Minimal:

```json
{
  "type": "adaptive"
}
```

**Insight:** Adaptive is the right default for **mixed workloads** — chit-chat skips thinking, harder prompts opt in. **Opus 4.7 requires adaptive** — passing `{"type": "enabled"}` returns HTTP 400. On Opus 4.6 / Sonnet 4.6, `enabled` is **deprecated** in favor of `adaptive`. Older Claude 4 models (and Sonnet 3.7) only support `enabled`. Check `ModelInfo.capabilities.thinking.types.adaptive.supported` before assuming (see [§4 Capability discovery](#capability-discovery--thinkingcapability)).

---

### `ThinkingBlockParam` (multi-turn pass-through)

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/thinking_block_param.py#L9)

To continue a conversation that used extended thinking, include the prior assistant turn **with its thinking blocks intact** in `messages`. The param shape is field-for-field identical to the response shape.

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me work through this step by step. First, the problem asks…",
      "signature": "EuYBCkQYAiJAxN2qx7L9..."
    },
    { "type": "text", "text": "Paris." }
  ]
}
```

**Insight:** The most common place to lose thinking continuity: serialization layers that strip `signature`, or template renderers that only inspect `type: "text"` blocks. **Round-trip rule:** whatever came out of `Message.content` for the prior turn goes back into `messages[].content` byte-for-byte. With **tool use**, omitting the block raises an API error. With `display: "omitted"`, the echoed `thinking` text is **ignored** by the server (the `signature` is decrypted back to the original reasoning) — you may pass `"thinking": ""` in that case, but the `signature` is still mandatory. If you must truncate context, drop **whole assistant turns**, not individual blocks within a turn.

---

### `RedactedThinkingBlockParam` (multi-turn pass-through)

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/redacted_thinking_block_param.py#L9)

Same shape as the response block — pass the opaque `data` back unchanged.

```json
{
  "type": "redacted_thinking",
  "data": "EvgBCoYBGAIqQDLkP..."
}
```

**Insight:** Treat `data` like `signature`: opaque, mandatory, never edit. The subtlest bug is JSON encoders that re-quote unicode or whitespace inside `data` — keep the original string.

---

## 3. Output Params

Shapes returned by `POST /v1/messages`, either in `Message.content` (non-streaming) or as streaming events.

### `ThinkingBlock`

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/thinking_block.py#L10)

The model's visible reasoning text plus a `signature` that authenticates the block. Appears in `Message.content` ahead of the final `text` block(s).

```json
{
  "type": "thinking",
  "thinking": "Let me work through this step by step. First, the problem asks…",
  "signature": "EuYBCkQYAiJAxN2qx7L9...REDACTED_FOR_BREVITY...K8w=="
}
```

Full response mixing thinking and text:

```json
{
  "id": "msg_01ABC...",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-4-6",
  "content": [
    {
      "type": "thinking",
      "thinking": "The user asked for the capital of France. That's Paris. I should answer concisely.",
      "signature": "EuYBCkQYAiJAxN2qx7L9..."
    },
    {
      "type": "text",
      "text": "Paris."
    }
  ],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 12,
    "output_tokens": 33,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

**Insight:** `signature` is **opaque and load-bearing** — it must travel back verbatim in the next request when you include this turn in `messages`. Do not log it as content (large, base64-encoded, not human-readable); do log its presence to debug "thinking lost" issues. Treat `thinking` text like model output for privacy review — it can quote the user verbatim and surface intermediate reasoning the final answer drops.

---

### `RedactedThinkingBlock`

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/redacted_thinking_block.py#L10)

Anthropic's safety classifiers redacted this block. The model still reasoned, but its content is hidden behind an opaque `data` blob. Echo it back unchanged on subsequent turns — `data` preserves continuity for surrounding `thinking` blocks.

```json
{
  "type": "redacted_thinking",
  "data": "EvgBCoYBGAIqQDLkP...REDACTED_OPAQUE_PAYLOAD..."
}
```

Full response mixing redacted thinking, visible thinking, and text:

```json
{
  "id": "msg_02XYZ...",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-4-6",
  "content": [
    { "type": "redacted_thinking", "data": "EvgBCoYBGAIqQDLkP..." },
    {
      "type": "thinking",
      "thinking": "Continuing past the redacted segment, the cleaner answer is…",
      "signature": "EuYBCkQYAiJA..."
    },
    { "type": "text", "text": "Here's the safe summary." }
  ],
  "stop_reason": "end_turn",
  "usage": {
    "input_tokens": 42,
    "output_tokens": 88,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0
  }
}
```

**Insight:** A `redacted_thinking` block is **not an error** — it is a normal outcome when the safety system suppresses reasoning that could be misused. Render nothing user-visible; pass it back unchanged; **do not strip it** (stripping breaks the signature chain for surrounding `thinking` blocks). For telemetry capture `{ "type": "redacted_thinking" }` plus the byte-length of `data` — never the `data` payload itself.

---

### `ThinkingDelta` (streaming)

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/thinking_delta.py#L9)

When `stream: true`, each chunk of thinking text arrives as a delta inside `content_block_delta` for the active block index.

```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": { "type": "thinking_delta", "thinking": "Let me work through" }
}
```

```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": { "type": "thinking_delta", "thinking": " this step by step." }
}
```

**Insight:** Accumulate `thinking` strings by `index` exactly as you would `text_delta` — concat in arrival order. **When the request used `display: "omitted"`, no `thinking_delta` events are emitted at all** — the block opens, receives one `signature_delta`, and closes. This is the primary reason `display: "omitted"` improves time-to-first-text-token.

---

### `SignatureDelta` (streaming)

[source](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/signature_delta.py#L9)

The block's `signature` is delivered as **a single delta** near the end of the thinking block — not chunked the way text or thinking is.

```json
{
  "type": "content_block_delta",
  "index": 0,
  "delta": { "type": "signature_delta", "signature": "EuYBCkQYAiJAxN2qx7L9..." }
}
```

**Insight:** Exactly one `signature_delta` per thinking block, always just before `content_block_stop`. Stash it alongside the accumulated `thinking` string at the same `index`. Branch on `delta.type` explicitly: `text_delta` → render; `thinking_delta` → render-as-thinking or hide; `signature_delta` → store on the block; `input_json_delta` / `citations_delta` → unrelated.

---

### `RawContentBlockDelta` — streaming envelope and block boundaries

[source: delta union](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/raw_content_block_delta.py#L13-L15)
[source: delta event](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/raw_content_block_delta_event.py#L9)
[source: start event](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/raw_content_block_start_event.py#L37)

The streaming envelope that carries every delta kind. Each block is bounded by `content_block_start` → one or more `*_delta` events → `content_block_stop`. The `index` field is the block's identity across events.

```json
{ 
  "type": "content_block_start", 
  "index": 0,
  "content_block": {
     "type": "thinking", 
     "thinking": "", 
     "signature": "" 
  } 
}
```

```json
{ 
  "type": "content_block_delta", 
  "index": 0,
  "delta": { 
    "type": "thinking_delta", 
    "thinking": "Let me work through" 
  }
}
```

```json
{ 
  "type": "content_block_delta", 
  "index": 0,
  "delta": { 
    "type": "signature_delta", 
    "signature": "EuYBCkQYAiJA..." 
  } 
}
```

```json
{ 
  "type": "content_block_stop", 
  "index": 0 
}
```

**Insight:** Treat `index` as the **identity** of a streaming block — never assume type from order. The discriminator on `content_block_start.content_block.type` is the reliable signal that "this block is reasoning; expect `thinking_delta` + `signature_delta`" rather than text or tool-use. For `redacted_thinking`, the entire `data` payload lands in `content_block_start` — there are no deltas to accumulate.

---

## 4. Examples

### Complete `messages.create()` request body

A full request wiring together thinking config, a system prompt, a user turn, and an echoed prior assistant turn with thinking intact.

```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 8000
  },
  "system": "You are a helpful assistant.",
  "messages": [
    {
      "role": "user",
      "content": "What is the capital of France?"
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "thinking",
          "thinking": "The user asked for the capital of France. That's Paris.",
          "signature": "EuYBCkQYAiJAxN2qx7L9..."
        },
        { "type": "text", "text": "Paris." }
      ]
    },
    {
      "role": "user",
      "content": "And what is the capital of Germany?"
    }
  ]
}
```

Adaptive variant (correct form for Opus 4.7; no `budget_tokens` required):

```json
{
  "model": "claude-opus-4-7",
  "max_tokens": 16000,
  "thinking": { "type": "adaptive" },
  "messages": [
    { "role": "user", "content": "Explain the halting problem." }
  ]
}
```

**Insight:** The prior assistant turn must carry its `thinking` blocks verbatim — the API authenticates the `signature` server-side. Omit the `thinking` key entirely (or pass `{"type": "disabled"}`) to suppress reasoning for a request; never pass `{"type": "enabled"}` to Opus 4.7 or you get HTTP 400.

---

### Streaming: `thinking` block end-to-end

Full event sequence for a thinking block followed by a text block (events shown in arrival order):

```json
{ 
  "type": "content_block_start", 
  "index": 0,
  "content_block": { 
    "type": "thinking", 
    "thinking": "", 
    "signature": "" 
  } 
}
```

```json
{ 
  "type": "content_block_delta", 
  "index": 0,
  "delta": { 
    "type": "thinking_delta", 
    "thinking": "Let me work through" 
  } 
}
```

```json
{ 
  "type": "content_block_delta", 
  "index": 0,
  "delta": { 
    "type": "thinking_delta", 
    "thinking": " this step by step." 
  } 
}
```

```json
{ 
  "type": "content_block_delta", 
  "index": 0,
  "delta": { 
    "type": "signature_delta", 
    "signature": "EuYBCkQYAiJA..." 
  } 
}
```

```json
{ 
  "type": "content_block_stop", 
  "index": 0 
}
```

```json
{ 
  "type": "content_block_start", 
  "index": 1,
  "content_block": { 
    "type": "text", 
    "text": "" 
  } 
}
```

```json
{ 
  "type": "content_block_delta", 
  "index": 1,
  "delta": { 
    "type": "text_delta", 
    "text": "Paris." 
  } 
}
```

```json
{ 
  "type": "content_block_stop", 
  "index": 1 
}
```

---

### Streaming: `redacted_thinking` block

A `redacted_thinking` block arrives **whole** on `content_block_start` — there is no `redacted_data_delta` type, so the entire `data` payload lands in the start event with no deltas:

```json
{ "type": "content_block_start", "index": 0,
  "content_block": { "type": "redacted_thinking", "data": "EvgBCoYBGAIqQDLkP..." } }
```

```json
{ "type": "content_block_stop", "index": 0 }
```

---

### Streaming: `display: "omitted"` path

When `display: "omitted"` is active, no `thinking_delta` events are emitted. The block opens and closes with only one `signature_delta`:

```json
{ 
  "type": "content_block_start", 
  "index": 0,
  "content_block": { 
    "type": "thinking", 
    "thinking": "", 
    "signature": "" 
  } 
}
```

```json
{ 
  "type": "content_block_delta", 
  "index": 0,
  "delta": { 
    "type": "signature_delta", 
    "signature": "EuYBCkQYAiJA..." 
  } 
}
```

```json
{ 
  "type": "content_block_stop", 
  "index": 0 
}
```

**Insight:** This is the primary reason `display: "omitted"` improves time-to-first-text-token — the thinking block produces no chunked text, only the single terminal signature.

---

### Multi-turn with tool-use interleave

When the assistant turn used thinking **and** called a tool, the order in `Message.content` is canonical and must be preserved across turns:

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "thinking",
      "thinking": "User asked for the weather in Paris. I'll call get_weather.",
      "signature": "EuYBCkQYAiJA..."
    },
    {
      "type": "tool_use",
      "id": "toolu_01ABC",
      "name": "get_weather",
      "input": { "city": "Paris" }
    }
  ]
}
```

The matching user turn returns only `tool_result`; do not insert thinking on the user side:

```json
{
  "role": "user",
  "content": [
    { 
      "type": "tool_result", 
      "tool_use_id": "toolu_01ABC", 
      "content": "18°C, partly cloudy" 
    }
  ]
}
```

**Insight:** "Echo the whole assistant turn" is the rule. The bug pattern to watch for: a tool-runner that rebuilds the assistant turn from only the `tool_use` block, dropping the leading `thinking`. When tool use is interleaved, stripping the `thinking` block causes the API to return an error (not merely degrade quality). **Tool-choice restriction:** when thinking is active, only `tool_choice: {"type": "auto"}` (default) and `tool_choice: {"type": "none"}` are accepted — `{"type": "any"}` and `{"type": "tool", "name": "..."}` return an error ([docs](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)). If your code path conditionally forces a tool, also conditionally pass `thinking: {"type": "disabled"}`.

---

### Capability discovery — `ThinkingCapability`

[source: capability](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/thinking_capability.py#L9)
[source: types matrix](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/thinking_types.py#L8)

`ModelInfo.capabilities.thinking` advertises which thinking variants a model supports:

```json
{
  "thinking": {
    "supported": true,
    "types": {
      "enabled": { "supported": true },
      "adaptive": { "supported": true }
    }
  }
}
```

**Insight:** Check this **at config time**, not at request time, so you fail fast in CI when a model rotation drops adaptive support. Fall back to `enabled` with a sane `budget_tokens` if `adaptive.supported` is false.

---

### Beta surfaces and interleaved thinking

[source: beta clear-thinking edit](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/beta/beta_clear_thinking_20251015_edit_param.py)
[source: beta thinking config union](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/beta/beta_thinking_config_param.py)

The SDK exposes a parallel `beta_*` family covering older or in-development variants — same shapes, different `type` strings and import paths:

- `BetaThinkingBlock` / `BetaThinkingBlockParam` — same `{type, thinking, signature}` shape under the beta namespace.
- `BetaRedactedThinkingBlock` / `BetaRedactedThinkingBlockParam` — same `{type, data}` shape.
- `BetaThinkingConfigEnabledParam` / `BetaThinkingConfigDisabledParam` / `BetaThinkingConfigAdaptiveParam` — config trio mirroring stable.
- `BetaThinkingDelta` / `BetaSignatureDelta` — streaming deltas.
- `BetaClearThinking20251015EditParam` — clears prior thinking from assistant history while keeping the rest of the turn intact (the supported way to drop signatures from context without risking rejection).

**Interleaved thinking.** When needed explicitly (Opus 4.5 / 4.1 / Sonnet 4.5 and older Claude 4 variants), set `anthropic-beta: interleaved-thinking-2025-05-14`. On Mythos Preview, interleaved thinking happens by default with no header; on Opus 4.7 / 4.6 / Sonnet 4.6, use `thinking: {"type": "adaptive"}` — the header is deprecated on those models ([docs](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)). When interleaved thinking is active, `budget_tokens` is a **per-turn total across all thinking blocks** and may exceed `max_tokens`.

**Insight:** Beta variants require the appropriate `anthropic-beta` header and model allowlists; do **not** mix `beta_*` request types with the stable client surface or you get type errors at runtime.

---

### Prompt-caching interaction

Two rules from the [docs](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) worth surfacing in telemetry:

- **System prompts and tools stay cached** across requests even when `thinking` parameters change.
- **Message-level cache breakpoints are invalidated** when `thinking.type` or `budget_tokens` changes between requests.

Because extended-thinking tasks routinely take longer than five minutes, Anthropic recommends the **1-hour cache duration** for turns that contain reasoning, especially in multi-step tool loops.

**Insight:** If your trace dashboard shows cache-hit rate cratering whenever a feature flag flips `thinking.budget_tokens`, this is expected behavior — not a regression. Log `thinking.type` and `budget_tokens` as span attributes so the correlation is visible.

---

## 5. Summary

### Discriminating `type`

[source: config union](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/thinking_config_param.py#L11-L13)
[source: content block union](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/content_block.py#L19-L34)
[source: delta union](https://github.com/anthropics/anthropic-sdk-python/blob/e8e6f6692632b5fdbea5df1e44cdbd0193fac521/src/anthropic/types/raw_content_block_delta.py#L13-L15)

Three discriminators do the heavy lifting in any instrumentation: (1) request `thinking.type` tells you the **intent**; (2) response `content[].type ∈ {"thinking","redacted_thinking"}` tells you what **arrived**; (3) streaming `delta.type ∈ {"thinking_delta","signature_delta"}` tells you what to **accumulate** at the active `index`. Capture all three and you can faithfully reconstruct reasoning behavior across turns without ever decoding the opaque `signature` / `data` payloads.

| Surface | `type` value | Python class | Required fields |
|---|---|---|---|
| Request config | `enabled` | `ThinkingConfigEnabledParam` | `budget_tokens` (≥ 1024, < `max_tokens`); optional `display` |
| Request config | `disabled` | `ThinkingConfigDisabledParam` | — |
| Request config | `adaptive` | `ThinkingConfigAdaptiveParam` | — (optional `display`) |
| Request pass-through | `thinking` | `ThinkingBlockParam` | `thinking`, `signature` |
| Request pass-through | `redacted_thinking` | `RedactedThinkingBlockParam` | `data` |
| Response block | `thinking` | `ThinkingBlock` | `thinking`, `signature` |
| Response block | `redacted_thinking` | `RedactedThinkingBlock` | `data` |
| Streaming delta | `thinking_delta` | `ThinkingDelta` | `thinking` (text chunk) |
| Streaming delta | `signature_delta` | `SignatureDelta` | `signature` (single delta near end of block) |

---

### See also (vendor docs)

Official behavior, model availability, and pricing change over time; confirm in the links below.

- [Extended thinking guide](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) — high-level usage, budgets, visibility, per-model support matrix.
- [Extended thinking with tool use](https://platform.claude.com/docs/en/build-with-claude/extended-thinking#extended-thinking-with-tool-use) — required ordering and signature pass-through in agent loops.
- [Controlling thinking display](https://platform.claude.com/docs/en/build-with-claude/extended-thinking#controlling-thinking-display) — `display: "summarized"` vs `"omitted"` semantics.
- [Preserving thinking blocks](https://platform.claude.com/docs/en/build-with-claude/extended-thinking#preserving-thinking-blocks) — multi-turn signature decryption rules.
- [Messages API reference](https://platform.claude.com/docs/en/api/messages) — request body fields including `thinking`.
- [Streaming Messages](https://platform.claude.com/docs/en/api/messages-streaming) — `content_block_start` / `content_block_delta` / `content_block_stop` semantics and delta types.
- [Interleaved thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking#interleaved-thinking) — beta header `interleaved-thinking-2025-05-14`, per-turn budget semantics, and multi-step agent patterns.
