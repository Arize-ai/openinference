# Reasoning surfaces — vendor comparison

Side-by-side comparison of how OpenAI, Anthropic, and Google Gemini expose reasoning / thinking on their respective APIs. Three axes:

1. **Request params** — how you opt in and configure reasoning.
2. **Response parts** — what comes back when the model reasoned.
3. **Multi-turn propagation** — what you have to echo back to keep the chain-of-thought alive.

For deep details on any single vendor, see the sibling docs:

- [`openai_reasoning.md`](./openai_reasoning.md)
- [`anthropic_thinking.md`](./anthropic_thinking.md)
- [`gemini_thinking.md`](./gemini_thinking.md)

---

## 0. At-a-glance

| Axis | OpenAI (Responses) | Anthropic (Messages) | Google (Gemini) |
|---|---|---|---|
| API endpoint | `POST /v1/responses` | `POST /v1/messages` | `POST /v1beta/models/{model}:generateContent` |
| Vendor term | "reasoning" | "extended thinking" | "thinking" |
| Opt-in shape | `reasoning: { effort, summary }` | `thinking: { type, budget_tokens?, display? }` | `generationConfig.thinkingConfig: { thinkingBudget? \| thinkingLevel?, includeThoughts? }` |
| Budget knob | categorical `effort` | numeric `budget_tokens` (2.x) — adaptive (no budget) on Opus 4.7 | numeric `thinkingBudget` (2.5) / categorical `thinkingLevel` (3.x) |
| Visible reasoning | summary parts (raw text gated) | full thinking text (or `display: "omitted"`) | summary parts only (`includeThoughts: true`) |
| Continuity token | `encrypted_content` (opaque) | `signature` (opaque, authenticates plaintext) | `thoughtSignature` (opaque, encrypts state) |
| Stateful path | `previous_response_id` + `store: true` | none — always echo | none — always echo |
| Stateless path | echo prior `output[]` + `include: ["reasoning.encrypted_content"]` | echo prior `content[]` with `signature` intact | echo prior `parts[]` with `thoughtSignature` intact |
| Drop continuity → result | server rejects when a reasoning item immediately preceding a `function_call` is dropped; otherwise reasoning replays from scratch | server rejects when a `thinking` block immediately preceding a `tool_use` is dropped; outside tool use the request is accepted but thinking is disabled | Gemini 3 + tools: HTTP 400 when `thoughtSignature` is missing from a `functionCall` part; Gemini 2.5: silent quality loss |
| Streaming envelope | typed SSE events (`response.*`) | typed SSE events (`content_block_*`) | partial `GenerateContentResponse` chunks (no envelope) |
| Usage field for reasoning tokens | `output_tokens_details.reasoning_tokens` | implicit in `output_tokens` (no separate field) | `usageMetadata.thoughtsTokenCount` |

---

## 1. Request params (side by side)

How each vendor accepts the "turn reasoning on" config. All three are optional, top-level keys on the request body.

### Minimal enable

OpenAI:

```json
{
  "model": "gpt-5",
  "input": "...",
  "reasoning": { "effort": "medium" }
}
```

Anthropic:

```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 8000
  },
  "messages": [...]
}
```

Gemini:

```json
{
  "contents": [...],
  "generationConfig": {
    "thinkingConfig": {
      "thinkingBudget": 8000
    }
  }
}
```

### Adaptive / dynamic budget

| OpenAI | Anthropic | Gemini |
|---|---|---|
| `"effort": "medium"` (categorical — model picks within the level) | `{"type": "adaptive"}` (no budget; required on Opus 4.7) | `"thinkingBudget": -1` (2.5) or `"thinkingLevel": "high"` (3.x) |

### Visible reasoning verbosity

| OpenAI | Anthropic | Gemini |
|---|---|---|
| `"summary": "auto" \| "concise" \| "detailed"` (Responses only) | `"display": "summarized" \| "omitted"` | `"includeThoughts": true` (else no summary parts at all) |

### Disable / off

| OpenAI | Anthropic | Gemini |
|---|---|---|
| Omit `reasoning` (or use non-reasoning model) | `{"type": "disabled"}` | `"thinkingBudget": 0` (Flash / Flash-Lite only; not 2.5 Pro / 3.1 Pro) |

### Full field tables

**OpenAI `Reasoning` (Responses API):**

| Field | Type | Notes |
|---|---|---|
| `effort` | `"none" \| "minimal" \| "low" \| "medium" \| "high" \| "xhigh"` | Model-dependent defaults & allowed values |
| `summary` | `"auto" \| "concise" \| "detailed"` | Responses API only |
| `generate_summary` | same as `summary` | Deprecated alias |
| `include[]` (top-level) | contains `"reasoning.encrypted_content"` | Required for stateless multi-turn |

Chat Completions exposes only `reasoning_effort` (top-level); no summary, no encrypted-content opt-in, no reasoning items in the response.

**Anthropic `ThinkingConfigParam` (union):**

| Variant | Fields | Notes |
|---|---|---|
| `"enabled"` | `budget_tokens` (≥1024, < `max_tokens`); optional `display` | Deprecated on Opus 4.6 / Sonnet 4.6; rejected on Opus 4.7 |
| `"disabled"` | — | Explicit off |
| `"adaptive"` | optional `display` | Required on Opus 4.7; preferred on 4.6 / Sonnet 4.6 |

**Gemini `ThinkingConfig`:**

| Field | Type | Notes |
|---|---|---|
| `thinkingBudget` | `int` | Gemini 2.5; `-1` dynamic, `0` disabled (Flash only) |
| `thinkingLevel` | `"minimal" \| "low" \| "medium" \| "high"` | Gemini 3.x; replaces budget |
| `includeThoughts` | `bool` | Surface summary parts in the response |

---

## 2. Response parts (side by side)

What lands in the response when the model reasoned. Each vendor uses a different container, but all three follow the same pattern: **one or more reasoning slots, then the visible answer**.

### Shape of a single reasoning element

| Vendor | Container path | Discriminator | Visible reasoning carrier | Continuity token field |
|---|---|---|---|---|
| OpenAI | `Response.output[]` | `type == "reasoning"` | `summary[]` items with `type: "summary_text"` (and gated `content[]` with `type: "reasoning_text"`) | `encrypted_content` (string, opaque) |
| Anthropic | `Message.content[]` | `type == "thinking"` or `type == "redacted_thinking"` | `thinking` field (plaintext) on the block | `signature` (string, base64) — or `data` for redacted blocks |
| Gemini | `candidates[].content.parts[]` | `thought == true` (summary), or any data-bearing part with `thoughtSignature` | `text` on a `thought: true` part | `thoughtSignature` (bytes / base64 on wire) |

### Canonical JSON examples

OpenAI reasoning item:

```json
{
  "id": "rs_abc123",
  "type": "reasoning",
  "summary": [
    {
      "type": "summary_text",
      "text": "User asked for the capital..."
    }
  ],
  "content": null,
  "encrypted_content": "gAAAAA...",
  "status": "completed"
}
```

Anthropic thinking block:

```json
{
  "type": "thinking",
  "thinking": "Let me work through this...",
  "signature": "EuYBCkQYAiJA..."
}
```

Gemini thought summary + signed answer:

```json
[
  {
    "thought": true,
    "text": "The user asked for the capital..."
  },
  {
    "text": "Paris.",
    "thoughtSignature": "CiQB..."
  }
]
```

### Ordering within an assistant turn

| OpenAI | Anthropic | Gemini |
|---|---|---|
| `reasoning` → `message` (or `reasoning` → `function_call`) | `thinking` (or `redacted_thinking`) → `text` (or `tool_use`) | `thought: true` summary part(s) → answer part(s) (`text` or `functionCall`) |

Across all three, the reasoning slot precedes the visible answer that depends on it.

### Where the continuity token sits

| OpenAI | Anthropic | Gemini |
|---|---|---|
| One `encrypted_content` per reasoning item; populated **only** if `include: ["reasoning.encrypted_content"]` was set on the request | One `signature` per `thinking` block; always populated for non-redacted blocks; `data` plays the same role for `redacted_thinking` | Zero-or-more `thoughtSignature`s, **attached to data-bearing parts** (functionCall, text, etc.) — **never on `thought: true` summary parts** |

### Streaming surface

| OpenAI | Anthropic | Gemini |
|---|---|---|
| Typed SSE events: `response.output_item.added/.done` envelope; inside: `response.reasoning_summary_part.added/.done`, `response.reasoning_summary_text.delta/.done`, gated `response.reasoning_text.delta/.done`. `encrypted_content` arrives **atomically** on `output_item.done`. | Typed SSE events: `content_block_start/.stop` envelope keyed by `index`; deltas are `{ type: "thinking_delta", thinking: "..." }` and a **single** `{ type: "signature_delta", signature: "..." }` near the end of the block. | No envelope. Each SSE chunk is a partial `GenerateContentResponse`; thought-summary chunks come with `thought: true` and incremental `text`. Signatures can arrive on a part with empty `text` — must parse until `finishReason` is set. |

---

## 3. Multi-turn propagation (side by side)

The load-bearing question: **what must you echo on the next turn to keep reasoning alive?**

### Two paths to continuity

| Vendor | Stateful path | Stateless path |
|---|---|---|
| OpenAI | `store: true` + `previous_response_id` (server replays reasoning) | `store: false` + `include: ["reasoning.encrypted_content"]` + echo every prior `output[]` item in `input[]` |
| Anthropic | none — there is no server-side conversation state | Always echo: prior assistant `content[]` blocks (with `signature`) must appear in `messages[].content` of the next request |
| Gemini | none for raw API; SDK `chats` helper manages history client-side | Always echo: prior model `parts[]` (with `thoughtSignature`) must appear in `contents[]` of the next request |

### What must be echoed

| OpenAI (stateless) | Anthropic | Gemini |
|---|---|---|
| Every `reasoning` + `message` + `function_call` + `function_call_output` item between the last user message and the assistant turn you're continuing | Every `thinking` / `redacted_thinking` / `text` / `tool_use` block of the prior assistant turn, byte-for-byte | Every `parts[]` element of the prior model turn, with `thoughtSignature` intact on its originating part |

### Side-by-side request snippets

OpenAI stateless echo (note `encrypted_content`):

```json
{
  "model": "gpt-5",
  "store": false,
  "include": ["reasoning.encrypted_content"],
  "reasoning": { "effort": "medium" },
  "input": [
    { "role": "user", "content": "..." },
    {
      "type": "reasoning",
      "id": "rs_1",
      "summary": [{ "type": "summary_text", "text": "..." }],
      "encrypted_content": "gAAAAA...=="
    },
    {
      "type": "message",
      "id": "msg_1",
      "role": "assistant",
      "content": [{ "type": "output_text", "text": "..." }]
    },
    { "role": "user", "content": "follow-up" }
  ]
}
```

Anthropic echo (note `signature`):

```json
{
  "model": "claude-opus-4-6",
  "max_tokens": 16000,
  "thinking": { "type": "enabled", "budget_tokens": 8000 },
  "messages": [
    { "role": "user", "content": "..." },
    {
      "role": "assistant",
      "content": [
        {
          "type": "thinking",
          "thinking": "...",
          "signature": "EuYBCkQYAiJA..."
        },
        { "type": "text", "text": "..." }
      ]
    },
    { "role": "user", "content": "follow-up" }
  ]
}
```

Gemini echo (note `thoughtSignature` on the functionCall part):

```json
{
  "contents": [
    { "role": "user", "parts": [{ "text": "..." }] },
    {
      "role": "model",
      "parts": [
        {
          "functionCall": {
            "name": "get_current_temperature",
            "args": { "location": "Paris" }
          },
          "thoughtSignature": "CiQB..."
        }
      ]
    },
    {
      "role": "user",
      "parts": [
        {
          "functionResponse": {
            "name": "get_current_temperature",
            "response": { "temperature": 14.5, "unit": "C" }
          }
        }
      ]
    }
  ],
  "generationConfig": {
    "thinkingConfig": { "thinkingBudget": 8000 }
  },
  "tools": [{ "functionDeclarations": [/* ... */] }]
}
```

### Tool-use interleave — the rule across all three

| OpenAI | Anthropic | Gemini |
|---|---|---|
| When reasoning immediately precedes a `function_call`, the server **rejects** a next-turn request that omits the reasoning item. Parallel branches: every branch's reasoning items must be echoed. | When thinking immediately precedes a `tool_use`, omitting the `thinking` block **returns an API error**. Tool-choice is also restricted: only `auto` / `none` are accepted while thinking is active. | Gemini 3 + tool use: dropping the `thoughtSignature` from a `functionCall` part returns **HTTP 400** (`Function call ... is missing a thought_signature.`). Parallel calls: only the **first** `functionCall` carries a signature; subsequent ones don't. Gemini 2.5: drop = silent quality loss, not an error. |

### Signature vs encrypted-content semantics (subtle)

| OpenAI `encrypted_content` | Anthropic `signature` | Gemini `thoughtSignature` |
|---|---|---|
| Opaque ciphertext that **carries** the chain-of-thought across stateless turns. Without it, the model re-reasons from scratch. | HMAC-like token that **authenticates** the plaintext `thinking` field. The text itself rides back too. With `display: "omitted"`, the text is replaced server-side from the signature. | Opaque, "encrypted representation of the model's internal thought process" — also **carries** state, not just verifies it. Most analogous to OpenAI's `encrypted_content`. |

### Round-trip rule (universal)

Whatever came out of the prior assistant turn's reasoning container goes back into the next request **verbatim** — same ordering, same bytes, same continuity tokens. The common bug across all three vendors is identical: a tool-runner that rebuilds the next turn from just the tool-call + tool-result, dropping the reasoning block that preceded them.

### OpenInference attribute mapping

| Provider field | OpenInference attribute | Notes |
|---|---|---|
| OpenAI `ResponseReasoningItem.id` | `message_content.id` | Provider id for the reasoning item; preserve for stateless replay. |
| OpenAI `encrypted_content` | `message_content.encrypted_content` | Opaque continuity token; populated only when `include: ["reasoning.encrypted_content"]` is requested. |
| Anthropic `thinking.signature` | `message_content.signature` | Authenticates the accompanying `thinking` text; echo both together. |
| Anthropic `redacted_thinking.data` | `message_content.data` | Redacted continuity payload; emit without `message_content.text`. |
| Gemini `thoughtSignature` on a text part | `message_content.signature` | Signature belongs to the non-tool content part. |
| Gemini `thoughtSignature` on a `functionCall` part | `tool_call.reasoning_signature` | Signature belongs to the tool call, not to the reasoning summary. |

---

## 4. Quick reference: capturing for telemetry

| Aspect | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| Intent (request) | `reasoning.effort`, `reasoning.summary`, `include[]` contains `reasoning.encrypted_content` | `thinking.type`, `thinking.budget_tokens`, `thinking.display` | `thinkingConfig.thinkingBudget` / `thinkingLevel`, `includeThoughts` |
| Arrival (response) | count of `output[].type == "reasoning"`; presence of `summary[]` vs `content[]` | count of `content[].type == "thinking"` and `"redacted_thinking"`; presence of `signature` | count of `parts[].thought == true`; count of `parts[].thoughtSignature` (and which kinds of parts carry it) |
| Continuity token | presence + length of `encrypted_content` | presence + length of `signature` / `data` | presence + length of `thoughtSignature` per part |
| Tokens | `output_tokens_details.reasoning_tokens` | (no separate field; included in `output_tokens`) | `usageMetadata.thoughtsTokenCount` |

Never log the continuity token's payload bytes — they're large, opaque, base64-encoded, and not human-readable. Log presence, length, and the discriminator that produced them.
