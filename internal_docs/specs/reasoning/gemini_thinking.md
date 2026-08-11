# Gemini API: thinking / reasoning surface

Practical JSON examples for every shape involved in Google's "thinking" (reasoning) surface on `POST /v1beta/models/{model}:generateContent` (or `client.models.generate_content(...)` in the Python SDK). The shapes follow the Pydantic / TypedDict names under `google/genai/types.py`, which mirror the HTTP schema. Use a section's example as a starting point and drop fields you do not need.

Permalinks below use commit `6fa256fae20738483a05d190736c0bfbafd06441` from [googleapis/python-genai](https://github.com/googleapis/python-genai) (v2.2.0). Behavior claims are cross-checked against the [thinking](https://ai.google.dev/gemini-api/docs/thinking), [thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures), and [function-calling](https://ai.google.dev/gemini-api/docs/function-calling) docs pages.

---

## 1. Definitions

**Thinking** — the model produces internal reasoning before its visible answer. Three surfaces: **request config** (`ThinkingConfig`) — opt in, set a budget or level, opt into summaries; **response parts** (`Part` objects with `thought: true` or with a `thoughtSignature`) inside `candidates[].content.parts`; **streaming chunks** that arrive as partial `GenerateContentResponse` envelopes carrying the same `parts[]` shape.

**Thought signature** — an opaque, base64-encoded byte blob (`Part.thought_signature`) that encodes the model's internal reasoning state. Unlike Anthropic's `signature` (an HMAC authenticating plaintext thinking), Gemini's signature is described as an **"encrypted representation of the model's internal thought process"** — i.e. it *carries* state, not just verifies it. Signature pass-through is **load-bearing**: when building a follow-up turn you must echo the originating `Part` back verbatim (with its `thoughtSignature` byte-for-byte). On Gemini 3 with tool use the API returns HTTP 400 (`Function call ... is missing a thought_signature.`) when a required signature is dropped.

**Thinking budget** — `thinking_budget` (Gemini 2.5) is the model's reasoning budget in tokens. Special values: `-1` = dynamic (model self-selects), `0` = disabled (Flash / Flash-Lite only). Soft target, not a hard ceiling — "the model might overflow or underflow the token budget."

**Thinking level** — `thinking_level` (Gemini 3) is the categorical replacement for `thinking_budget`: `"minimal" | "low" | "medium" | "high"`. Using `thinking_budget` on Gemini 3 Pro "may result in unexpected performance."

**Thought summaries** — when `include_thoughts: true`, the response interleaves additional `Part`s with `thought: true` and a `text` field containing a human-readable summary of the model's reasoning. The full thinking is **not** returned; only a summary. Thinking tokens are still billed in full.

**Full request-config shape** (Pydantic model, three optional fields):

[source](https://github.com/googleapis/python-genai/blob/6fa256fae20738483a05d190736c0bfbafd06441/google/genai/types.py#L5333-L5349)

```python
class ThinkingConfig(_common.BaseModel):
    include_thoughts: Optional[bool]
    thinking_budget: Optional[int]      # Gemini 2.5: tokens (-1 = dynamic, 0 = disabled)
    thinking_level: Optional[ThinkingLevel]  # Gemini 3: minimal | low | medium | high
```

**Full response Part shape** (thinking-related fields highlighted):

[source: Part schema](https://ai.google.dev/api/caching#Part) · [source: Pydantic Part](https://github.com/googleapis/python-genai/blob/6fa256fae20738483a05d190736c0bfbafd06441/google/genai/types.py#L1967-L1974)

```
Part {
  text:                string                  # one-of: data union
  inlineData:          Blob                    # one-of
  functionCall:        FunctionCall            # one-of
  functionResponse:    FunctionResponse        # one-of
  fileData:            FileData                # one-of
  executableCode:      ExecutableCode          # one-of
  codeExecutionResult: CodeExecutionResult     # one-of
  videoMetadata:       VideoMetadata           # optional
  thought:             bool                    # <-- "this part is a thought summary"
  thoughtSignature:    bytes (base64 on wire)  # <-- opaque reasoning-state token
}
```

`thought` and `thoughtSignature` are **siblings** of the data union — any data-bearing part (text, functionCall, etc.) can also carry a signature.

---

## 2. Input Params

Shapes sent on the **request** side of `generateContent`.

### `ThinkingConfig` — Gemini 2.5 token budget

[source](https://github.com/googleapis/python-genai/blob/6fa256fae20738483a05d190736c0bfbafd06441/google/genai/types.py#L5333-L5349)

Turn thinking on with an explicit budget and request summaries:

```json
{
  "generationConfig": {
    "thinkingConfig": {
      "thinkingBudget": 8000,
      "includeThoughts": true
    }
  }
}
```

Dynamic — let the model self-select (default on 2.5 Pro / 2.5 Flash):

```json
{
  "generationConfig": {
    "thinkingConfig": { "thinkingBudget": -1 }
  }
}
```

Disable thinking (2.5 Flash and 2.5 Flash-Lite only — **not supported on 2.5 Pro**):

```json
{
  "generationConfig": {
    "thinkingConfig": { "thinkingBudget": 0 }
  }
}
```

**Insight:** `thinking_budget` is a **soft target**, not a hard cap — docs note the model may over- or under-shoot. The valid range is **per-model** (2.5 Pro: 128–32768; 2.5 Flash: 0–24576; 2.5 Flash-Lite: 512–24576). Pass `-1` for dynamic on mixed workloads; pass `0` to skip thinking entirely on Flash where latency matters. **`thoughtsTokenCount` is billed at the output rate** whether summaries are surfaced or not (`include_thoughts` controls visibility only, not generation).

---

### `ThinkingConfig` — Gemini 3 level

[source: ThinkingLevel enum](https://github.com/googleapis/python-genai/blob/6fa256fae20738483a05d190736c0bfbafd06441/google/genai/types.py#L317-L328)

Gemini 3 replaces the integer budget with a categorical level. Values: `"minimal"`, `"low"`, `"medium"`, `"high"`.

```json
{
  "generationConfig": {
    "thinkingConfig": {
      "thinkingLevel": "high",
      "includeThoughts": true
    }
  }
}
```

Minimal (the "no thinking for most queries" setting on Gemini 3 Flash / Flash-Lite — **not supported on Gemini 3.1 Pro**, which cannot disable thinking):

```json
{
  "generationConfig": {
    "thinkingConfig": { "thinkingLevel": "minimal" }
  }
}
```

**Insight:** Gemini 3 expects `thinkingLevel`; passing `thinkingBudget` is documented as producing "unexpected performance." Defaults differ across the 3.x family — Gemini 3.1 Pro and Gemini 3 Flash default to `"high"` (dynamic); Gemini 3.1 Flash-Lite defaults to `"minimal"`. Verify defaults against the [model support table](https://ai.google.dev/gemini-api/docs/thinking) before relying on them.

---

### `Part` (multi-turn pass-through with signature)

[source](https://github.com/googleapis/python-genai/blob/6fa256fae20738483a05d190736c0bfbafd06441/google/genai/types.py#L1967-L1974)

To continue a conversation that used thinking, echo the prior model turn's parts **with their `thoughtSignature` intact** in `contents[]`. The param shape is field-for-field identical to the response shape.

```json
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
}
```

**Insight:** Signatures stay **glued to their originating Part** — do not move, merge, or concatenate. From the docs: *"Don't concatenate parts with signatures together. Don't merge one part with a signature with another part without a signature."* For **parallel function calls**, only the **first** `functionCall` part carries a signature; subsequent ones do not. For **sequential function calls** across turns, every signature ever returned must be echoed back. Plain text/`inlineData` signatures are recommended but not strictly validated; `functionCall` signatures on Gemini 3 **are validated** and return HTTP 400 if omitted.

---

## 3. Output Params

Shapes returned by `generateContent`, either in `candidates[].content.parts` (non-streaming) or as streamed chunks.

### `Part` with `thought: true` (thought summary)

[source: Part.thought](https://github.com/googleapis/python-genai/blob/6fa256fae20738483a05d190736c0bfbafd06441/google/genai/types.py#L1967-L1970)

The model's summarized reasoning text. Appears in `candidates[].content.parts` **ahead of** the final answer part(s). Only present when the request set `includeThoughts: true`.

```json
{
  "thought": true,
  "text": "The user asked for the capital of France. That's Paris. I should answer concisely."
}
```

Full response mixing a thought summary and the answer:

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [
          {
            "thought": true,
            "text": "The user asked for the capital of France. That's Paris. I should answer concisely."
          },
          { "text": "Paris." }
        ]
      },
      "finishReason": "STOP",
      "index": 0
    }
  ],
  "usageMetadata": {
    "promptTokenCount": 12,
    "candidatesTokenCount": 2,
    "thoughtsTokenCount": 87,
    "totalTokenCount": 101
  },
  "modelVersion": "gemini-2.5-flash"
}
```

**Insight:** Thought-summary parts are **summaries, not the raw reasoning** — `thoughtsTokenCount` reflects the full thinking that was generated, not the short summary text you see. Treat summary `text` like model output for privacy review — it can quote the user verbatim. The Python SDK's `response.text` convenience accessor **skips** parts where `thought == true` ([source](https://github.com/googleapis/python-genai/blob/6fa256fae20738483a05d190736c0bfbafd06441/google/genai/types.py#L7987-L7997)), so iterate `parts` explicitly when you need the summary.

---

### `Part` with `thoughtSignature` (multi-turn token)

[source: Part.thought_signature](https://github.com/googleapis/python-genai/blob/6fa256fae20738483a05d190736c0bfbafd06441/google/genai/types.py#L1971-L1974)

An opaque, base64-encoded byte blob attached to a data-bearing part. Most commonly seen on `functionCall` parts (Gemini 2.5 + tools) or on any part on Gemini 3.

```json
{
  "functionCall": {
    "name": "check_flight",
    "args": { "flight": "AA100" }
  },
  "thoughtSignature": "EvgBCoYBGAIqQDLkP..."
}
```

On a text part (Gemini 3):

```json
{
  "text": "The flight is on time.",
  "thoughtSignature": "CiQB..."
}
```

**Insight:** `thoughtSignature` is **opaque and load-bearing** — must travel back verbatim in the next request when you include this turn in `contents[]`. Do not log the bytes as content (large, base64, not human-readable); do log presence and length to debug "signature lost" issues. **Signatures do NOT attach to `thought: true` summary parts** — they ride on the answer-bearing parts (functionCall, text, etc.). The Python SDK exposes the field as `bytes`; over the wire it is a base64-encoded `string`.

---

### `usageMetadata.thoughtsTokenCount`

[source: REST reference](https://ai.google.dev/api/generate-content#UsageMetadata)

Generated thinking tokens are reported separately but **billed at the output rate**:

```json
{
  "usageMetadata": {
    "promptTokenCount": 14,
    "candidatesTokenCount": 32,
    "thoughtsTokenCount": 412,
    "totalTokenCount": 458,
    "promptTokensDetails": [{ "modality": "TEXT", "tokenCount": 14 }]
  }
}
```

**Insight:** `totalTokenCount = promptTokenCount + candidatesTokenCount + toolUsePromptTokenCount + thoughtsTokenCount`. Pricing page lists output as "Output price (including thinking tokens)" — there is no separate thinking-token price tier. Echoing a `thoughtSignature` back on a subsequent turn increases **input** token count, billed at the input rate.

---

### Streaming chunks

The streaming API emits partial `GenerateContentResponse` envelopes with the same `candidates[].content.parts[]` shape — there is no equivalent of Anthropic's `content_block_start` / `content_block_delta` / `content_block_stop` envelope. Each SSE chunk is a self-describing partial response.

```
data: {"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":"Let me consider"}]},"index":0}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":" the user's actual question."}]},"index":0}]}

data: {"candidates":[{"content":{"role":"model","parts":[{"text":"The answer is 42."}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":87,"totalTokenCount":102}}
```

**Insight:** Thought-summary chunks arrive **incrementally** during generation ("rolling, incremental summaries"); concatenate `text` across consecutive chunks where `thought: true`. Signatures may arrive on a **part with empty text** in streaming mode — per the docs: *"the model may return the thought signature in a part with an empty text content part. It is advisable to parse the entire request until the finish_reason is returned."*

---

## 4. Examples

### Complete `generate_content` request body

Full request wiring a thinking config, a user turn, and an echoed prior model turn carrying a signature:

```json
{
  "contents": [
    { "role": "user", "parts": [{ "text": "What's the weather in Paris?" }] },
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
    "thinkingConfig": {
      "thinkingBudget": 8000,
      "includeThoughts": true
    }
  },
  "tools": [{ "functionDeclarations": [ /* ... */ ] }]
}
```

Python SDK equivalent:

```python
from google import genai
from google.genai import types

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents=[...],
    config=types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=8000,
            include_thoughts=True,
        ),
        tools=[...],
    ),
)
```

**Insight:** The prior model turn must carry its `thoughtSignature` verbatim — for Gemini 3 + tool calls the API returns HTTP 400 (`Function call ... in the ... content block is missing a thought_signature.`) when it's dropped. For Gemini 2.5, dropping signatures degrades quality but does not error. **The official SDKs handle signature pass-through automatically in chat mode** — you only manage signatures manually when editing conversation history or calling REST directly.

---

### OpenAI-compat shim

For users hitting Gemini via the OpenAI-compatible endpoint, signatures travel under `extra_content.google.thought_signature`:

```json
{
  "tool_calls": [
    {
      "function": {
        "name": "check_flight",
        "arguments": "{\"flight\":\"AA100\"}"
      },
      "extra_content": {
        "google": { "thought_signature": "CiQB..." }
      }
    }
  ]
}
```

[source: OpenAI compat docs](https://ai.google.dev/gemini-api/docs/openai#thinking)

---

### Function-calling: parallel and sequential calls

**Parallel** — only the first `functionCall` part in a single response carries a signature:

```json
{
  "role": "model",
  "parts": [
    {
      "functionCall": { "name": "get_weather", "args": { "city": "Paris" } },
      "thoughtSignature": "CiQB..."
    },
    {
      "functionCall": { "name": "get_weather", "args": { "city": "Tokyo" } }
    }
  ]
}
```

**Sequential** — every signature returned across turns must be echoed back in the next request; omitting any triggers Gemini 3 validation failure.

**Insight:** A common bug pattern: a tool-runner that rebuilds the model turn from only the `functionCall` block, dropping the `thoughtSignature` riding on the same part. On Gemini 2.5 this silently degrades reasoning; on Gemini 3 it returns HTTP 400. There is no equivalent of Anthropic's `tool_choice` restriction — Gemini accepts any function-calling mode while thinking is on.

---

### Model support matrix

| Model | Config knob | Default | Range | Disable | Dynamic |
|---|---|---|---|---|---|
| Gemini 2.5 Pro | `thinkingBudget` | dynamic | 128–32768 | **No** | `-1` (default) |
| Gemini 2.5 Flash | `thinkingBudget` | dynamic | 0–24576 | `0` | `-1` (default) |
| Gemini 2.5 Flash-Lite | `thinkingBudget` | off | 512–24576 | `0` | `-1` |
| Gemini 3.1 Pro | `thinkingLevel` | `high` | low / medium / high | **No** | `high` (default) |
| Gemini 3 Flash | `thinkingLevel` | `high` | minimal / low / medium / high | `minimal` | `high` (default) |
| Gemini 3.1 Flash-Lite | `thinkingLevel` | `minimal` | minimal / low / medium / high | `minimal` (default) | `high` |

**Insight:** Check the [official model support table](https://ai.google.dev/gemini-api/docs/thinking) before deploying — defaults and ranges shift between model revisions. For mixed-model code paths, prefer feeding a `ThinkingConfig` shape that includes only the right field family (`thinkingBudget` for 2.5; `thinkingLevel` for 3.x) rather than passing both.

---

## 5. Summary

### Discriminating fields

Three signals do the heavy lifting in any instrumentation: (1) request `thinkingConfig.thinkingBudget` / `thinkingLevel` tells you the **intent**; (2) response `parts[].thought == true` tells you what **arrived** as a summary; (3) response `parts[].thoughtSignature` tells you what to **echo back** on the originating Part. Capture all three and you can faithfully reconstruct reasoning behavior across turns without ever decoding the opaque signature bytes.

| Surface | Field | Python class | Notes |
|---|---|---|---|
| Request config | `thinkingBudget` | `ThinkingConfig.thinking_budget: int` | Gemini 2.5; `-1` dynamic, `0` disabled |
| Request config | `thinkingLevel` | `ThinkingConfig.thinking_level: ThinkingLevel` | Gemini 3; `minimal`/`low`/`medium`/`high` |
| Request config | `includeThoughts` | `ThinkingConfig.include_thoughts: bool` | Surfaces thought-summary parts |
| Request pass-through | `thoughtSignature` | `Part.thought_signature: bytes` | Echo verbatim on originating Part |
| Response part | `thought` | `Part.thought: bool` | `true` = summary part |
| Response part | `thoughtSignature` | `Part.thought_signature: bytes` | Opaque encrypted reasoning state |
| Usage | `thoughtsTokenCount` | `UsageMetadata.thoughts_token_count: int` | Billed at output rate |

---

### See also (vendor docs)

Official behavior, model availability, and pricing change over time; confirm in the links below.

- [Thinking guide](https://ai.google.dev/gemini-api/docs/thinking) — high-level usage, budgets, levels, per-model support matrix.
- [Thought signatures](https://ai.google.dev/gemini-api/docs/thought-signatures) — multi-turn pass-through rules and validation behavior.
- [Function calling](https://ai.google.dev/gemini-api/docs/function-calling) — required ordering and signature pass-through in agent loops; signature dummy-value escape hatches.
- [generateContent reference](https://ai.google.dev/api/generate-content) — request body fields including `generationConfig.thinkingConfig` and `usageMetadata.thoughtsTokenCount`.
- [Part schema reference](https://ai.google.dev/api/caching#Part) — `thought` and `thoughtSignature` field definitions.
- [OpenAI compat: thinking](https://ai.google.dev/gemini-api/docs/openai#thinking) — `extra_content.google.thought_signature` shape.
- [Vertex AI thinking docs](https://cloud.google.com/vertex-ai/generative-ai/docs/thinking) — Vertex variant with same semantics.
- [Pricing](https://ai.google.dev/pricing) — output rates "including thinking tokens" per model.
