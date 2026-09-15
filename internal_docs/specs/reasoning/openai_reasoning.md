# OpenAI Responses API: reasoning item variants

Practical JSON examples for every shape involved in OpenAI's reasoning surface on `POST /v1/responses` (or `client.responses.create(...)` in the Python SDK), plus the much narrower `reasoning_effort` knob on `POST /v1/chat/completions`. The shapes follow the `BaseModel` / `TypedDict` names under `openai/types/`, which mirror the HTTP schema. Use a section's example as a starting point and drop fields you do not need.

Permalinks below use commit `38d75d74a5626472cd7d1be9705ea8aba29a6b22` from [openai/openai-python](https://github.com/openai/openai-python) (v2.36.0). Behavior claims are cross-checked against the [reasoning guide](https://developers.openai.com/api/docs/guides/reasoning) and the [Responses API reference](https://platform.openai.com/docs/api-reference/responses).

---

## 1. Definitions

**Reasoning** — the model produces internal chain-of-thought before its visible answer. Three surfaces: **request config** (`Reasoning` object with `effort` + `summary`) — opt into a level and choose summary verbosity; **response items** (`ResponseReasoningItem`) inside `Response.output[]`, carrying `summary[]`, optional `content[]`, and an opaque `encrypted_content`; **streaming events** (`response.reasoning_summary_*` and `response.reasoning_text.*`) inside the SSE stream during `stream=true`.

**Encrypted content** — an opaque ciphertext blob attached to each reasoning item when the request opts into `include: ["reasoning.encrypted_content"]`. It is the **only load-bearing field** for multi-turn continuity on the stateless / ZDR path (`store: false`). The plain-text `summary` is informational; without `encrypted_content` the next turn re-reasons from scratch (and the server rejects requests that drop a reasoning item which immediately preceded a `function_call`).

**Summary parts** — human-readable digests of the model's reasoning, each `{type: "summary_text", text: ...}`. Verbosity controlled by `summary: "auto" | "concise" | "detailed"`. May be empty (`[]`) for `effort: "minimal"` or models that emit no summaries.

**Reasoning text parts** — `{type: "reasoning_text", text: ...}` content parts carrying raw chain-of-thought. Gated; most production reasoning models emit summaries only.

**Effort** — the model's reasoning budget. Literal: `none | minimal | low | medium | high | xhigh`. Defaults and accepted values are **model-dependent** (see capability table below).

**Stateful vs stateless paths** — two ways to carry reasoning across turns. **Stateful (default):** `store: true`, send `previous_response_id` on the next turn, server replays reasoning internally — you do not echo reasoning items. **Stateless:** `store: false` (or ZDR org), opt into `include: ["reasoning.encrypted_content"]`, and echo every prior `output[]` item (reasoning + function_call + function_call_output) back in `input[]`.

**Full request-config object** (single class, no union):

[source: response](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/shared/reasoning.py#L12-L52)
[source: request](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/shared_params/reasoning.py#L13-L53)

```python
class Reasoning:
    effort: Optional[ReasoningEffort]         # "none"|"minimal"|"low"|"medium"|"high"|"xhigh"
    summary: Optional[Literal["auto", "concise", "detailed"]]
    generate_summary: Optional[Literal["auto", "concise", "detailed"]]  # deprecated alias of summary
```

**Full response output-item union** (reasoning-related member highlighted):

[source](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_output_item.py#L225-L254)

```python
ResponseOutputItem = Annotated[
    Union[
        ResponseOutputMessage,
        ResponseFileSearchToolCall,
        ResponseFunctionToolCall,
        ResponseFunctionWebSearch,
        ResponseComputerToolCall,
        ResponseReasoningItem,        # type: "reasoning"
        ResponseCodeInterpreterToolCall,
        # ...other tool-call variants
    ],
    PropertyInfo(discriminator="type"),
]
```

**Full streaming event union** (reasoning-related members highlighted):

[source](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_stream_event.py)

```
ResponseStreamEvent =
    | response.output_item.added | response.output_item.done            # envelope
    | response.content_part.added | response.content_part.done          # envelope
    | response.reasoning_summary_part.added | .done                     # reasoning
    | response.reasoning_summary_text.delta | .done                     # reasoning
    | response.reasoning_text.delta | .done                             # reasoning
    | response.output_text.delta | .done                                # message
    | ...function_call / file_search / tool-call events...
```

---

## 2. Input Params

Shapes sent on the **request** side of `POST /v1/responses`.

### `Reasoning` (request config)

[source](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/shared_params/reasoning.py#L13-L53)

Turn reasoning on with an effort level and (optionally) a summary verbosity. Only `gpt-5` / `o-series` / `computer-use-preview` models accept this object.

```json
{
  "model": "gpt-5",
  "input": "Plan a 3-day Tokyo trip.",
  "reasoning": {
    "effort": "medium",
    "summary": "auto"
  }
}
```

Minimal:

```json
{
  "model": "o4-mini",
  "input": "Solve x^2 - 4 = 0.",
  "reasoning": { "effort": "low" }
}
```

Opt into raw / persisted chain-of-thought:

```json
{
  "model": "gpt-5",
  "input": "...",
  "reasoning": { "effort": "high", "summary": "detailed" },
  "include": ["reasoning.encrypted_content"],
  "store": false
}
```

**Insight:** `effort` defaults are **model-specific**. `gpt-5.1` defaults to `"none"`; pre-`gpt-5.1` reasoning models default to `"medium"`. `gpt-5-pro` accepts **only** `"high"`. `"minimal"` was introduced with the `gpt-5` family; `"xhigh"` only on `gpt-5.1-codex-max` and later. `summary` is Responses-API-only — Chat Completions has no equivalent. `generate_summary` is a deprecated alias for `summary`; do not use it in new code. The reasoning object itself is silently ignored on non-reasoning models, so guard it at config time rather than trusting the type system.

---

### `reasoning_effort` (Chat Completions only)

[source](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/chat/completion_create_params.py#L196-L211)

Chat Completions exposes **only the effort knob** — no summary, no encrypted-content opt-in, no reasoning items in the response. Field name is `reasoning_effort` (top-level, snake_case), same literal enum as `Reasoning.effort`.

```json
{
  "model": "o3-mini",
  "messages": [{"role": "user", "content": "Solve x^2 - 4 = 0."}],
  "reasoning_effort": "low",
  "max_completion_tokens": 4096
}
```

**Insight:** On Chat Completions for reasoning models, use **`max_completion_tokens`**, not `max_tokens` (the SDK explicitly notes `max_tokens` "is not compatible with o-series models"). The only reasoning signal you get back is `usage.completion_tokens_details.reasoning_tokens`. To capture reasoning summaries or carry chain-of-thought across turns, you must use the Responses API.

---

### `include: ["reasoning.encrypted_content"]`

[source: includable enum](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_includable.py#L7-L16)
[source: param docstring](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_create_params.py#L55-L75)

Opt into receiving the opaque `encrypted_content` blob on every `ResponseReasoningItem` in the output. Required when `store: false` (stateless) or for ZDR-enrolled orgs that want to carry reasoning across turns.

```json
{
  "model": "gpt-5",
  "input": "What's the weather in Paris?",
  "reasoning": { "effort": "medium" },
  "store": false,
  "include": ["reasoning.encrypted_content"]
}
```

**Insight:** Without this include, `encrypted_content` is `null` on every reasoning item, and your only options for multi-turn are `previous_response_id` (which requires `store: true`) or losing the chain-of-thought. The literal string is **exactly** `"reasoning.encrypted_content"` — typos fail silently (the include is dropped).

---

### `ResponseReasoningItemParam` (multi-turn echo)

[source](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_reasoning_item_param.py#L31-L62)

To continue a stateless conversation that used reasoning, include the prior assistant turn's reasoning item(s) **with `encrypted_content` intact** in `input[]`. The param shape is field-for-field identical to the output shape.

```json
{
  "type": "reasoning",
  "id": "rs_abc123",
  "summary": [
    { "type": "summary_text", "text": "Need to call get_weather for Paris." }
  ],
  "encrypted_content": "gAAAAA...REDACTED_OPAQUE_PAYLOAD...=="
}
```

**Insight:** `summary[]` is human-readable and lossy — `encrypted_content` is the load-bearing state. The most common bug pattern: an agent loop that rebuilds the next-turn input from just the `function_call` and `function_call_output`, silently dropping the leading reasoning item. When reasoning immediately precedes a `function_call`, the server rejects the next turn if the reasoning item is omitted. **Round-trip rule:** echo every reasoning + function_call + function_call_output item between the last user message and the assistant turn you are continuing.

---

## 3. Output Params

Shapes returned by `POST /v1/responses`, either in `Response.output[]` (non-streaming) or as streaming events.

### `ResponseReasoningItem`

[source](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_reasoning_item.py#L31-L62)

The reasoning step the model took before (or between) its visible output. Appears in `Response.output[]` **before** the `message` or `function_call` items it informs.

```json
{
  "id": "rs_abc123",
  "type": "reasoning",
  "summary": [
    { "type": "summary_text", "text": "User asked for the capital of France. That's Paris." }
  ],
  "content": null,
  "encrypted_content": "gAAAAA...REDACTED...==",
  "status": "completed"
}
```

Full response with reasoning + message:

```json
{
  "id": "resp_01ABC",
  "object": "response",
  "model": "gpt-5",
  "output": [
    {
      "id": "rs_1",
      "type": "reasoning",
      "summary": [
        { "type": "summary_text", "text": "User asked for the capital of France. That's Paris." }
      ],
      "content": null,
      "encrypted_content": null,
      "status": "completed"
    },
    {
      "id": "msg_1",
      "type": "message",
      "role": "assistant",
      "status": "completed",
      "content": [{ "type": "output_text", "text": "Paris." }]
    }
  ],
  "usage": {
    "input_tokens": 12,
    "input_tokens_details": { "cached_tokens": 0 },
    "output_tokens": 33,
    "output_tokens_details": { "reasoning_tokens": 24 },
    "total_tokens": 45
  }
}
```

**Insight:** `summary[]` may be empty for `effort: "minimal"` or models that don't emit summaries — the reasoning item is still produced as a discrete output slot (so ordering for multi-turn echo is preserved). `content[]` is usually `null` or `[]`; raw `reasoning_text` parts only surface on models / orgs gated for it. `encrypted_content` is populated **only** when `include: ["reasoning.encrypted_content"]` is set on the request. Treat reasoning summaries like model output for privacy review — they can quote the user verbatim.

---

### `Summary` (summary part)

[source](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_reasoning_item.py#L11-L17)

```json
{ "type": "summary_text", "text": "User asked for the capital of France. That's Paris." }
```

**Insight:** Single discriminated variant today (`summary_text`). The same shape is redeclared inline inside each `reasoning_summary_part.*` streaming event — they are structurally identical but separate Python classes.

---

### `Content` (raw reasoning text part)

[source](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_reasoning_item.py#L21-L28)

```json
{ "type": "reasoning_text", "text": "Let me think step by step. First..." }
```

**Insight:** Most production reasoning models do not surface raw `reasoning_text` parts — `content` arrives as `null` or `[]`. When it is emitted, treat it like full chain-of-thought (more sensitive than the summary).

---

### `status` semantics

[source](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_reasoning_item.py#L57-L62)

- `"in_progress"` — reasoning item is mid-stream. Seen on `output_item.added` events and on retrieved in-flight responses. `summary[]` and `content[]` may be partial; `encrypted_content` not yet populated.
- `"completed"` — finished normally. Echo this item on subsequent stateless turns.
- `"incomplete"` — truncated (e.g. `max_output_tokens` hit, content filter, upstream error). Partial summaries may be present; still echo it back.

---

### Streaming events — block boundaries

[source: stream-event union](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_stream_event.py)
[source: output_item.added](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_output_item_added_event.py)
[source: output_item.done](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_output_item_done_event.py)

Each reasoning item is bounded by `response.output_item.added` → summary / content sub-events → `response.output_item.done`. Use `output_index` to identify the item across events.

```json
{
  "type": "response.output_item.added",
  "sequence_number": 3,
  "output_index": 0,
  "item": {
    "id": "rs_abc123",
    "type": "reasoning",
    "summary": [],
    "content": [],
    "encrypted_content": null,
    "status": "in_progress"
  }
}
```

```json
{
  "type": "response.output_item.done",
  "sequence_number": 90,
  "output_index": 0,
  "item": {
    "id": "rs_abc123",
    "type": "reasoning",
    "summary": [
      { "type": "summary_text", "text": "User asked for the capital of France. That's Paris." }
    ],
    "content": null,
    "encrypted_content": "gAAAAA...==",
    "status": "completed"
  }
}
```

**Insight:** `encrypted_content` arrives **atomically on `output_item.done`** — there is no `encrypted_content.delta` event. The initial `output_item.added` always carries `encrypted_content: null`.

---

### `ResponseReasoningSummaryPart*` (streaming)

[source: added](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_reasoning_summary_part_added_event.py#L10-L39)
[source: done](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_reasoning_summary_part_done_event.py#L10-L39)

A new `summary_text` part opens / closes. Identified by `(item_id, summary_index)`.

```json
{
  "type": "response.reasoning_summary_part.added",
  "sequence_number": 4,
  "item_id": "rs_abc123",
  "output_index": 0,
  "summary_index": 0,
  "part": { "type": "summary_text", "text": "" }
}
```

```json
{
  "type": "response.reasoning_summary_part.done",
  "sequence_number": 28,
  "item_id": "rs_abc123",
  "output_index": 0,
  "summary_index": 0,
  "part": {
    "type": "summary_text",
    "text": "User asked for the capital of France. That's Paris."
  }
}
```

---

### `ResponseReasoningSummaryText*` (streaming)

[source: delta](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_reasoning_summary_text_delta_event.py#L10-L29)
[source: done](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_reasoning_summary_text_done_event.py#L10-L29)

Token-level deltas for the active summary part. Scoped by `summary_index` (not `content_index`). The `.done` event carries the full final text.

```json
{
  "type": "response.reasoning_summary_text.delta",
  "sequence_number": 5,
  "item_id": "rs_abc123",
  "output_index": 0,
  "summary_index": 0,
  "delta": "User asked for the "
}
```

```json
{
  "type": "response.reasoning_summary_text.done",
  "sequence_number": 27,
  "item_id": "rs_abc123",
  "output_index": 0,
  "summary_index": 0,
  "text": "User asked for the capital of France. That's Paris."
}
```

**Insight:** Concatenate `delta` strings by `(item_id, summary_index)` in arrival order, exactly as you would `output_text.delta`. If `summary[]` ends up empty (effort `"minimal"`, no-summary model), these events are simply absent — the reasoning item still opens and closes via `output_item.added/.done`.

---

### `ResponseReasoningText*` (streaming, gated)

[source: delta](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_reasoning_text_delta_event.py#L10-L29)
[source: done](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_reasoning_text_done_event.py#L10-L29)

Token-level deltas for raw `reasoning_text` content parts. Scoped by `content_index` (the index inside `item.content[]`, **not** `summary_index`). Bracketed by `response.content_part.added` / `.done` envelopes.

```json
{
  "type": "response.reasoning_text.delta",
  "sequence_number": 30,
  "item_id": "rs_abc123",
  "output_index": 0,
  "content_index": 0,
  "delta": "Let me think step by step. "
}
```

```json
{
  "type": "response.reasoning_text.done",
  "sequence_number": 88,
  "item_id": "rs_abc123",
  "output_index": 0,
  "content_index": 0,
  "text": "Let me think step by step. First..."
}
```

**Insight:** Most reasoning models never emit this cycle — only summaries surface. Treat its presence as the discriminator for "this org / model is allowed to receive raw chain-of-thought."

---

## 4. Examples

### Complete `responses.create()` request body

A full stateful request wiring reasoning config, prior-response continuity, and a user follow-up:

```json
{
  "model": "gpt-5",
  "input": "And what is the capital of Germany?",
  "reasoning": {
    "effort": "medium",
    "summary": "auto"
  },
  "previous_response_id": "resp_01ABC",
  "store": true
}
```

Stateless / ZDR variant (echo prior output, include `encrypted_content`):

```json
{
  "model": "gpt-5",
  "store": false,
  "include": ["reasoning.encrypted_content"],
  "reasoning": { "effort": "medium" },
  "input": [
    { "role": "user", "content": "What is the capital of France?" },
    {
      "type": "reasoning",
      "id": "rs_1",
      "summary": [
        { "type": "summary_text", "text": "User asked for the capital of France." }
      ],
      "encrypted_content": "gAAAAA...=="
    },
    {
      "type": "message",
      "id": "msg_1",
      "role": "assistant",
      "status": "completed",
      "content": [{ "type": "output_text", "text": "Paris." }]
    },
    { "role": "user", "content": "And what is the capital of Germany?" }
  ]
}
```

**Insight:** Pick exactly one continuity path per request — either `previous_response_id` **or** echoed `input[]`. Mixing them is a foot-gun: passing both makes the server prefer the prior response and silently ignore your echoed items. The reasoning + message + reasoning + message... chain reflects the canonical order each turn produces.

---

### Streaming: reasoning block with summary only (the common case)

Event sequence for a single reasoning item that emits one summary part (events in arrival order):

```json
{ "type": "response.output_item.added", "sequence_number": 3, "output_index": 0,
  "item": { "id": "rs_1", "type": "reasoning", "summary": [], "content": [],
            "encrypted_content": null, "status": "in_progress" } }
```

```json
{ "type": "response.reasoning_summary_part.added", "sequence_number": 4,
  "item_id": "rs_1", "output_index": 0, "summary_index": 0,
  "part": { "type": "summary_text", "text": "" } }
```

```json
{ "type": "response.reasoning_summary_text.delta", "sequence_number": 5,
  "item_id": "rs_1", "output_index": 0, "summary_index": 0,
  "delta": "User asked for the " }
```

```json
{ "type": "response.reasoning_summary_text.delta", "sequence_number": 6,
  "item_id": "rs_1", "output_index": 0, "summary_index": 0,
  "delta": "capital of France." }
```

```json
{ "type": "response.reasoning_summary_text.done", "sequence_number": 27,
  "item_id": "rs_1", "output_index": 0, "summary_index": 0,
  "text": "User asked for the capital of France." }
```

```json
{ "type": "response.reasoning_summary_part.done", "sequence_number": 28,
  "item_id": "rs_1", "output_index": 0, "summary_index": 0,
  "part": { "type": "summary_text", "text": "User asked for the capital of France." } }
```

```json
{ "type": "response.output_item.done", "sequence_number": 29, "output_index": 0,
  "item": { "id": "rs_1", "type": "reasoning",
            "summary": [{ "type": "summary_text", "text": "User asked for the capital of France." }],
            "content": null,
            "encrypted_content": "gAAAAA...==",
            "status": "completed" } }
```

---

### Streaming: reasoning block with raw `reasoning_text` (gated path)

When raw chain-of-thought is emitted, an additional `content_part` cycle runs after the summary cycle inside the same reasoning item:

```json
{ "type": "response.content_part.added", "sequence_number": 29,
  "item_id": "rs_1", "output_index": 0, "content_index": 0,
  "part": { "type": "reasoning_text", "text": "" } }
```

```json
{ "type": "response.reasoning_text.delta", "sequence_number": 30,
  "item_id": "rs_1", "output_index": 0, "content_index": 0,
  "delta": "Let me think step by step. " }
```

```json
{ "type": "response.reasoning_text.done", "sequence_number": 88,
  "item_id": "rs_1", "output_index": 0, "content_index": 0,
  "text": "Let me think step by step. First..." }
```

```json
{ "type": "response.content_part.done", "sequence_number": 89,
  "item_id": "rs_1", "output_index": 0, "content_index": 0,
  "part": { "type": "reasoning_text", "text": "Let me think step by step. First..." } }
```

---

### Streaming: empty-summary path (`effort: "minimal"`)

When the model emits no summary parts, the reasoning item still opens and closes:

```json
{ "type": "response.output_item.added", "sequence_number": 3, "output_index": 0,
  "item": { "id": "rs_1", "type": "reasoning", "summary": [], "content": [],
            "encrypted_content": null, "status": "in_progress" } }
```

```json
{ "type": "response.output_item.done", "sequence_number": 4, "output_index": 0,
  "item": { "id": "rs_1", "type": "reasoning", "summary": [], "content": null,
            "encrypted_content": "gAAAAA...==", "status": "completed" } }
```

**Insight:** There is **no dedicated "redacted" event** — redaction is signaled by the absence of summary events. `encrypted_content` is the only way to round-trip the hidden chain-of-thought on a subsequent stateless turn.

---

### Multi-turn with tool-use interleave

Reasoning + function_call + function_call_output must be echoed together for the next turn. Order is canonical: reasoning leads, function_call follows, function_call_output lives on the input side.

```json
{
  "model": "gpt-5",
  "store": false,
  "include": ["reasoning.encrypted_content"],
  "reasoning": { "effort": "medium" },
  "input": [
    { "role": "user", "content": "What's the weather in Paris?" },

    {
      "type": "reasoning",
      "id": "rs_1",
      "summary": [{ "type": "summary_text", "text": "Need to call get_weather for Paris." }],
      "encrypted_content": "gAAAAA...=="
    },
    {
      "type": "function_call",
      "id": "fc_1",
      "call_id": "call_1",
      "name": "get_weather",
      "arguments": "{\"city\":\"Paris\"}"
    },
    {
      "type": "function_call_output",
      "call_id": "call_1",
      "output": "{\"temp_c\":12,\"conditions\":\"cloudy\"}"
    }
  ],
  "tools": [{ "type": "function", "name": "get_weather", "parameters": { /* ... */ } }]
}
```

**Insight:** The bug pattern to watch for: a tool-runner that rebuilds the next turn from just `function_call` + `function_call_output`, dropping the leading reasoning item. The server rejects this — not silently. From the reasoning guide: *"if the model calls multiple functions consecutively, you should pass back all reasoning items, function call items, and function call output items, since the last `user` message."* On parallel tool calls, every branch's reasoning items must be echoed.

---

### Usage tokens

[source: Responses usage](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_usage.py)
[source: Chat usage](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/completion_usage.py)

Responses API:

```json
"usage": {
  "input_tokens": 412,
  "input_tokens_details": { "cached_tokens": 256 },
  "output_tokens": 980,
  "output_tokens_details": { "reasoning_tokens": 832 },
  "total_tokens": 1392
}
```

Chat Completions:

```json
"usage": {
  "prompt_tokens": 120,
  "completion_tokens": 845,
  "total_tokens": 965,
  "prompt_tokens_details": { "cached_tokens": 64, "audio_tokens": 0 },
  "completion_tokens_details": {
    "reasoning_tokens": 800,
    "audio_tokens": 0,
    "accepted_prediction_tokens": 0,
    "rejected_prediction_tokens": 0
  }
}
```

**Insight:** Reasoning tokens replayed via `previous_response_id` on the next stateful turn are billed under that turn's `input_tokens` (with portions counted as `input_tokens_details.cached_tokens` when the prompt cache hits) — there is no separate `cached_reasoning_tokens` field as of v2.36.0. On streaming Chat Completions, `usage` only appears on the final chunk when `stream_options.include_usage: true`.

---

### Model capability matrix

Source: [shared/reasoning.py docstring](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/shared/reasoning.py#L13-L34) + the [reasoning guide](https://platform.openai.com/docs/guides/reasoning).

| Model family | `effort` values | Default | `summary` (Responses API) |
|---|---|---|---|
| `o1`, `o1-preview`, `o1-mini` | `low`, `medium`, `high` | `medium` | unsupported on early o1; later o-series support `auto`/`detailed` |
| `o3`, `o3-mini`, `o4-mini` | `low`, `medium`, `high` | `medium` | `auto`, `detailed` |
| `gpt-5`, `gpt-5-mini`, `gpt-5-nano` | adds `minimal` | `medium` | adds `concise` |
| `gpt-5-pro` | only `high` | `high` | n/a |
| `gpt-5.1`, `gpt-5.1-mini` | `none`, `low`, `medium`, `high` | `none` | `auto`, `concise`, `detailed` |
| `gpt-5.1-codex-max` and later | adds `xhigh` | `none` | same |
| `computer-use-preview` | (reasoning model) | — | supports `concise` |

**Insight:** Defaults rotate per model. Pin `effort` explicitly when you have a budget assumption, or read `Reasoning.effort` back from a sample response to confirm what the server picked. Non-reasoning models silently ignore the `reasoning` object.

---

## 5. Summary

### Discriminating `type`

[source: includable enum](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_includable.py#L7-L16)
[source: output_item union](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_output_item.py#L225-L254)
[source: stream-event union](https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/responses/response_stream_event.py)

Three discriminators do the heavy lifting in any instrumentation: (1) request `reasoning.effort` and `reasoning.summary` tell you the **intent**, and `include[]` containing `"reasoning.encrypted_content"` flags the stateless path; (2) response `output[].type == "reasoning"` tells you what **arrived**, and `summary[].type` / `content[].type` discriminate `summary_text` vs `reasoning_text`; (3) streaming event `type` (`response.reasoning_summary_text.delta` vs `response.reasoning_text.delta`) tells you which buffer to **accumulate** at the active `(item_id, summary_index)` or `(item_id, content_index)`. Capture all three and you can faithfully reconstruct reasoning behavior across turns without ever decoding the opaque `encrypted_content` payload.

| Surface | `type` value | Python class | Required fields |
|---|---|---|---|
| Request config (Responses) | — | `Reasoning` | `effort`; optional `summary` |
| Request config (Chat) | — | top-level `reasoning_effort` kwarg | one of the `ReasoningEffort` literals |
| Request include flag | `reasoning.encrypted_content` | `ResponseIncludable` literal | — |
| Request pass-through | `reasoning` | `ResponseReasoningItemParam` | `id`, `type`, `summary`; `encrypted_content` (stateless) |
| Response item | `reasoning` | `ResponseReasoningItem` | `id`, `type`, `summary`; optional `content`, `encrypted_content`, `status` |
| Response sub-part | `summary_text` | `Summary` | `text` |
| Response sub-part | `reasoning_text` | `Content` | `text` |
| Streaming event | `response.output_item.added` / `.done` | `ResponseOutputItem*Event` | `output_index`, `item` |
| Streaming event | `response.reasoning_summary_part.added` / `.done` | `ResponseReasoningSummaryPart*Event` | `item_id`, `summary_index`, `part` |
| Streaming event | `response.reasoning_summary_text.delta` | `ResponseReasoningSummaryTextDeltaEvent` | `item_id`, `summary_index`, `delta` |
| Streaming event | `response.reasoning_summary_text.done` | `ResponseReasoningSummaryTextDoneEvent` | `item_id`, `summary_index`, `text` |
| Streaming event | `response.reasoning_text.delta` | `ResponseReasoningTextDeltaEvent` | `item_id`, `content_index`, `delta` |
| Streaming event | `response.reasoning_text.done` | `ResponseReasoningTextDoneEvent` | `item_id`, `content_index`, `text` |
| Usage (Responses) | — | `output_tokens_details.reasoning_tokens` | int |
| Usage (Chat) | — | `completion_tokens_details.reasoning_tokens` | int |

---

### See also (vendor docs)

Official behavior, model availability, and pricing change over time; confirm in the links below.

- [Reasoning guide (developer portal)](https://developers.openai.com/api/docs/guides/reasoning) — high-level usage, effort/summary, encrypted-content opt-in, per-model support matrix.
- [Reasoning guide (platform)](https://platform.openai.com/docs/guides/reasoning) — richer examples; same canonical content.
- [Conversation state](https://platform.openai.com/docs/guides/conversation-state) — stateful (`previous_response_id`) vs stateless (`store: false`) paths.
- [Function calling — reasoning interleave](https://developers.openai.com/api/docs/guides/function-calling) — required ordering of reasoning + function_call + function_call_output across turns.
- [Responses API reference](https://platform.openai.com/docs/api-reference/responses) — request/response body fields including `reasoning`, `include`, `store`, `previous_response_id`.
- [Responses streaming reference](https://platform.openai.com/docs/api-reference/responses-streaming) — SSE event taxonomy.
- [Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — how replayed reasoning interacts with `input_tokens_details.cached_tokens`.
