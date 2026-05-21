# Reasoning round-trip demo scripts

Three scripts — one per provider — that **prove** the proposed OpenInference reasoning attributes are sufficient to round-trip a thinking-aware LLM turn end-to-end.

Each script:

1. Calls the live API with thinking enabled.
2. Writes the assistant turn into an OTel `LLM` span as flat `llm.output_messages.*` attributes — including each vendor's opaque continuity token (`encrypted_content`, `signature`, `thoughtSignature`). Tools, model name, and reasoning config are captured too (`llm.tools.{i}.tool.json_schema`, `llm.model_name`, `llm.invocation_parameters`).
3. Exports both the raw instrumentor span (`current`) and the augmented span (`future`) to a local Phoenix instance.
4. Reconstructs the prior assistant turn purely from the in-memory `future` span attributes.
5. Issues turn 2 through replay helpers whose `model=`, `tools=`, and `**invocation_parameters` arguments come **only** from that captured attribute dict. The only Python-side inputs allowed are the genuinely-new follow-up user text and the tool execution result.

That last point is what makes the round-trip a real proof: turn 2 cannot fall back on Python globals (`WEATHER_TOOL`, reasoning config literals, model name) — if any of those weren't captured into attributes, turn 2 would crash with a `KeyError`.

The semantic conventions that already exist in `openinference-semantic-conventions` are imported directly (e.g. `SpanAttributes.LLM_OUTPUT_MESSAGES`, `MessageContentAttributes.MESSAGE_CONTENT_TYPE`, `ToolCallAttributes.TOOL_CALL_ID`). The new keys this PR is exploring live as string literals in `common.py`, clearly marked under `# Proposed`.

## Prerequisites

```bash
# 1. Phoenix locally
pip install arize-phoenix
phoenix serve   # listens on http://localhost:6006

# 2. API keys
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...

# 3. (optional) override Phoenix endpoint
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
```

## Run

```bash
uv run --script internal_docs/specs/reasoning/scripts/openai_roundtrip.py
uv run --script internal_docs/specs/reasoning/scripts/anthropic_roundtrip.py
uv run --script internal_docs/specs/reasoning/scripts/anthropic_legacy.py
uv run --script internal_docs/specs/reasoning/scripts/gemini_roundtrip.py
```

OpenAI reasoning summaries default to `detailed`. Toggle them with:

```bash
uv run --script internal_docs/specs/reasoning/scripts/openai_roundtrip.py --reasoning-summary off
uv run --script internal_docs/specs/reasoning/scripts/openai_roundtrip.py --reasoning-summary concise
export OPENAI_REASONING_SUMMARY=detailed  # or concise, auto, off
```

Anthropic defaults to Opus 4.7 adaptive thinking with `display=summarized`:

```bash
uv run --script internal_docs/specs/reasoning/scripts/anthropic_roundtrip.py
uv run --script internal_docs/specs/reasoning/scripts/anthropic_roundtrip.py --thinking-display omitted
export ANTHROPIC_THINKING_DISPLAY=summarized  # or omitted, or both
```

Legacy Anthropic extended thinking (`thinking.type=enabled` with `budget_tokens`):

```bash
uv run --script internal_docs/specs/reasoning/scripts/anthropic_legacy.py
export ANTHROPIC_BUDGET_TOKENS=4000
```

Gemini thought summaries default to on. Toggle `thinkingConfig.includeThoughts` with:

```bash
uv run --script internal_docs/specs/reasoning/scripts/gemini_roundtrip.py --include-thoughts off
export GEMINI_INCLUDE_THOUGHTS=on  # or off, true, false
```

Each script prints PASS/FAIL per scenario and exits non-zero on failure.

Open the Phoenix UI at <http://localhost:6006> to inspect the two-turn spans and confirm the reasoning attributes are present.

## Scenarios

Every script covers two scenarios:

| Scenario | What it proves |
|---|---|
| **A. text** | Reasoning → text. Captures the continuity token; the next turn lands without an error and the model picks up where it left off. |
| **B. tool** | Reasoning → tool_use → tool_result → follow-up user question → final answer. Uses `TOOL_USER_PROMPT` / `TOOL_FOLLOW_UP_PROMPT` in `common.py` (heat-advisory + °F conversion) so the model must plan before the tool and compute step-by-step after it—not a one-line weather lookup. OpenAI tool turns use `reasoning.effort: high`. |

Anthropic and Gemini-3 scripts additionally run a **negative assertion** where the signature is stripped before echo; the API is expected to return HTTP 400, confirming the signature is genuinely load-bearing (and confirming we actually captured it).

Gemini 2.5 silently drops thinking on signature loss instead of erroring, so its negative assertion is skipped with a warning when the model is not on the Gemini 3 family.

## Continuity-token normalization

Each vendor names its echo-on-next-turn token differently:

| Vendor | API field | Mechanism |
|---|---|---|
| OpenAI | `encrypted_content` | opaque ciphertext that **carries** the chain-of-thought state |
| Anthropic | `signature` | HMAC-like token that **authenticates** the plaintext `thinking` (which rides back too) |
| Gemini | `thoughtSignature` | opaque encrypted representation that **carries** internal reasoning state |

These scripts normalize response-level continuity tokens to `message_content.signature` inside the same flat content-block shape, indexed by the same `llm.output_messages.{i}.message.contents.{j}.*` walker. Gemini `functionCall` signatures use `tool_call.signature` because the token belongs to the tool-call part, not the surrounding reasoning summary.

## Follow-ups

Open design questions for when these conventions get promoted into `openinference-semantic-conventions`:

- **Redaction interaction with `TraceConfig`.** The existing `TraceConfig.hide_llm_output_messages` masks the *visible* `message_content.text`. Does it also need to mask reasoning text and continuity tokens, or should there be a separate `hide_reasoning_content` / `hide_continuity_token` flag? Continuity tokens are opaque bytes but not safe to log indefinitely (they encode state the model trusts).
- **Anthropic `redacted_thinking` round-trip.** This script captures the `data` blob as `message_content.signature` but the scenarios never exercise the redacted path — Anthropic only emits it for content their safety classifier flagged. Recommend adding a synthetic injection test (or a recorded fixture) once we have a reproducible trigger.
- **`hide_*` masking semantics for redacted content.** A reasoning block carrying vendor-redacted data in `message_content.signature` is already vendor-redacted; OpenInference's own masking should be additive, not duplicative. Settle the precedence rule.
- **Cost / token accounting.** Reasoning tokens are billed (OpenAI `reasoning_tokens`, Gemini `thoughtsTokenCount`, Anthropic rolled into `output_tokens`). Existing `LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING` covers OpenAI / Gemini cleanly; Anthropic needs documentation that it's not separately attributable.
- **Streaming.** These scripts use non-streaming responses. Streaming surfaces (OpenAI's typed SSE, Anthropic's `signature_delta`, Gemini's chunked `parts[]`) emit the continuity token only at well-defined points — the instrumentor's stream wrapper has to buffer correctly. Out of scope here but worth a sibling demo before the conventions land.

## Per-provider quirks captured

| Quirk | Where |
|---|---|
| OpenAI `encrypted_content` only populated when `include: ["reasoning.encrypted_content"]` is set | `openai_roundtrip.py` request config |
| OpenAI `reasoning` items carry an `id` that must be echoed | `set_output_reasoning(item_id=...)` |
| Anthropic `signature` is byte-for-byte HMAC over the `thinking` plaintext | `set_output_reasoning(signature=...)` |
| Anthropic `redacted_thinking` carries a `data` blob instead of `thinking` text | `message_content.signature` with no `message_content.text` |
| Gemini `thoughtSignature` is `bytes` on the SDK / base64 on the wire | base64-encoded for storage in `gemini_roundtrip.py`, decoded on echo |
| Gemini summary parts (`thought: true`) **never** carry a signature; signatures ride on the sibling data part | `message_content.signature` on text blocks or `tool_call.signature` on tool-use blocks |

## Attribute keys used

Existing semconv (imported from `openinference.semconv.trace`):

| Key | Source |
|---|---|
| `llm.model_name`, `llm.provider`, `llm.system`, `llm.invocation_parameters`, `openinference.span.kind` | `SpanAttributes` |
| `llm.tools.{i}.tool.json_schema` | `SpanAttributes.LLM_TOOLS` + `tool.json_schema` |
| `llm.input_messages.{i}.message.role`, `.message.content`, `.message.tool_call_id` | `MessageAttributes` |
| `llm.output_messages.{i}.message.role`, `.message.contents.{j}.*` | `MessageAttributes` |
| `message_content.type`, `message_content.text` | `MessageContentAttributes` |
| `tool_call.id`, `tool_call.function.name`, `tool_call.function.arguments` | `ToolCallAttributes` |

Proposed (string literals in `common.py`, **not yet** in semconv):

| Purpose | Attribute suffix |
|---|---|
| Reasoning block discriminator | `message_content.type` = `"reasoning"` |
| Tool-use block discriminator | `message_content.type` = `"tool_use"` |
| OpenAI reasoning item id | `message_content.id` |
| Response continuity token | `message_content.signature` |
| Gemini tool-call continuity token | `tool_call.signature` |

A single `message_content.type = "reasoning"` covers every reasoning surface — raw thinking, summarized thinking (Gemini thought parts, OpenAI summaries), and Anthropic redacted thinking. There is no separate `reasoning_summary` or `redacted_reasoning` type: a redacted Anthropic block is represented as a reasoning block with `message_content.signature` and no `message_content.text`.

Reasoning request config (effort / budget / level / display) is captured as JSON inside the existing `llm.invocation_parameters` attribute — no new top-level keys are proposed for it.

## Layout

```
scripts/
├── README.md
├── common.py               — OTel setup, attribute (de)serializers, span export helpers
├── openai_roundtrip.py     — Responses API, stateless path
├── anthropic_roundtrip.py              — Opus 4.7 adaptive thinking, display omitted/summarized
├── anthropic_legacy.py                 — Legacy extended thinking with budget_tokens
└── gemini_roundtrip.py                 — generateContent, thinkingConfig, thoughtSignature handling
```
