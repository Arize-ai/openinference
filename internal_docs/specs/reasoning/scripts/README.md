# Reasoning round-trip demo scripts

Three scripts — one per provider — that **prove** the proposed OpenInference reasoning attributes are sufficient to round-trip a thinking-aware LLM turn end-to-end.

Each script:

1. Calls the live API with thinking enabled.
2. Writes the assistant turn into an OTel `LLM` span as flat `llm.output_messages.*` attributes — including each vendor's opaque continuity token (`encrypted_content`, `signature`, `thoughtSignature`). Tools, model name, and reasoning config are captured too (`llm.tools.{i}.tool.json_schema`, `llm.model_name`, `llm.invocation_parameters`).
3. Force-flushes the span to a local Phoenix instance.
4. Uses `phoenix.client.Client` to fetch the span **back out of Phoenix**.
5. Reconstructs the prior assistant turn purely from the fetched attributes.
6. Issues turn 2 through a `replay_turn2(...)` helper whose `model=`, `tools=`, and `**invocation_parameters` arguments come **only** from the Phoenix-fetched attribute dict. The only Python-side inputs allowed are the genuinely-new follow-up user text and the tool execution result.

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
uv run --script internal_docs/specs/reasoning/scripts/gemini_roundtrip.py
```

Each script prints PASS/FAIL per scenario and exits non-zero on failure.

Open the Phoenix UI at <http://localhost:6006> to inspect the two-turn spans and confirm the reasoning attributes are present.

## Scenarios

Every script covers two scenarios:

| Scenario | What it proves |
|---|---|
| **A. text** | Reasoning → text. Captures the continuity token; the next turn lands without an error and the model picks up where it left off. |
| **B. tool** | Reasoning → tool_use → tool_result → final answer. This is the load-bearing case: across all three vendors the API rejects the next request if the continuity token attached to (or preceding) the tool call is dropped. |

Anthropic and Gemini-3 scripts additionally run a **negative assertion** where the signature is stripped before echo; the API is expected to return HTTP 400, confirming the signature is genuinely load-bearing (and confirming we actually captured it).

Gemini 2.5 silently drops thinking on signature loss instead of erroring, so its negative assertion is skipped with a warning when the model is not on the Gemini 3 family.

## Continuity-token normalization

Each vendor names its echo-on-next-turn token differently:

| Vendor | API field | Mechanism |
|---|---|---|
| OpenAI | `encrypted_content` | opaque ciphertext that **carries** the chain-of-thought state |
| Anthropic | `signature` | HMAC-like token that **authenticates** the plaintext `thinking` (which rides back too) |
| Gemini | `thoughtSignature` | opaque encrypted representation that **carries** internal reasoning state |

These scripts normalize them **structurally**: all three live as siblings of `message_content.type` inside the same flat content-block shape, indexed by the same `llm.output_messages.{i}.message.contents.{j}.*` walker. The vendor's field name is preserved as the sub-field (`message_content.encrypted_content`, `message_content.signature`, `message_content.thought_signature`) so consumers can tell which echo discipline applies — but the discovery path, ordering, and round-trip logic are identical across providers. That is what makes `read_output_contents` and `set_input_assistant_echo` provider-agnostic.

A single normalized alias (e.g. `message_content.continuity_token`) is **not** proposed here — see Follow-ups below.

## Follow-ups

Open design questions for when these conventions get promoted into `openinference-semantic-conventions`:

- **Normalized continuity-token alias.** Should we additionally write a single `message_content.continuity_token` (and `message_content.continuity_token.kind` ∈ `{encrypted_content, signature, thought_signature}`) so provider-agnostic consumers don't need to check three field names? Trade-off: aliasing duplicates bytes; tagging-only keeps the wire payload single but adds a discriminator field.
- **Redaction interaction with `TraceConfig`.** The existing `TraceConfig.hide_llm_output_messages` masks the *visible* `message_content.text`. Does it also need to mask reasoning text and continuity tokens, or should there be a separate `hide_reasoning_content` / `hide_continuity_token` flag? Continuity tokens are opaque bytes but not safe to log indefinitely (they encode state the model trusts).
- **Anthropic `redacted_thinking` round-trip.** This script captures `redacted_data` as `message_content.redacted_data` but the scenarios never exercise the redacted path — Anthropic only emits it for content their safety classifier flagged. Recommend adding a synthetic injection test (or a recorded fixture) once we have a reproducible trigger.
- **`hide_*` masking semantics for redacted content.** If a span carries `type: redacted_reasoning` the `data` blob is already vendor-redacted; OpenInference's own masking should be additive, not duplicative. Settle the precedence rule.
- **Cost / token accounting.** Reasoning tokens are billed (OpenAI `reasoning_tokens`, Gemini `thoughtsTokenCount`, Anthropic rolled into `output_tokens`). Existing `LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING` covers OpenAI / Gemini cleanly; Anthropic needs documentation that it's not separately attributable.
- **Streaming.** These scripts use non-streaming responses. Streaming surfaces (OpenAI's typed SSE, Anthropic's `signature_delta`, Gemini's chunked `parts[]`) emit the continuity token only at well-defined points — the instrumentor's stream wrapper has to buffer correctly. Out of scope here but worth a sibling demo before the conventions land.

## Per-provider quirks captured

| Quirk | Where |
|---|---|
| OpenAI `encrypted_content` only populated when `include: ["reasoning.encrypted_content"]` is set | `openai_roundtrip.py` request config |
| OpenAI `reasoning` items carry an `id` that must be echoed | `set_output_reasoning(item_id=...)` |
| Anthropic `signature` is byte-for-byte HMAC over the `thinking` plaintext | `set_output_reasoning(signature=...)` |
| Anthropic `redacted_thinking` carries a `data` blob instead of `thinking` text | `set_output_redacted_reasoning(data=...)` |
| Gemini `thoughtSignature` is `bytes` on the SDK / base64 on the wire | base64-encoded for storage in `gemini_roundtrip.py`, decoded on echo |
| Gemini summary parts (`thought: true`) **never** carry a signature; signatures ride on the sibling data part | `set_output_reasoning_summary` vs `thought_signature` on text / tool_use blocks |

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
| Gemini thought-summary block | `message_content.type` = `"reasoning_summary"` |
| Anthropic redacted-thinking block | `message_content.type` = `"redacted_reasoning"` |
| Tool-use block discriminator | `message_content.type` = `"tool_use"` |
| OpenAI reasoning item id | `message_content.id` |
| OpenAI continuity token | `message_content.encrypted_content` |
| Anthropic continuity token | `message_content.signature` |
| Gemini continuity token | `message_content.thought_signature` (sibling of any data part) |
| Anthropic redacted blob | `message_content.redacted_data` |

Reasoning request config (effort / budget / level / display) is captured as JSON inside the existing `llm.invocation_parameters` attribute — no new top-level keys are proposed for it.

## Layout

```
scripts/
├── README.md
├── common.py               — OTel setup, attribute (de)serializers, Phoenix span fetcher
├── openai_roundtrip.py     — Responses API, stateless path
├── anthropic_roundtrip.py  — Messages API, extended thinking, signature negative test
└── gemini_roundtrip.py     — generateContent, thinkingConfig, thoughtSignature handling
```
