# Reasoning round-trip demo scripts

Three scripts — one per provider — that **prove** the proposed OpenInference reasoning attributes are sufficient to round-trip a thinking-aware LLM turn end-to-end.

Each script:

1. Calls the live API with thinking enabled.
2. Writes the assistant turn into an OTel `LLM` span as flat string-valued `llm.output_messages.*` attributes — including each vendor's opaque continuity token (`encrypted_content`, `signature`, `thoughtSignature`).
3. Force-flushes the span to a local Phoenix instance.
4. Uses `phoenix.client.Client` to fetch the span **back out of Phoenix**.
5. Reconstructs the prior assistant turn purely from the fetched attributes (no in-memory references to the original SDK objects).
6. Calls the API a second time with that reconstructed history and asserts success.

The point of fetching back out of Phoenix is that round-trip via Python objects in the same process is not a real test — the only thing we care about is whether the on-the-wire attribute representation is lossless.

The semantic-convention keys are **string literals**; nothing here is wired into the `openinference-semantic-conventions` package.

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

## Per-provider quirks captured

| Quirk | Where |
|---|---|
| OpenAI `encrypted_content` only populated when `include: ["reasoning.encrypted_content"]` is set | `openai_roundtrip.py` request config |
| OpenAI `reasoning` items carry an `id` that must be echoed | `set_output_reasoning(item_id=...)` |
| Anthropic `signature` is byte-for-byte HMAC over the `thinking` plaintext | `set_output_reasoning(signature=...)` |
| Anthropic `redacted_thinking` carries a `data` blob instead of `thinking` text | `set_output_redacted_reasoning(data=...)` |
| Gemini `thoughtSignature` is `bytes` on the SDK / base64 on the wire | base64-encoded for storage in `gemini_roundtrip.py`, decoded on echo |
| Gemini summary parts (`thought: true`) **never** carry a signature; signatures ride on the sibling data part | `set_output_reasoning_summary` vs `thought_signature` on text / tool_use blocks |

## Proposed attribute keys (string literals, not promoted to `SpanAttributes`)

| Purpose | Attribute key |
|---|---|
| Output message role | `llm.output_messages.{i}.message.role` |
| Visible text | `llm.output_messages.{i}.message.contents.{j}.message_content.type` = `"text"`, `.text` |
| Reasoning block | `...message_content.type` = `"reasoning"` |
| Reasoning visible text | `...message_content.text` |
| Reasoning item id | `...message_content.id` (OpenAI) |
| OpenAI continuity token | `...message_content.encrypted_content` |
| Anthropic continuity token | `...message_content.signature` |
| Anthropic redacted blob | `...message_content.redacted_data` (with `type` = `"redacted_reasoning"`) |
| Gemini continuity token | `...message_content.thought_signature` (sibling of data type) |
| Gemini thought summary | `...message_content.type` = `"reasoning_summary"`, `.text` |
| Tool call | `...message_content.type` = `"tool_use"`, `.tool_call.id`, `.tool_call.function.name`, `.tool_call.function.arguments` |
| Tool result (user turn) | `llm.input_messages.{i}.message.role` = `"tool"`, `.tool_call_id`, `.content` |
| Reasoning request config | Captured as JSON on the existing `llm.invocation_parameters` attribute (e.g. `{"reasoning":{"effort":"medium",...}}` for OpenAI, `{"thinking":{"type":"enabled","budget_tokens":8000}}` for Anthropic, `{"generationConfig":{"thinkingConfig":{...}}}` for Gemini). No new top-level keys proposed. |

## Layout

```
scripts/
├── README.md
├── common.py               — OTel setup, attribute (de)serializers, Phoenix span fetcher
├── openai_roundtrip.py     — Responses API, stateless path
├── anthropic_roundtrip.py  — Messages API, extended thinking, signature negative test
└── gemini_roundtrip.py     — generateContent, thinkingConfig, thoughtSignature handling
```
