---
name: genai-conformance
description: Run, interpret, and iterate on the OpenInference GenAI conformance MVP at python/openinference-instrumentation/scripts/conformance/. Use when the user mentions GenAI conformance, OTel GenAI semantic conventions, Weaver registry live-check, the dual-write conversion (`_genai_conversion.py`, `enable_genai_semconv`), `gen_ai.*` attribute coverage, or asks to add new providers / scenarios to the conformance harness.
---

# GenAI Conformance

The repo ships a self-contained conformance harness at [python/openinference-instrumentation/scripts/conformance/](../../python/openinference-instrumentation/scripts/conformance/) that exercises OpenInference instrumentors against deterministic mock provider APIs, exports OTLP traces to `weaver registry live-check`, and prints a console summary of registry attributes seen / missing / advice-level counts. It validates the **dual-write** logic in [_genai_conversion.py](../../python/openinference-instrumentation/src/openinference/instrumentation/_genai_conversion.py) that translates OpenInference's native attributes (`llm.*`, `input.*`, `output.*`, `openinference.*`) into the OTel GenAI semantic conventions (`gen_ai.*`).

## When to Use

- User asks to run the conformance harness, "test conformance", or "run weaver".
- User asks to maximize / improve `gen_ai.*` registry coverage.
- User wants to extend the dual-write conversion in `_genai_conversion.py`.
- User wants to add a new provider, a new test scenario, or a new mock endpoint.
- User mentions specific `gen_ai.*` attributes (response.id, system_instructions, tool.call.*, retrieval.*, etc.) and whether they're being emitted.

## Layout

```
scripts/conformance/
├── run.py                 # orchestrator (PEP 723, stdlib only)
├── mock_server.py         # Flask mock with all providers' endpoints
├── anthropic_conformance.py      # PEP 723 + editable [tool.uv.sources]
├── openai_conformance.py         # PEP 723 + editable [tool.uv.sources]
├── google_genai_conformance.py   # PEP 723 + editable [tool.uv.sources]
├── README.md
└── results/                      # gitignored Weaver output
```

Each provider script declares its deps as PEP 723 inline metadata and pins the local OpenInference packages via `[tool.uv.sources.<pkg>]` blocks (multi-section dotted-key form — single-line inline tables exceed ruff's 100-char limit). `run.py` invokes everything via `uv run`. Filenames avoid the bare provider name (`openai.py`, `anthropic.py`) because that would shadow the SDK package on `sys.path[0]`.

`run.py` lives in [PROVIDER_SCRIPTS](../../python/openinference-instrumentation/scripts/conformance/run.py) — a tuple iterated for both prewarm and execution. To add a provider, append to `PROVIDER_SCRIPTS` and add the corresponding `<provider>_conformance.py` and any new mock endpoints.

## Running

```bash
uv run python/openinference-instrumentation/scripts/conformance/run.py
```

First run downloads pinned `weaver v0.22.1` and `semantic-conventions v1.40.0` to `~/.cache/oi-conformance/`; subsequent runs are fast. uv caches each provider script's env by PEP 723 metadata hash.

## Interpreting the summary

- **Registry attributes seen** — `gen_ai.*` (and a few `service.*` / `telemetry.sdk.*`) attrs the run emitted, with sample counts.
- **Non-registry attributes seen** — OpenInference's native vocabulary. These show up as Weaver `missing_attribute` violations by design — they aren't (and shouldn't be) in the OTel registry.
- **Missing registry attributes (`gen_ai.*`)** — registry attrs the run did *not* emit. Categorize each one:
  1. **Real dual-write gap** — provider API has the data, instrumentor captures it as an OI attr, but `_genai_conversion.py` doesn't map it. **Fixable in conversion.**
  2. **Test scenario gap** — conversion handles it, but the test doesn't exercise the relevant scenario (e.g. `gen_ai.tool.call.*` need a TOOL span; `gen_ai.embeddings.*` need an EMBEDDING span). **Fixable in `<provider>_conformance.py`.**
  3. **Mock data gap** — instrumentor would capture it if the response included it (e.g. `gen_ai.usage.cache_read.input_tokens` requires `cache_read_input_tokens` in the mock's usage block). **Fixable in mock_server.py.**
  4. **Provider doesn't support it** — e.g. Anthropic has no `frequency_penalty`. Document and skip.
  5. **Application-level / not auto-emittable** — `gen_ai.agent.*`, `gen_ai.evaluation.*`, `gen_ai.prompt.name`, `gen_ai.data_source.id`. Require explicit user attribution; out of scope for SDK instrumentation.
  6. **Metric-only** — `gen_ai.token.type` lives on `gen_ai.client.token.usage` metric, not spans.
- **Advice levels** — `violation` counts are predominantly `missing_attribute` for the OI native vocab (expected); `improvement` counts are `not_stable` warnings for development-stage `gen_ai.*` attrs (also expected). The dual-write itself is well-formed — Weaver does not flag type/shape/value errors on the emitted `gen_ai.*` attrs.

## Iterating to maximize coverage

For category 1 (dual-write gap):

1. Inspect `results/live_check.json` to see exactly what OI attributes the instrumentor emitted (look for the relevant span's `attributes` array).
2. Decide where to extend `_genai_conversion.py` (`get_genai_request_attributes`, `get_genai_response_attributes`, etc.).
3. **Always add a unit test in [test_genai.py](../../python/openinference-instrumentation/tests/test_genai.py)** for the new path. The existing tests cover the major span kinds; mirror that style.
4. Re-run the conformance harness. Verify the missing list shrinks and no existing `gen_ai.*` attribute regressed.

For category 2 (test scenario gap):

- Use `OITracer(trace.get_tracer(__name__), TraceConfig(enable_genai_semconv=True))` to manually emit non-LLM spans (TOOL, RETRIEVER, EMBEDDING, AGENT) inside a provider script. The Anthropic script already does this for TOOL / RETRIEVER / EMBEDDING — copy the pattern.

For category 3 (mock data gap):

- Mock responses are simple dicts at the top of `mock_server.py`. The Anthropic mock already returns `cache_creation_input_tokens` / `cache_read_input_tokens`; the OpenAI mock returns `prompt_tokens_details.cached_tokens`. Add fields the SDK will surface and the OI instrumentor will turn into `LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_*`.

## Bumping the semconv version

The harness pins `SEMCONV_VERSION` (currently `v1.41.1`) and `WEAVER_VERSION` (currently `v0.23.0`) in `run.py`. When OTel cuts a new semconv release, walk this checklist:

1. **Check for a newer Weaver release too** — always run `gh release list --repo open-telemetry/weaver --limit 5` alongside the semconv check. Weaver and the registry version independently; the harness depends on both. Bump `WEAVER_VERSION` whenever a newer release exists, and skim its notes for `live-check`-relevant fixes.
2. **Bump the constants** in [run.py](../../python/openinference-instrumentation/scripts/conformance/run.py): `SEMCONV_VERSION` and `WEAVER_VERSION` to the latest releases.
3. **Run the harness once** (`uv run python/openinference-instrumentation/scripts/conformance/run.py`) so it downloads the new registry into `~/.cache/oi-conformance/semconv/<new-version>/`.
4. **Refresh the vendored JSON schemas** at [tests/fixtures/genai_schemas/](../../python/openinference-instrumentation/tests/fixtures/genai_schemas/) from `~/.cache/oi-conformance/semconv/<new-version>/docs/gen-ai/gen-ai-{input,output}-messages.json`.
5. **Run unit tests** (`pytest tests/test_genai.py`). The `_load_json_attribute` validator runs the new schemas against every emitted message payload — any breaking shape change surfaces here.
6. **Skim the semconv changelog** for these specific risks (each one usually requires a code change in `_genai_conversion.py`):
   - New required fields on `ChatMessage` / `OutputMessage` parts (`TextPart`, `ToolCallRequestPart`, etc.) → builder functions need to populate them.
   - New `Role` enum values → `_normalize_message_role` may need a mapping.
   - New `FinishReason` enum values → `_normalize_finish_reason` may need a mapping.
   - Added `gen_ai.*` registry attrs → opportunity for new dual-write mappings; re-run the harness and look at the "Missing registry attributes" summary.
   - Removed / renamed `gen_ai.*` attrs → drop from `_genai_attributes.py` and stop emitting in `_genai_conversion.py`.
7. **Refresh inline version refs**: the semconv-version mentions in [test_genai.py](../../python/openinference-instrumentation/tests/test_genai.py) (schema-source comment), [README.md](../../python/openinference-instrumentation/scripts/conformance/README.md) (caveats section, includes Weaver version too), and [_genai_conversion.py](../../python/openinference-instrumentation/src/openinference/instrumentation/_genai_conversion.py) (the encoding comment inside `get_genai_message_attributes`).
8. **Re-run the conformance harness** end-to-end; verify no `gen_ai.*` attribute regressed and no genuine shape errors appear in `results/live_check.json` (advice with `id != "missing_attribute"`).

## Gotchas

- **PEP 723 inline-table line length**: `[tool.uv.sources.<pkg>] { path = "...", editable = true }` on one line easily exceeds 100 chars and trips ruff E501. Use the multi-section form (`[tool.uv.sources.<pkg>]\npath = "..."\neditable = true`).
- **The conformance dir is excluded from package-level checks**: `pyproject.toml` excludes `scripts/*` from mypy and `scripts` from pytest's `norecursedirs`. ruff still lints it. Don't add Python imports that mypy/pytest can't resolve in the lint env (e.g. provider SDKs) outside this directory.
- **Weaver inactivity timeout** is 90s. First-run uv installs of OpenTelemetry/OpenAI/Google SDKs can take a while — `run.py` prewarms each provider env via `--prewarm` early-exit before starting Weaver. If you add a new provider script, give it a `--prewarm` early-exit too.
- **Single mock server** for all providers: one Flask app handles `/v1/messages` (Anthropic), `/v1/chat/completions` + `/v1/embeddings` + `/v1/responses` (OpenAI), and `/v1beta/models/<path:model>` (Google). Each endpoint discriminates between text and tool variants by checking `body.get("tools")`.
- **Editable installs are mandatory**: PyPI versions of `openinference-instrumentation` and the per-provider packages won't have in-progress dual-write changes. The `[tool.uv.sources]` blocks in each `<provider>_conformance.py` pin the local repo paths.
- **System messages stay in `gen_ai.input.messages`.** The dual-write does not emit `gen_ai.system_instructions`; system instructions are assumed to flow through as a system-role entry in `LLM_INPUT_MESSAGES`. The `gen_ai.input.messages` JSON schema explicitly admits `"system"` as a valid role.
- **`tool_call` (singular) finish_reason** — the conversion normalizes both `tool_calls` (OpenAI plural) and `function_call` (legacy) to `"tool_call"`. The OTel registry doesn't constrain values for `gen_ai.response.finish_reasons`, but OTel's conventional value is plural `tool_calls`. If you change the normalization target, update the asserting tests in `test_genai.py` too.
- **Don't conflate violations with shape errors**: the headline "violation: N" counts `missing_attribute` advice on OI native attrs (expected) plus any genuine shape mismatches on `gen_ai.*` attrs (real bugs). To find genuine shape errors, parse `results/live_check.json` and look at advice with `id != "missing_attribute"`.
