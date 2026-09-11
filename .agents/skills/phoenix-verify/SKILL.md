---
name: phoenix-verify
description: Prove an OpenInference instrumentation change end to end by running a package example against a local Phoenix (http://localhost:6006) in a dedicated project and inspecting the resulting spans with the px CLI. Use when an instrumentation fix or feature changes what a user sees in Phoenix (span tree, span kind, status, attributes, sessions, metadata), when the user asks for evidence in Phoenix, a before/after comparison, or a round trip, or when adding a runnable example under a package's examples/ directory.
---

# Phoenix Verify

Unit tests gate the push. When a change alters what a user sees in the Phoenix UI, pair the
tests with a real round trip: run an example that exports to local Phoenix, read the spans
back with `px`, and check one specific claim. The output of that check is the evidence that
goes in the PR or summary. Detailed recipes live in [REFERENCE.md](REFERENCE.md).

## Prerequisites

- **Phoenix reachable.** `px project list --format raw --no-progress > /dev/null && echo "phoenix ok at ${PHOENIX_HOST:-http://localhost:6006}"` must print the ok line. If px exits non-zero (`fetch failed`), stop and report NOT VERIFIED (template in step 7). Do not rely on `px auth status`; it exits 0 when the server is unreachable. Default host is `http://localhost:6006`; set `PHOENIX_HOST` for px if different, and point the example's exporter at the same host. Start one with `pip install arize-phoenix && phoenix serve` (`PHOENIX_PORT=<n>` for another port).
- **`px` and `jq` on PATH** (`npx @arizeai/phoenix-cli` also works). `--attribute` filters need Phoenix server >= 14.9.0.
- **Provider API keys** exported when the example calls a model. Prefer an offline example (canned replies, tool-only) when the claim does not depend on model output. Some packages have none; say so rather than inventing one.

## Workflow

Copy this checklist and work through it in order. `$SCRATCH` is your session scratchpad directory; `<pkg>` is the short provider name (`openai`, `ag2`, `anthropic`) and the package directory is `openinference-instrumentation-<pkg>`.

1. **State the claim.** One sentence naming the span, attribute, kind, or status that should be a certain way. Everything below proves or disproves this sentence. Decide the mode now: **single run** (verify existing behavior, or a new feature) or **before/after** (a fix or a parity check).
2. **Pick or write the example.** Python: `python/instrumentation/<pkg>/examples/<scenario>.py`. JS: `js/packages/<pkg>/examples/<scenario>.ts`. Reuse an existing example that exercises the path; otherwise write one from the [templates](REFERENCE.md#example-templates). Examples use the plain OpenTelemetry SDK: a `TracerProvider` whose resource carries `openinference.project.name` = `<pkg>-<scenario>`, an OTLP HTTP exporter to `http://localhost:6006/v1/traces`, and a `SimpleSpanProcessor`. `<scenario>` is the example's filename stem. To run into a different project (before/after) or host, or when an example sets no project resource (spans land in `default`) or shares a bootstrap file, copy it to `$SCRATCH` and edit the copy; do not rewrite the committed example just to run it.
3. **Set up an isolated environment** so the working tree, not a release, is what runs. Python needs a venv on the package's `requires-python` (usually >= 3.10; tox venvs install non-editably and do not see working-tree edits):

   ```bash
   uv venv --python 3.12 "$SCRATCH/venv-<pkg>"
   uv pip install --python "$SCRATCH/venv-<pkg>/bin/python" -e python/instrumentation/openinference-instrumentation-<pkg> -r python/instrumentation/openinference-instrumentation-<pkg>/examples/requirements.txt
   "$SCRATCH/venv-<pkg>/bin/python" -c 'import openinference.instrumentation.<mod> as m; print(m.__file__)'   # working tree: a path under the repo's src/; a PyPI "before" venv: site-packages
   ```

   Install `-e` and the requirements in one command: `requirements.txt` lists the package from PyPI and would shadow the editable install if installed afterwards. JS: `pnpm install --frozen-lockfile -r` then `pnpm --filter "<npm-name>..." run build` inside `js/`; examples import `../src` directly, the build is for workspace deps. For before/after, use two venvs; ways to get a "before" build are in [REFERENCE.md](REFERENCE.md#getting-a-before-build).
4. **Confirm the project is empty, then run.** `px span list --project <project> --format raw --no-progress 2>&1 | head -c 200` must show `[]` or a `404 Not Found` error (the project does not exist yet). Anything else means old spans are present: pick a new name, because they are not evidence. Run the example and read its output: an exit code of 0 does not mean spans were exported. `Failed to export` or `Connection refused` in the log means Phoenix never received them. For before/after, run into `<project>-before` and `<project>-after`.
5. **Read the spans back.** Start with the tree, then narrow:

   ```bash
   S=.agents/skills/phoenix-verify/scripts/span_tree.sh
   $S <project>            # name [KIND] status, nested
   $S <project> keys       # attribute keys per span
   $S <project> values     # attribute values, volatile keys dropped (good for diffs)
   $S <project> errors     # ERROR spans + exception.message
   $S <project> -- --last-n-minutes 10 --limit 500    # any px span list flags after --
   px span list --project <project> --format raw --no-progress --limit 500 > "$SCRATCH/<project>.spans.json"
   jq length "$SCRATCH/<project>.spans.json"          # check the count before trusting any filter
   ```

   Name saved files after the project. A shared name like `spans.json` gets overwritten by a parallel run. If jq reports `Invalid numeric literal`, px printed an error, not JSON. Targeted jq recipes and the before/after diff are in [REFERENCE.md](REFERENCE.md#jq-recipes).
6. **Judge the claim.** Compare the output to step 1. If it does not match, the fix is not done. Do not weaken the claim to fit the output. An empty before/after diff is the right answer for a parity claim and the wrong answer for a fix.
7. **Report the evidence.** Use one of these shapes in the PR body or final message:

   ```
   Verified against local Phoenix (project `<project>`):
   <exact commands>
   <trimmed output showing the claim>
   Before (`<project>-before`): <how it differed, or "identical">   # before/after mode only
   Builds: before=<__file__ output>, after=<__file__ output>          # before/after mode only

   NOT VERIFIED: <what failed, e.g. Phoenix unreachable, example exited 1, 0 spans exported>
   Phoenix: <host used, and how it was set>
   <command> -> <error text>
   Example ran: yes/no. Spans read back: <n>. Claim status: unverified | falsified.
   ```
8. **Wire the example into the repo** only if you wrote or changed one. Add a row to `examples/README.md` (create it if the package has none) or a `## Examples` section in the package README, add new direct imports to `examples/requirements.txt` (Python) or `devDependencies` (JS), and lint exactly as CI does: Python from the package directory with the pinned ruff, `tox run -e ruff-<pkg>` or `uvx ruff@0.9.2 format --diff . && uvx ruff@0.9.2 check --no-fix .` (an unpinned ruff reports false import-order errors). JS examples are excluded from oxlint and oxfmt, so nothing runs on them. Never commit API keys or cloud endpoints.

## Gotchas

- **Flush before exit.** `SimpleSpanProcessor` exports synchronously; failures are log-only and never change the exit code. With `BatchSpanProcessor`, call `tracer_provider.force_flush()` before the script ends. In JS, a pending OTLP request keeps Node alive, but `process.exit()` or an unhandled rejection drops it, so `await provider.forceFlush()` at the end of `main()`.
- **Metadata is flattened.** `using_attributes(metadata={"run": "x"})` lands as `metadata.run`, not a `metadata` key. Filter with `startswith("metadata.")`. Sessions, users, and tags are `session.id`, `user.id`, `tag.tags`.
- **Zero spans looks like a 404.** A project that never received a span does not exist, so `px` returns `Failed to resolve project`. To prove suppression, make one traced call and one call inside `suppress_tracing()` in the same run and assert `jq length` is `1`.
- **Newest first, capped.** `px span list` returns the newest 100 spans by default. Raise `--limit` for long traces or filter with `--trace-id`.
- **`llm.model_name` is the resolved snapshot** (`gpt-4o-mini-2024-07-18`), not the alias you requested. Do not `--attribute`-filter on the alias.
- **OpenTelemetry resources API differs by major.** `resourceFromAttributes` exists only in `@opentelemetry/resources` 2.x; most instrumentation packages pin 1.x and use `new Resource({...})`. Check the package's `package.json` before copying the JS template.
- **Run JS examples with `pnpm exec tsx examples/<scenario>.ts`** from the package directory. Some package READMEs say `npx tsx`; the repo rule is pnpm only.

## Related

- `phoenix-cli` skill: full `px` command reference, trace and session JSON shapes, GraphQL.
- `python-code-reviewer` / `java-code-reviewer`: run after the fix is proven, before opening the PR.
