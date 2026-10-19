---
name: python-review
description: >
  Review an OpenInference Python instrumentor under python/instrumentation/ for
  correctness and project conventions, then prove the findings with the CI tox env and
  the phoenix-verify skill. Use for a new instrumentor, a PR that changes one, an audit
  of an existing package, or a repo-wide convention sweep. Trigger on "review the Python
  instrumentor", "review PR <n>" when it touches python/instrumentation/, "audit the
  <name> instrumentor", "is this instrumentor correct", "check conventions across the
  Python instrumentors", or when another skill (python-canary-fix, issue-triage) asks for
  a Python review. Not for Java or JS packages.
compatibility: >
  python3 and git for the check script; uv and tox-uv for the CI env; gh for PR review.
  Phoenix verification needs a local Phoenix and the px CLI, as phoenix-verify describes.
invocable: true
---

# Python Review

A review has three layers, and each one feeds the report:

1. **Script.** `scripts/check_package.py` runs the deterministic checks (tox wiring,
   packaging, suppression misuse, cassette secrets, and grep-level test and runtime
   signals). Never hand-check what it covers.
2. **Judgment.** Read the code against the instrumented library's real source, using only
   the reference files that match what changed.
3. **Proof.** The package's CI env proves the build and tests. The `phoenix-verify` skill
   proves any claim about spans a user would see in Phoenix.

## Inputs

- **Target.** One package (`openai`, `google_adk`, or the directory name), a PR number,
  or `--all` for a repo-wide sweep.
- **Focus** (optional). A subset of dimensions (`ci`, `tests`, `semconv`, `runtime`)
  when the caller only needs some. The script always runs in full; it is cheap.
- **Mode.** `interactive` (default) ends by asking what to fix. `headless` (CI, or a
  calling skill) never asks, never blocks on Phoenix, and returns the report only.

## Workflow

1. **Scope.** For a PR, list touched packages with
   `gh pr diff <n> --name-only | grep '^python/instrumentation/' | cut -d/ -f3 | sort -u`
   and review each one separately. Read the diff, then the code it calls and is called by.
   For a stacked PR (its body names a parent PR), set `--base` in step 2 to the parent's
   head and review only this PR's commits.
2. **Run the script.**
   ```bash
   python3 .agents/skills/python-review/scripts/check_package.py <pkg> --base origin/main --format json
   ```
   The script accepts a directory name or a tox factor. Use its JSON `package` field as
   `<pkg>` in the tox env names below. `--base` tags each finding `introduced` or
   `pre-existing`. Use `--base HEAD` for uncommitted changes, and drop it for an audit.
   The JSON `kind` field says which checks apply:
   - `instrumentor`: a `BaseInstrumentor` that starts spans, by wrapping library calls or
     by registering a callback or tracing processor with the library
   - `processor`: an OTel `SpanProcessor` that rewrites spans the library emits itself
   - `propagator`: a `BaseInstrumentor` that only carries context and starts no spans
   - `alias`: a shim that re-exports another instrumentor
   Confirm every test and runtime finding by reading its location, because those checks
   are grep-level. Dismiss a false positive in the report with the reason. Use `--all` to
   see whether a gap is repo-wide before you report it per package.
3. **Start the CI env in the background** while you read code:
   ```bash
   cd python && uvx --with tox-uv tox run -e py314-ci-<pkg>
   ```
   Envs are `py3{10,14}-ci-<pkg>` and `py3{10,14}-ci-<pkg>-latest`. Also run `-latest`
   when the change is about library compatibility, or reads private library attributes or
   version-dependent APIs. Add `-r` when `test-requirements.txt`
   or `commands_pre` changed, because tox reuses a stale env otherwise. A failing env is a
   Critical finding.
4. **Read the library as ground truth.** After step 3 installs it, the library is under
   `python/.tox/py314-ci-<pkg>/lib/python3.14/site-packages/` (from the repo root). The
   distribution name in `test-requirements.txt` often differs from the import directory
   (`openai-agents` installs `agents/`), so read the directory from
   `site-packages/<dist>-*.dist-info/top_level.txt` or its `RECORD`. Before you flag a
   finding, confirm that the patched method exists with that signature and that the edge
   case can occur. Rate by user impact. A case the library's types rule out is at most Low.
5. **Review by dimension.** Load only the references that match what the change touches:

   | Change touches | Read |
   | --- | --- |
   | `python/tox.ini`, `pyproject.toml`, `test-requirements.txt`, `package.py`, `version.py` | [ci-and-packaging.md](references/ci-and-packaging.md) |
   | `tests/`, cassettes, `conftest.py` | [tests.md](references/tests.md) |
   | attribute extraction, span kinds, messages, tools, tokens | [semconv.md](references/semconv.md) |
   | wrappers, `_instrument` or `_uninstrument`, streams, threads, errors, `TraceConfig` | [runtime.md](references/runtime.md) |

   A new package or a full audit reads all four.
6. **Prove what you can.**
   - Report the step 3 result whether it passed or failed.
   - For each introduced High or Critical finding about span output, and for each claim the
     PR makes about span output, run the `phoenix-verify` skill. Pass it one claim
     sentence, using before/after mode for a fix. Its step 7 output goes into the report
     verbatim. When Phoenix is unreachable, in any mode, record its `NOT VERIFIED` block
     and keep going. Headless mode never waits for Phoenix to start.
   - When the checkout under review is not the one holding this skill, run phoenix-verify's
     `scripts/span_tree.sh` from this skill's repository, not from the PR checkout.
   - When the change touches `gen_ai.*` attributes or `_genai_conversion.py`, run the
     `genai-conformance` skill for that provider.
7. **Report** with [report.md](references/report.md).

## Composing with other skills

- **Calls:** `phoenix-verify` (span proof), `genai-conformance` (gen_ai dual-write),
  `phoenix-cli` (ad hoc trace queries during proof).
- **Called by:** `python-canary-fix` (headless, focus on the changed package, after its
  fix passes tox), and `issue-triage`, which reads the references as the requirement
  checklist and may run the script with `--all`. `phoenix-verify` points here for the
  review after a fix is proven.
- **Adding a convention.** If a rule can be checked mechanically, add a `@check` to the
  script, give it a heading in the matching reference, and run `--validate`. Write prose
  only for rules that need judgment.
