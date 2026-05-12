---
name: python-canary-fix
description: Investigate and propose fixes for Python canary cron failures in the openinference repo. Use when the user mentions Python canary failures, Python cron failures, or when the auto-fix CI job reports Python instrumentation canary issues.
---

# Python Canary Fix

Investigate failures in the Python canary cron (`python-cron.yaml`) workflow and propose a fix for Python instrumentation packages.

## When to Use

- User mentions Python canary cron failures
- A scheduled trigger reports canary failures
- User asks to investigate `*-latest` test environment failures
- Invoked automatically by the `auto-fix` job in `python-cron.yaml`

## Workflow

Each failing package is investigated and fixed **independently**, with its own branch and its own PR. Do not bundle fixes for multiple packages together, even when they share an apparent root cause — keep PRs scoped to a single instrumentor so they can be reviewed, reverted, and released independently.

### Step 1 — Enumerate failing packages

1. **Identify failures**: Run `gh run list --workflow=python-cron.yaml --limit=5 --repo Arize-ai/openinference` to find the latest run, then `gh run view <id> --repo Arize-ai/openinference` and filter for `X` (failure markers) in the output.
2. **Build a per-package list**: Extract the package token from each failed testenv name (e.g. `py310-ci-openllmetry-latest` → `openllmetry`). Track each package as a separate work item — one TodoWrite entry per package is a good fit. Deduplicate the list so each package appears at most once, even if multiple testenvs failed for it. In particular, **Python version variants (e.g. `py310-ci-<package>-latest` and `py314-ci-<package>-latest`) collapse into a single investigation** — do not open separate work items or PRs per Python version. If only some Python versions failed, note that in the investigation, but still fix the package once.
3. **Filter transient failures**: Some failures are flaky (network, PyPI outages, 503s). For each package, if the error looks transient, check whether the same job passed in the previous cron run before investing time. Drop transient ones from the list.
4. **Drop packages already being fixed**: For each remaining package, run `gh pr list --repo Arize-ai/openinference --search "<package>" --state open` once up front. If an open PR already addresses the canary failure for that package, drop it from the list — do not re-investigate or open a duplicate PR. If the existing PR is stale or incomplete, update it in place rather than starting over.

### Step 2 — Per package: investigate, fix, and open a PR

Repeat the following for **each** remaining package, fully completing the cycle (including the PR) before moving on to the next package. Use a fresh branch per package.

1. **Get failure logs**: `gh run view <run_id> --repo Arize-ai/openinference --job <job_id> --log-failed`.
2. **Identify the root cause**: The canary cron tests against `*-latest` versions of upstream dependencies. Failures almost always mean an upstream package changed its API or behavior. Key signals:
   - The assertion or import error in the log
   - What upstream dependency was upgraded (check `python/tox.ini` for the `-latest` env's `uv pip install -U` commands)
3. **Investigate the upstream change**: Search PyPI versions, GitHub releases, or changelogs for the upstream package to find what changed. Focus on attribute/API changes that would break our instrumentation.
4. **Decide: backward-compatible fix vs. drop old support**: Default to a backward-compatible fix that keeps the current minimum supported version of the upstream package. **But** if the upstream change is a significant incompatibility — e.g. it forces gnarly type-checking branches, runtime version sniffing in many places, or large parallel code paths — it is acceptable (and often preferable) to bump the package's minimum supported upstream version, drop support for older versions, and ship the fix as a major-version bump of the instrumentor. When you choose this path:
   - Update the package's minimum version constraints (e.g. `pyproject.toml` dependency pins, `tox.ini` pinned env) to the new floor.
   - Remove the now-dead compatibility branches rather than leaving them behind.
   - Use a conventional commit with `!` (e.g. `fix(<package>)!: drop support for <upstream> < X.Y`) so release-please cuts a new **major** version of the instrumentor.
   - Call out the dropped support clearly in the PR body so reviewers and downstream users see the breaking change.
5. **Draft and test the fix**: Modify only this package's instrumentor code and tests. Run both the pinned and `-latest` tox environments to verify the fix:
   - `uvx --with tox-uv tox r -e ruff-mypy-test-<package>` (pinned deps, includes ruff formatting/linting + mypy + tests)
   - `uvx --with tox-uv tox r -e py310-ci-<package>-latest -- -ra -x` (latest deps)
   - If ruff reformats files, include those edits in the same package commit (don't split formatting into a separate PR).
   - Both envs must pass. For drop-old-support fixes, you've raised the pinned floor — the pinned env now exercises the new minimum supported upstream version, which is expected.
6. **Run /simplify**: Review the changed code for reuse, quality, and efficiency. Fix any issues found.
7. **Run the Python code reviewer**: Run `/python-code-reviewer` against the changed package to verify it follows project conventions (test patterns, semantic conventions, CI config).
8. **Final duplicate-PR check**: Right before opening the PR, re-run `gh pr list --repo Arize-ai/openinference --search "<package>" --state open`. The Step 1.4 filter caught duplicates that existed at the start of the run, but one may have been opened during steps 2.1–2.7. If a duplicate now exists, update it instead of creating a new PR.
9. **Create a PR for this package only**: Branch (e.g. `fix/canary-<package>`), commit, push, and open a PR citing the upstream change. The PR title and body should reference only this package; do not lump in other failing packages. Use `fix(<package>)!:` (or `feat(<package>)!:`) in the commit/PR title when you dropped support for older upstream versions, so release-please releases a new major version.

Once the PR is open, return to the package list and repeat for the next failing package.

## Gotchas

- **One PR per package, even with a shared root cause**: When several instrumentors fail for the same reason (e.g., a core library like `opentelemetry-sdk` or `opentelemetry-semantic-conventions-ai` changed), it can be tempting to bundle them. Don't — open a separate PR for each affected instrumentor. Cross-link the PRs in their descriptions if helpful, but keep the diffs scoped per package.
- **Branch hygiene**: Start each package's work from a clean `main` checkout so unrelated package changes don't bleed into the next PR.
- The tox token for a package strips `openinference-instrumentation-` and replaces hyphens with underscores (e.g., `openllmetry`, `llama_index`, `google_genai`).
- The `-latest` tox env typically just does `uv pip install -U <upstream-package>` on top of the pinned deps. Check `python/tox.ini` for the exact commands.
- Common failure patterns: removed/renamed span attributes, changed event formats, new required parameters, deprecated API removals, and type-system overhauls (renamed/restructured TypedDicts, Pydantic model changes) — the last category is often the trigger for the drop-old-support path.
- Test against both pinned and latest deps. If you've dropped support for older upstream versions (raising the pinned floor), the pinned env now represents the new minimum — that's expected.
- **When to drop old support**: If maintaining compatibility with the old upstream API requires significant ongoing complexity (e.g. parallel implementations, brittle isinstance/hasattr branches, type-check escape hatches, large try/except scaffolding), prefer dropping support over carrying the burden. Bump the floor, delete the old branches, and ship a major version with `!` in the commit type.
