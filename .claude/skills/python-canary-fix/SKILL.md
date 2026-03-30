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

1. **Identify failures**: Run `gh run list --workflow=python-cron.yaml --limit=5 --repo Arize-ai/openinference` to find the latest run, then `gh run view <id> --repo Arize-ai/openinference` and filter for `X` (failure markers) in the output.

2. **Get failure logs**: For each failed job, run `gh run view <run_id> --repo Arize-ai/openinference --job <job_id> --log-failed` to get the actual error output.

3. **Identify the root cause**: The canary cron tests against `*-latest` versions of upstream dependencies. Failures almost always mean an upstream package changed its API or behavior. Key signals:
   - Which package's test failed (the testenv name encodes the package, e.g. `py310-ci-openllmetry-latest`)
   - The assertion or import error in the log
   - What upstream dependency was upgraded (check tox.ini for the `-latest` env's `uv pip install -U` commands)

4. **Investigate the upstream change**: Search PyPI versions, GitHub releases, or changelogs for the upstream package to find what changed. Focus on attribute/API changes that would break our instrumentation.

5. **Draft and test the fix**: Modify the instrumentor code and tests. Run both the pinned and `-latest` tox environments to verify backward compatibility:
   - `uvx --with tox-uv tox r -e ruff-mypy-test-<package>` (pinned deps)
   - `uvx --with tox-uv tox r -e py310-ci-<package>-latest -- -ra -x` (latest deps)

6. **Run /simplify**: Review the changed code for reuse, quality, and efficiency. Fix any issues found.

7. **Run the Python code reviewer**: Run `/python-code-reviewer` against the changed package to verify it follows project conventions (test patterns, semantic conventions, CI config).

8. **Create a PR**: Branch, commit, push, and open a PR citing the upstream change that triggered the failure.

## Gotchas

- The tox token for a package strips `openinference-instrumentation-` and replaces hyphens with underscores (e.g., `openllmetry`, `llama_index`, `google_genai`).
- The `-latest` tox env typically just does `uv pip install -U <upstream-package>` on top of the pinned deps. Check `python/tox.ini` for the exact commands.
- Common failure patterns: removed/renamed span attributes, changed event formats, new required parameters, deprecated API removals.
- Always test against both pinned and latest deps to ensure backward compatibility.
