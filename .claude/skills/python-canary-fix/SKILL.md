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
- Invoked automatically for one package-scoped auto-fix job in `python-cron.yaml`

## Workflow

1. **Identify failures**: If the caller provides a run ID, use that exact run as the source of truth. Otherwise run `gh run list --workflow=python-cron.yaml --limit=5 --repo Arize-ai/openinference` to find the latest run, then `gh run view <id> --repo Arize-ai/openinference` and filter for `X` (failure markers) in the output. If the caller also provides a package token or explicit env list, stay within that scope only. When the caller already gave explicit failing envs, assume those jobs are complete and do not use `gh run watch` or wait for the overall workflow run to finish.

2. **Get failure logs**: If the caller already staged failing logs inside the repo workspace, use those first. Otherwise, for each failed job, run `gh run view <run_id> --repo Arize-ai/openinference --job <job_id> --log-failed` to get the actual error output.
   - If the staged log directory contains `*.lookup-error.txt` files or zero-byte `*.failed.log` / `*.full.log` files, treat the staging as incomplete and then query `gh run view` again for that env.

3. **Reproduce before deep forensics**: In package-scoped mode, rerun one failing `*-latest` env for that package before doing deeper upstream investigation. Only move on to PyPI metadata, wheel inspection, or changelog archaeology if the rerun and staged logs still leave the root cause ambiguous.

4. **Identify the root cause**: The canary cron tests against `*-latest` versions of upstream dependencies. Failures almost always mean an upstream package changed its API or behavior. Key signals:
   - Which package's test failed (the testenv name encodes the package, e.g. `py310-ci-openllmetry-latest`)
   - The assertion or import error in the log
   - What upstream dependency was upgraded (check tox.ini for the `-latest` env's `uv pip install -U` commands)

5. **Investigate the upstream change**: Search PyPI versions, GitHub releases, or changelogs for the upstream package to find what changed. Focus on attribute/API changes that would break our instrumentation.

6. **Draft and test the fix**: Modify the instrumentor code and tests. Before considering the work complete, run both the pinned and `-latest` tox environments for every package/root cause you changed to verify backward compatibility:
   - Run tox from the repo root with the explicit config path `python/tox.ini`
   - Run the exact relevant failing `-latest` env(s) from the canary run whenever practical, for example `uvx --with tox-uv tox -c python/tox.ini r -e py310-ci-<package>-latest -- -ra -x`
   - Run the matching pinned env(s), for example `uvx --with tox-uv tox -c python/tox.ini r -e ruff-mypy-test-<package>` (pinned deps, includes ruff formatting/linting + mypy + tests)
   - If multiple Python versions failed for the same package and your change could affect both, run each affected failing env
   - Do not open or update a PR unless the relevant tox envs pass locally, unless the failure is clearly transient or external
   - If ruff reformats any files, commit the formatting changes before proceeding.
   - In package-scoped mode, if the required fix touches shared Python code used by multiple packages or package directories outside the target package, stop and emit the workflow-requested shared-root-cause marker (for example by writing `shared_root_cause` to the status file named in the prompt). Do not open or update a PR in that case.
   - In manual debug mode, do not create commits, push branches, or open/update PRs. Stop after the package-local patch and verification summary.
   - In CI/package-scoped mode, use only repo-local scratch paths under the workspace. Do not use `/tmp` for helper scripts, extracted wheels, or temporary outputs.
   - When the prompt provides marker files inside the repo workspace, prefer writing the exact marker value with the `Write` tool. Repo-local shell redirection is acceptable, but do not write marker files outside the workspace.
   - Prefer one shell command per Bash invocation. Avoid compound shell commands with `&&`, `;`, output redirection, or long shell pipelines when `Read`, `Grep`, `Edit`, or a repo-local file can do the job.
   - Do not wrap tox commands in `cd`, `2>&1`, `| tail`, `| head`, `| grep`, or shell redirection. Run the tox command directly so the workflow can observe the true exit and output.
   - If a command is blocked by the sandbox or workflow permissions, write the workflow-requested `blocked_command` marker to the status file named in the prompt and stop after documenting the blocked command and the package-local fix candidate.

7. **Run /simplify**: Review the changed code for reuse, quality, and efficiency. Fix any issues found. In CI auto-fix mode, skip this unless the change is non-trivial and you still have time after verification.

8. **Run the Python code reviewer**: Run `/python-code-reviewer` against the changed package to verify it follows project conventions (test patterns, semantic conventions, CI config). In CI auto-fix mode, skip this unless the change is non-trivial and you still have time after verification.

9. **Check for existing PRs**: Before creating a new PR, search for open PRs that already address the same failure (`gh pr list --repo Arize-ai/openinference --search "<package>" --state open`). If one exists, update it instead of creating a duplicate. In package-scoped mode, only update a PR that clearly targets the same package canary fix. Skip this step in manual debug mode.

10. **Create a PR**: Branch, commit, push, and open a PR citing the upstream change that triggered the failure. Include the exact verification commands you ran and whether they passed. Skip this step in manual debug mode.

## Gotchas

- **Transient failures**: Some failures are caused by flaky network calls, temporary PyPI outages, or rate limits. If the error looks transient (timeout, connection reset, 503), check whether the same job passed in the previous cron run before investing time in a fix.
- **Shared root causes**: Multiple instrumentors may fail for the same reason (e.g., a core library like `opentelemetry-sdk` or `opentelemetry-semantic-conventions-ai` changed). In the general workflow, group related failures and fix them in a single PR rather than opening one PR per instrumentor. In package-scoped auto-fix mode, do not take that cross-package fix path automatically; emit the workflow-requested `shared_root_cause` marker instead.
- The tox token for a package strips `openinference-instrumentation-` and replaces hyphens with underscores (e.g., `openllmetry`, `llama_index`, `google_genai`).
- The `-latest` tox env typically just does `uv pip install -U <upstream-package>` on top of the pinned deps. Check `python/tox.ini` for the exact commands.
- Common failure patterns: removed/renamed span attributes, changed event formats, new required parameters, deprecated API removals.
- Always test against both pinned and latest deps to ensure backward compatibility, and treat passing local tox runs as a release gate for the auto-fix PR.
- If the caller staged per-env logs or scratch directories inside the repo, use those exact paths instead of inventing new `/tmp` locations or re-downloading artifacts.
