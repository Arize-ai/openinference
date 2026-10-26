# Bug fix rubric (starter - tune from real runs)

The session ends in exactly one of two states. Grade the criteria for the state it reached, from the PR and the session transcript.

## State A: draft PR opened

1. The transcript shows a test added in this PR failing for the reason the issue reports before the fix and passing after it.
2. Every test added in this PR drives real code, the SDK client through the instrumentor against a cassette or mock transport or the function that holds the logic, and would fail if the fix were reverted. None mocks the instrumentor, its wrappers or the SDK method it wraps, builds spans by hand, or silences warnings so fake payloads pass.
3. The diff is confined to the bug's path; no unrelated refactors, formatting churn, or comments that narrate the change.
4. For every touched Python package, `tox run -e ci-<env>` ran and passed in the transcript; for a change to `python/openinference-instrumentation/` or `python/openinference-semantic-conventions/`, that is `ci-instrumentation` or `ci-semconv` plus the instrumentors the issue names. For every touched TypeScript package, its tests and type check ran and passed, and so did `pnpm run lint` and `pnpm run fmt:check`; for a change to `openinference-core`, `openinference-semantic-conventions` or `openinference-genai`, `pnpm run -r test` and `pnpm run type:check` ran and passed.
5. Every language the diff touches (Python, TypeScript) got a review pass, shown by the reviewer's final report quoted in the transcript. Each blocking finding is fixed or, if disputed, listed in the PR body.
6. No file under `.github/`, `.claude/`, `.agents/` or `factory/`, no `AGENTS.md`, `CLAUDE.md` or `CHANGELOG.md`, and neither `release-please-config.json` nor `.release-please-manifest.json` is modified.
7. Changed `js/packages/` packages have a changeset; no package version is edited by hand; no `pyproject.toml` pins a production dependency to a pre-release of the instrumented library (OpenTelemetry's `0.NNbN` beta packages are exempt); no public API breaks.
8. The PR is a draft against `main` from branch `factory/issue-<number>`, with a conventional-commit `fix` title scoped to the package.
9. The PR body starts with `resolves #<number>`, its Verified line names the commands that ran, and its Not verified line names what the sandbox couldn't run, including any hand-written cassette that a live recording should confirm and, when the fix changes emitted spans, the trace output a reviewer should check.
10. Exactly one assessment comment on the PR follows `factory/assessment.md`, matches the PR's final state, and takes its risk ratings from the reviewers' risk lines.

## State B: not reproduced, or out of scope

1. The reproducer's attempts are concrete: the tests or commands that were run, and their output. Or the bug is only in Java, Go or the spec, and the comment says so.
2. No branch was pushed and no PR was opened.
3. One comment on the issue states what was tried and what specific information would let someone reproduce it, or that the factory doesn't take that language yet.
