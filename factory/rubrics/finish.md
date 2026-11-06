# Finish rubric (starter - tune from real runs)

The session ends in exactly one of two states. Grade the criteria for the state it reached, from the PR, its issue and the session transcript.

## State A: the pull request was finished

1. The PR body starts with `resolves #<number>`, and that issue describes the bug or feature the diff addresses. An issue the session filed follows the repo's issue template for its type, carries the template's labels and a language label, and copies no text from the PR.
2. The transcript shows the verdict graded by `factory/agents/pr-reviewer.md` § Verdict, with each language reviewer's final report quoted verbatim, and the verdict was not Close.
3. The transcript shows the PR's non-test changes taken out of the working tree, the PR's own added tests run there and failing for the reason the issue reports, and, for a bug, the reproducer called on that tree. Once only the PR's paths were restored, so the reproducer's test survived into the PR's commits, the transcript shows those tests passing. A PR test that passed without its change, or a test that still failed with it, was treated as a blocking finding; a `NOT_REPRODUCED`, or a feature with no test showing `main` lacks it, is named in the assessment.
4. Every blocking and non-blocking finding from the reviewers and the verification is fixed in the diff, or the PR body lists the ones the session disputes and why. The fixes follow the project's conventions, not the PR's.
5. Every test in the PR drives real code, the SDK client through the instrumentor against a cassette or mock transport or the function that holds the logic, and would fail if the change were reverted. None mocks the instrumentor, its wrappers or the SDK method it wraps, builds spans by hand, or silences warnings so fake payloads pass.
6. After the last fix, for every touched Python package, `tox run -e ci-<env>` ran and passed in the transcript; for a change to `python/openinference-instrumentation/` or `python/openinference-semantic-conventions/`, that is `ci-instrumentation` or `ci-semconv` plus the instrumentors the PR names. For every touched TypeScript package, its tests and type check ran and passed, and so did `pnpm run lint` and `pnpm run fmt:check`; for a change to `openinference-core`, `openinference-semantic-conventions` or `openinference-genai`, `pnpm run -r test` and `pnpm run type:check` ran and passed.
7. The session's commits were appended to the PR's branch: no amend, rebase, squash or force-push, and the contributor's commits are unchanged. Commit messages are conventional commits scoped to the package, and so is the PR title.
8. No file under `.github/`, `.claude/`, `.agents/` or `factory/`, no `AGENTS.md`, `CLAUDE.md` or `CHANGELOG.md`, and neither `release-please-config.json` nor `.release-please-manifest.json` was modified by the session; one the PR already changed is named in the assessment.
9. Changed `js/packages/` packages have a changeset; no package version is edited by hand; no `pyproject.toml` pins a production dependency to a pre-release of the instrumented library (OpenTelemetry's `0.NNbN` beta packages are exempt); no public API breaks.
10. The PR body has a Verified line naming the commands that ran and a Not verified line naming what the sandbox couldn't run, including any hand-written cassette and, when the change alters emitted spans, the trace output a reviewer should check.
11. Exactly one comment to the author lists what changed, grouped by finding, with the rule or convention behind each, and matches the final diff.
12. Exactly one assessment comment follows `factory/assessment.md`, matches the PR's final state, and takes its risk ratings from the reviewers' risk lines.

## State B: the pull request was closed

1. The transcript shows the verdict graded by `factory/agents/pr-reviewer.md` § Verdict as Close, or shows that Fit fails and the fix would be a rewrite, with the evidence.
2. One comment on the PR names the rule and tells the contributor what it means for them, in one paragraph.
3. The PR is closed, its issue exists, and no commit was pushed to the PR's branch. A Python or TypeScript bug issue is labeled `factory`; any other issue is not, and the comment says a maintainer picks it up.
