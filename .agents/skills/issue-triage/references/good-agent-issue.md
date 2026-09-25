# Policy: good-agent-issue

**Gate label:** `good-agent-issue` — GitHub defines it as "Self-contained issue
for an autonomous coding agent. Applying it triggers claude-implement-issue".

> **Off by default. Applying this label runs code.**
> `.github/workflows/claude-implement-issue.yml` fires on the `labeled` event
> for exactly this label: it assigns the issue, adds `agent-in-progress`, runs
> Claude Code against the repo and opens a pull request. Every false positive
> costs a paid run and a PR someone has to close. Enable this policy only when
> the caller passes `policy: good-agent-issue`; otherwise stage 8 never writes
> this label, and the Hard limits in SKILL.md forbid it everywhere else.
>
> One mechanic to know: the scheduled triage workflow writes with
> `GITHUB_TOKEN`, and GitHub does not start workflows from events that token
> creates. A label this policy applies therefore does **not** fire
> `claude-implement-issue` by itself — a maintainer re-applies it to start the
> agent. Report every issue you gate here so that hand-off is visible.

## Audience

An autonomous coding agent with the full repo checked out, the language
toolchains, and the reviewer skills — but **no API keys** for any provider, no
way to record a cassette, no one to ask a design question, and no judgment
about what a maintainer would prefer.

That sets the bar higher than `good first issue` on specification and lower on
reading: the agent can read the whole package in seconds, but it cannot invent
requirements, cannot choose between two reasonable attribute names, and cannot
run a live provider call.

## Qualifies when all six hold

| Condition | How to check |
| --- | --- |
| **Specified to a test** | Passed stage 2 **and** the expected behavior is concrete enough to become a unit-test assertion: a named attribute with a stated value, a stated span kind, an exception that must not propagate. "Should look right in Phoenix" does not qualify |
| **Scoped** | `size:S` or `size:M`. Never `size:L` |
| **Contained** | Exactly one `language: *` label and exactly one `instrumentation: *` label (or `c/core` alone). No `c/semcov`, no `c/genai` mapping decisions |
| **No design latitude** | The fix follows a pattern that already exists in the repo, and the issue or the stage 6 comment names where (`Pattern: …`). If the implementer must choose an attribute name, a span boundary or an API shape, it does not qualify |
| **Requirements visible** | If the issue adds spans or attributes, the Requirements checklist from stage 4b is in the body. The agent has to be told to keep suppression, context attributes and masking working, because the workflow's review will check them |
| **Testable offline** | The fix can be verified with existing cassettes, mocks or pure-function tests. If the stage 6 investigation found that a new recording against a live provider is needed, it does not qualify |

Good shapes: an `or 0` guard on a token field; a dropped content-part type in
a message extractor; a dead alias removal (`cleanup`); a test renamed to match
what it exercises; an attribute already extracted in the non-streaming path
added to the streaming accumulator with the same helper.

## Never label when any one holds

- **`assignees` is non-empty** or it carries `agent-in-progress` — someone or
  something is already on it
- it carries `blocked`, `needs information`, `cannot reproduce`, `duplicate`,
  `stale`, `wontfix`, `invalid`, `security`, `good first issue` or
  `good student issue` (do not take a human's issue away from them — one
  implementer per issue)
- it is `new instrumentation`, or touches `spec/`
- it spans two languages or two instrumentor packages
- it is a `documentation` or `question` issue — prose judgment is the wrong job
- it names a customer, an internal ticket, or a Slack thread the agent cannot
  read
- it is an epic, or it already shipped
- the body states a blocker in prose (`Blocked by #NNNN`, "after X lands")

## Regression

On the regression pass, never remove `good-agent-issue` from an issue that
carries `agent-in-progress` or has an assignee — the workflow is mid-flight or
done. Otherwise re-judge as SKILL.md stage 8 describes.
