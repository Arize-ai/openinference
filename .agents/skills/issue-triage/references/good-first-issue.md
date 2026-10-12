# Policy: good-first-issue

**Gate label:** `good first issue` — GitHub defines it as "Good for newcomers".

On by default. Remove this policy by deleting this file and its row in the
SKILL.md gate-policy table; stages 1–7 do not depend on it.

## Audience

An open-source contributor with working knowledge of the language, **no**
OpenInference familiarity, no mentor, and possibly no paid API key for the
provider the instrumentor wraps.

That sets the bar: reading one instrumentor package and its tests is the
intended exercise, not a disqualifier. Reading `python/DEVELOPMENT.md` or
`js/CLAUDE.md` to learn the five requirements is expected. Context that spans
packages, languages, or undocumented history is too much, and so is a test that
cannot be written without recording against a live provider.

## Qualifies when all five hold

By this stage the triage workflow has already established most of this — the
gate reads its output rather than re-deriving it.

| Condition | How to check |
| --- | --- |
| **Specified** | Passed stage 2. An issue carrying `needs information` never qualifies |
| **Scoped** | `size:S` or `size:M` (see complexity note) |
| **Contained** | Exactly one `language: *` label and at most one `instrumentation: *` label, or `c/core` alone. No `c/semcov` spec change |
| **Actionable** | Typed `bug`, `enhancement`, `cleanup` or `documentation` in stage 1 |
| **Requirements visible** | If the issue adds spans or attributes, the Requirements checklist is in the body (stage 4b). A newcomer should not discover `TraceConfig` from a review comment |

**Complexity.** `size:S` and `size:M` qualify. `size:L` never does here — the
audience has no mentor to absorb a design decision with.

**Fixtures.** Prefer issues whose test can be written from an **existing
cassette or mock** — an attribute-extraction fix over a response shape the
suite already records, a guard, a dropped field. An issue whose fix needs a new
recording against a paid provider still qualifies, but say so in the report, so
a maintainer can offer to record the cassette on the PR.

Spread gating across packages and languages rather than emptying one area, so
newcomers are not all queued on the Python OpenAI instrumentor.

## Never label when any one holds

However well-specified the issue otherwise is:

- **`assignees` is non-empty** — claimed is off the table (new candidates only;
  see "Assignment is not drift" in SKILL.md)
- it carries `agent-in-progress`, `good-agent-issue`, `blocked`,
  `needs information`, `cannot reproduce`, `duplicate`, `stale`, `wontfix`,
  `invalid` or `security`
- it is `new instrumentation` — a whole instrumentor is never a first task
- it changes `spec/` or the semantic-conventions packages — those need
  maintainer agreement first
- it spans two languages (parity work) or two instrumentor packages
- it is a third-party integration or package submission with promotional intent
- it is an epic — a `☂️` title or a checklist of sub-tasks rather than one change
- it already shipped

`good student issue` may coexist with this label. `good-agent-issue` may not —
one implementer per issue. The [classification path](gate-classification.md)
decides all three together.
