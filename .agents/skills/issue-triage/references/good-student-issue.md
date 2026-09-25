# Policy: good-student-issue

**Gate label:** `good student issue` — GitHub defines it as "A good issue for
students taking CS146S". Same label, colour and cohort as in `Arize-ai/phoenix`.

> **Temporary.** This policy serves one cohort of Stanford CS146S (class runs
> 2026-09-22 to 2026-12-15; contributions start week 2, 2026-09-28). When the
> class ends, remove it in three edits: delete this file, delete its row in the
> SKILL.md gate-policy table, and drop the `"good student issue"` example from
> the SKILL.md `description`. Stages 1–7 of the triage workflow do not depend on
> this policy, and the other two gates keep working.

## Audience

First- and second-year master's students with a rigorous CS systems core
(threading, memory models, C, algorithms), **zero** OpenInference or
OpenTelemetry familiarity, and a mentor. Most will know Python; some know
TypeScript; few know the Java tracing stack.

That sets the bar: reading one instrumentor package end to end — its wrappers,
its tests, and the library it patches — is the intended exercise, not a
disqualifier. Learning what a span, an attribute and a context are is part of
the course. Only context spanning packages or languages, or undocumented
history, is too much.

A mentor explains the codebase; they should not have to invent the
requirements. An under-specified issue is never a good student issue no matter
how appealing the idea.

**Students sit between the other two audiences.** They read more slowly than an
agent but can make a bounded design call with a mentor; they have more
scaffolding than an anonymous newcomer but less freedom to pick their own
timeline. The [classification path](gate-classification.md) walks the three
gates in one pass so they are decided consistently.

## Qualifies when all five hold

By this stage the triage workflow has already established most of this — the
gate reads its output rather than re-deriving it.

| Condition | How to check |
| --- | --- |
| **Specified** | Passed stage 2. An issue carrying `needs information` never qualifies |
| **Scoped** | `size:S` or `size:M`. See complexity note below |
| **Self-contained** | One `language: *` label and at most one `instrumentation: *` label, or `c/core` alone. No decision the student cannot make with a mentor |
| **Actionable** | Typed `bug`, `enhancement` or `cleanup` in stage 1. Questions, discussions and pure `documentation` never qualify — the course wants code |
| **Requirements visible** | If the issue adds spans or attributes, the Requirements checklist from stage 4b is in the body. The five requirements are exactly what the student is here to learn; they should read them on the issue, not in review |

**Complexity.** `size:S` and `size:M` qualify. `size:L` qualifies **only** if
the issue is confined to a single package and its difficulty is depth rather
than breadth — a contained problem a strong student can sit with, such as
bringing an instrumentor's streaming accumulator in line with its non-streaming
path, or a bug that needs the instrumented library read carefully. A `size:L`
that spans packages, languages, core or `spec/` does not qualify; that is
exactly the tribal-knowledge case the audience cannot absorb.

Prefer a spread of complexity: the cohort needs `size:S` issues to start on, not
only hard ones.

**Fixtures.** A fix that needs a new recording against a paid provider still
qualifies — the mentor can record the cassette or supply a key — but say so in
the report so the mentor knows before the student starts. Prefer issues whose
test can be written from an existing cassette or mock when choosing among
equals.

Candidate areas, **illustrative not exhaustive** — a well-specified,
self-contained issue anywhere in the repo can qualify:

- attribute-extraction bugs in a single Python instrumentor (dropped content
  parts, unguarded fields, wrong MIME type)
- a token-usage or reasoning attribute one instrumentor captures and a sibling
  does not, following the sibling's pattern
- a masking or context-attribute gap on one code path (`c/trace-config`,
  `c/context-attributes`, `c/suppress-tracing`) — the requirements are the
  course material
- a `cleanup` with a clear before and after: dead aliases, a mis-named test
- a contained JS instrumentor bug for students who know TypeScript

Spread gating across packages rather than emptying one area, so the cohort is
not 90 students queued on the Python OpenAI instrumentor.

## Never label when any one holds

However well-specified the issue otherwise is:

- **`assignees` is non-empty** — claimed is off the table, full stop (new
  candidates only; see "Assignment is not drift" in SKILL.md)
- it carries `agent-in-progress`, `good-agent-issue`, `blocked`,
  `needs information`, `cannot reproduce`, `duplicate`, `stale`, `wontfix`,
  `invalid` or `security`
- it is `new instrumentation` — a whole instrumentor is a quarter, not an issue
- it changes `spec/` or the semantic-conventions packages — those need
  maintainer agreement the student cannot obtain
- it spans two languages (parity work) or two instrumentor packages
- it is a third-party integration or package submission with promotional intent
- it names a customer, an internal ticket, or a Slack thread the student cannot
  read
- it is an epic — a `☂️` title or a checklist of sub-tasks rather than one change
- it already shipped

`good first issue` may coexist with this label; an issue good for a newcomer is
usually good for a student. `good-agent-issue` may not — one implementer per
issue.
