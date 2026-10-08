# Gate classification path

Stage 8 may have up to three policies in effect. Their conditions overlap, so
deciding them one at a time invites inconsistency — an issue gated for a
student but not a newcomer with no statable reason. Walk this path once per
issue instead; each step names the labels it reads. A step that says **stop**
means no gate applies and the remaining steps are skipped.

Policies stay the source of truth for their own criteria. This path orders
them; when they disagree with it, fix the path.

## Step 0 — Is there anything to pick up?

Read: `assignees`, `agent-in-progress`, the already-shipped finding.

- Assigned, or `agent-in-progress` → **stop**. Someone or something has it.
  (Regression pass: leave existing gates alone — assignment after gating is the
  label working.)
- Already shipped, fully → **stop**. Comment with evidence; let a maintainer
  close it.

## Step 1 — Is it startable?

Read: `needs information`, `blocked`, `cannot reproduce`, `duplicate`, `stale`,
`wontfix`, `invalid`, `security`, and the body for a prose blocker.

- Any of those present, or a `Blocked by #NNNN` / "after X lands" line → **stop**.
  Recommend `blocked` in the report if the label is missing.
- `security` → **stop**. Not a teaching or automation target.

## Step 2 — Is it one change in one place?

Read: `language: *`, `instrumentation: *`, `c/core`, `c/semcov`,
`new instrumentation`, the epic finding.

- Epic (`☂️`, checklist body) → **stop**.
- `new instrumentation` → **stop**. Too big for every gate.
- Touches `spec/` or `c/semcov` → **stop**. Needs maintainer agreement first.
- Two `language: *` labels, or two `instrumentation: *` labels → **stop**.
  Parity and cross-package work fail every gate's containment rule.
- Exactly one language and at most one package (or `c/core` alone) → continue.
- Anything else — no `language: *` label and not `c/core` (a `c/ci` or
  `examples` issue, a docs-site issue) → **stop**. Every policy's containment
  rule needs a language to stand in.

## Step 3 — Is it a code change?

Read: stage 1 type.

| Type | Continue for |
| --- | --- |
| `bug`, `enhancement`, `cleanup` | all three gates |
| `documentation` | `good first issue` only |
| `question`, untyped | **stop** |

## Step 4 — Are the requirements visible?

Read: stage 4b — did the issue add surface, and is the Requirements checklist
in the body?

- Adds spans or attributes, checklist **absent** → **stop** for every gate, and
  add the checklist (stage 7 writes it). The issue may qualify on the next run.
- Adds surface, checklist present → continue.
- Does not add surface (value fix, guard, cleanup) → continue.

## Step 5 — Size, then split by audience

Read: `size:*`, and the stage 6 finding on whether a fix needs a new recording
against a live provider.

The tree below names every gate an issue *could* take. Apply only the ones
whose policy is in effect for this run **and** that step 3 left open — a
`documentation` issue takes `good first issue` alone, and a run with
`policy: none` applies nothing here.

```
size:S or size:M
├── needs no new live recording
│   ├── follows a named existing pattern, no naming or boundary decision
│   │   → good first issue + good student issue
│   │     (+ good-agent-issue only if that policy is enabled AND the
│   │      expected behavior is stated as a testable assertion; then
│   │      apply good-agent-issue INSTEAD of the other two — one
│   │      implementer per issue)
│   └── a bounded decision remains (which helper, which of two attribute
│       names the sibling instrumentor already uses)
│       → good first issue + good student issue; never good-agent-issue
└── needs a new live recording
    → good first issue + good student issue, and say so in the report;
      never good-agent-issue

size:L
├── one package, depth not breadth (contained algorithmic or accumulator
│   problem; a careful read of the instrumented library)
│   → good student issue only
└── anything else
    → no gate
```

When `good-agent-issue` is enabled and an issue qualifies for it, prefer it
over the human gates **only** when the human gates are already well stocked for
that package and language. The cohort and the newcomers need `size:S` issues
more than the agent does; the report's per-gate breakdown is where that
judgment is visible.

## Step 6 — Record the decision

For every issue that reached step 5, the report line names the gate(s) applied
or the single condition that stopped it:

```
#3499   gated: good first issue, good student issue — size:S, existing cassette
#3757   ungated at step 2: two packages (groq, together)
#3392   ungated at step 4: adds tool spans, checklist added this run
#3374   ungated at step 1: body names a customer; SENSITIVE
#2107   gated: good student issue only — size:L depth, single package (dspy)
```

An issue that stops at step 0 or 1 usually gets no line; the totals cover it.
