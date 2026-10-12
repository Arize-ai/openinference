# Body formatting

How to restructure an unreadable issue body (stage 7) without changing what the
reporter said, and how to write the triage block that stages 2 and 4 feed.

## The one rule

**The reporter's words stay verbatim.** Reorganise, label and quote them; never
paraphrase, summarise, correct or delete them. Their text is the evidence for
the bug. If restructuring would require rewording, stop and leave the body
alone.

Everything you add is clearly yours, inside the triage block below. A reader
must always be able to tell reported text from triage text.

## Target shapes

Follow the repo's issue templates, which are the house shape. They use bold
paragraph headings, not `###`.

**Bug** — `.github/ISSUE_TEMPLATE/bug_report.md`

```markdown
**Describe the bug**

**To Reproduce**

**Expected behavior**

**Additional context**
```

The template's *Screenshots* and *Desktop* sections rarely apply to an
instrumentor; fold a pasted span or trace under **Describe the bug** and an OS
or version line under **Additional context**. Package and library versions
belong under **Additional context** unless the reporter already put them in a
heading of their own — then keep that heading.

**Enhancement** — `.github/ISSUE_TEMPLATE/feature_request.md`

```markdown
**Is your feature request related to a problem? Please describe.**

**Describe the solution you'd like**

**Describe alternatives you've considered**

**Additional context**
```

Include only the headings you can actually fill from the reporter's text. An
empty heading is worse than a missing one — it looks answered.

Many issues here use `## Summary` / `## Problem` / `## Proposed fix` headings
instead of the templates. That is readable; leave it.

## Recipe

1. Read the current body:
   `gh issue view <n> --repo Arize-ai/openinference --json body --jq .body`
2. Decide whether it needs work at all. Skip the restructure if it is already
   scannable. The triage block may still be appended (below).
3. Map each piece of existing text to a heading, moving it **unchanged**. Keep
   code blocks, stack traces, span dumps, JSON and images exactly as they are,
   fenced.
4. Text that fits no heading goes last under **Additional context**. Never drop
   it.
5. Put anything you contribute — missing-info requests, the Requirements
   checklist, pointers — inside the triage block.
6. Write the result to a file and apply it:
   `gh issue edit <n> --repo Arize-ai/openinference --body-file <file>`

Compose in a file rather than inline: bodies contain backticks, quotes and
newlines that do not survive shell quoting.

## The triage block

Append your own content in exactly this form, at the end of the body:

```markdown
<!-- triage:begin -->
---
**Triage notes**

- Missing: expected behavior — which attribute should carry the reasoning content, and on which span?
- Missing: the instrumented library version (`pip show groq`)

**Requirements** — every instrumentor must (see `CLAUDE.md` → Requirements):

- [ ] **Suppress tracing** — …
<!-- triage:end -->
```

The markers make stage 7 idempotent, which matters because triage runs
repeatedly over the same backlog:

- If `<!-- triage:begin -->` is already present, **replace the whole block**
  between the markers. Never append a second one.
- Everything outside the markers is reporter text or a previous restructure —
  leave the reporter's portions untouched.
- Never put triage content outside the markers, and never put reporter text
  inside them.
- Replacing the block **re-derives** its content from the current stage 2 and
  stage 4 verdicts. If the reporter has since supplied what was missing, the
  "Missing:" bullets go away; if a maintainer ticked checklist boxes, keep
  their ticks — copy the checklist rows from the existing block rather than the
  template when the rows match.

Keep the block short. The Requirements checklist text is in
`instrumentor-requirements.md`; investigation findings belong in a stage 6
comment, not in the body.

## When not to touch a body

- It is already clear, even if terse, **and** stages 2 and 4 produced nothing
  for the triage block.
- Restructuring would need rewording to make sense.
- The body is a single coherent paragraph or a filled-in template already.
- The only problem is missing information — add `needs information` (stage 2)
  and note the gap in the triage block; do not invent structure around nothing.
- The issue is a question or discussion rather than a work item.
- The issue carries `agent-in-progress`. An agent is reading it.
- The body contains what looks like a leaked credential or personal data.
  Report it (SKILL.md stage 4) and leave the body for a maintainer; a body edit
  from triage would bury the evidence in the edit history without removing it.
