# Factory assessment

Coordinators post this on their PR with `add_issue_comment` right after opening it, so a reviewer can pick a review depth before reading the diff. It complements the PR body, which describes the change but never grades it. Keep each line to one sentence, with nothing outside the template. Post it once per PR. When a later change makes a line wrong, edit that comment with `update_issue_comment`; never post a second assessment.

```markdown
## Factory assessment

**Fit:** <implements the issue fully | partially, and what is left>
**Evidence:** <the failing test before the change, whether it uses an existing cassette, a hand-written one or a mock transport, and the commands that passed after it>
**Risk:** side effects <none/low/medium/high>, performance <…>, backwards compatibility <…>, from the language reviewers
**Review depth:** <glance | focused | deep>, because <reason>
```

Review depth:

- **glance**: only docs, tests or comments change.
- **deep**: the change touches a package's public API, the semantic conventions, `openinference-instrumentation` or `openinference-core` (which every instrumentor depends on), TraceConfig masking, or any risk is medium or high.
- **focused**: everything else.
