# Report

## Severity

- **Critical**: wrong behavior for users, or CI fails.
- **High**: a required convention or test is missing.
- **Medium**: works, but departs from the repo's pattern.
- **Low**: style or a minor improvement.

In PR mode only `introduced` findings block. List `pre-existing` ones as follow-ups. When
`check_package.py --all` shows the same gap in many packages, say so, so the user files one
issue rather than one per package.

## Template

```
## python-code-reviewer: <pkg> (<kind>)<, PR #n>

| Severity | Finding | Location | Origin | Source |
| --- | --- | --- | --- | --- |
| Critical | <one sentence> | `path:line` | introduced | script `<check>` / judgment `<reference>#<heading>` |

Dismissed script findings: <check> at <location> because <reason>.

CI: `uvx --with tox-uv tox run -e py314-ci-<pkg>` -> <OK, or the failing command and error>
    (one line per env run, including `-latest`)
Phoenix: <phoenix-verify step 7 block, one per claim, or "no span-output claims">
Working well: <checks that passed and patterns worth keeping>
```

Drop the Origin column for an audit. In headless mode, stop after the template. In
interactive mode, ask whether to fix the introduced findings.
