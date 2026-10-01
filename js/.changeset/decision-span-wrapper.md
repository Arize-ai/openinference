---
"@arizeai/openinference-core": minor
---

Add decision span helpers: the `traceDecision` wrapper, which creates spans with the `DECISION` OpenInference span kind, and `getDecisionAttributes`, which builds the `decision.*` model identification (`system`, `provider`, model names) and token count attributes with the same semantics as `getLLMAttributes`.
