---
"@arizeai/openinference-semantic-conventions": minor
---

Generate the semantic convention constants from the machine-readable registry in `semantic-conventions/`. Existing exports are unchanged. Adds `COMPLETION_TEXT`, `EMBEDDING_INVOCATION_PARAMETERS`, `LLM_CHOICES`, and `PROMPT_TEXT`; the `AUDIO_*` attributes on `SemanticConventions`; `OpenInferenceSpanKind.UNKNOWN`; `LLMSystem.XAI` and `LLMSystem.DEEPSEEK`; and the entries the prefix and postfix helper objects were missing (for example `LLMAttributePostfixes.choices`).
