---
"@arizeai/openinference-semantic-conventions": minor
"@arizeai/openinference-instrumentation-langchain": patch
"@arizeai/openinference-instrumentation-langchain-v0": patch
---

Add `PROMPT` to the `OpenInferenceSpanKind` enum, aligning the JS package with the OpenInference spec and the Python semantic conventions. LangChain prompt template spans now correctly report `openinference.span.kind = "PROMPT"` instead of falling through to `"CHAIN"`.
