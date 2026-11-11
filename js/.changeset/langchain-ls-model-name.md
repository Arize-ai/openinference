---
"@arizeai/openinference-instrumentation-langchain": minor
---

Read the LangSmith params (`ls_provider` and `ls_model_name`) that LangChain chat models report in their run metadata, matching the Python instrumentation.

- `llm.model_name` now comes from the model the response reports in `llmOutput`, then `ls_model_name`, then the invocation params' `model_name` or `model`, which is the Python instrumentation's order. Chat models that report their model only as `ls_model_name`, e.g. `@langchain/google`'s, now get a model name, so cost tracking can price their spans. Blank names are skipped, and the model name no longer depends on the run having invocation params.
- `llm.provider` and `llm.system` are now set from `ls_provider`, using the Python instrumentation's mappings (e.g. `google_genai` → `google` / `vertexai`, `azure` → `azure` / `openai`) plus the JS-only `cerebras`, `mistral` and `watsonx`. Other providers are recorded as their lowercased `ls_provider`.
