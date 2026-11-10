---
"@arizeai/openinference-instrumentation-langchain": patch
---

Set `llm.model_name` from LangChain's `ls_model_name` run metadata when a chat model's invocation params have no model, i.e. for `@langchain/google`'s chat models, which report their model only as `ls_model_name`. Without a model name, their LLM spans could not be priced by cost tracking. The model in the invocation params still takes precedence. The Python instrumentation already reads `ls_model_name`.
