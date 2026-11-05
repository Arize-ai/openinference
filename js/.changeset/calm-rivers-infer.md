---
"@arizeai/openinference-instrumentation-openai": patch
---

Detect Doubleword from the request host, so an OpenAI client pointed at its OpenAI-compatible endpoint (`https://api.doubleword.ai/v1`) records `llm.provider = doubleword` instead of falling back to `openai`. Adds `api.doubleword.ai` → `doubleword` to `HOST_SUFFIX_TO_PROVIDER`.
