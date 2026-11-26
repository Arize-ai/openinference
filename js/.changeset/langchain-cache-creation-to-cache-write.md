---
"@arizeai/openinference-instrumentation-langchain": patch
"@arizeai/openinference-instrumentation-langchain-v0": patch
---

Record LangChain usage metadata `input_token_details.cache_creation` as `llm.token_count.prompt_details.cache_write`, matching the Python instrumentation. Cache read and audio input token details were already recorded.
