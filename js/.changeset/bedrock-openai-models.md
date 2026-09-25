---
"@arizeai/openinference-instrumentation-bedrock": patch
---

Record OpenAI models on Bedrock (gpt-oss, GPT-5.x, GPT-6) with `llm.system` "openai" instead of "amazon", and parse their Chat Completions body on InvokeModel and InvokeModelWithResponseStream so the spans get token counts and output messages.
