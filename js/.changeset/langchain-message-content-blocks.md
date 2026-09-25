---
"@arizeai/openinference-instrumentation-langchain": patch
"@arizeai/openinference-instrumentation-langchain-v0": patch
---

Keep the content blocks of multimodal LangChain messages on spans: text and image blocks (OpenAI-style `image_url` and LangChain standard `image` blocks, given as a url or base64 data) are recorded under `message.contents` instead of being dropped.
