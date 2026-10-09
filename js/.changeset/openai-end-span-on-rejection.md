---
"@arizeai/openinference-instrumentation-openai": patch
---

End the span with an ERROR status and an `exception` event when the request made by `chat.completions.create`, `completions.create`, `embeddings.create` or `responses.create` fails (for example a 404, 429, 5xx, timeout or abort). Previously the span was never ended, so failed calls were missing from traces.
