---
"@arizeai/openinference-instrumentation-openai": patch
---

Record the output and token counts of OpenAI Responses streams that end with `response.incomplete` or `response.failed`. Previously only `response.completed` was kept, so the span ended with an error status and no output or usage. A failed response still sets an error status; an incomplete one (for example `max_output_tokens`) is recorded like a completed one.
