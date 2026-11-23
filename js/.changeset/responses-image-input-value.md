---
"@arizeai/openinference-instrumentation-openai": patch
---

fix: leave the base64 `result` of `image_generation_call` items out of `input.value` on Responses spans, when a later turn replays an earlier turn's output as input. The request sent to OpenAI is unchanged. The image was already recorded on the `output.images.*` of the span that generated it.
