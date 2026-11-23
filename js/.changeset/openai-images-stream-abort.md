---
"@arizeai/openinference-instrumentation-openai": patch
---

fix(openai): leave the status of an Images span unset when its stream is aborted, as Responses spans already do. A stream that ends without a completed image and was not aborted still ends with status ERROR.
