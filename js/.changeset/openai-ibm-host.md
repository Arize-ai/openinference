---
"@arizeai/openinference-instrumentation-openai": patch
---

Detect IBM watsonx.ai from the request host, so an OpenAI client pointed at its OpenAI-compatible endpoint (`https://<region>.ml.cloud.ibm.com`) records `llm.provider = ibm` instead of falling back to `openai`. Adds `ml.cloud.ibm.com` → `ibm` to `HOST_SUFFIX_TO_PROVIDER`.
