---
"@arizeai/openinference-instrumentation-openai": patch
---

Detect OCI Generative AI from the request host, so an OpenAI client pointed at its OpenAI-compatible endpoint (`https://inference.generativeai.<region>.oci.oraclecloud.com/openai/v1`) records `llm.provider = oracle` instead of falling back to `openai`. Adds `oci.oraclecloud.com` → `oracle` to `HOST_SUFFIX_TO_PROVIDER`; matching stays suffix-based and anchored at a label boundary, so every regional endpoint resolves and unrelated hosts are unaffected.
