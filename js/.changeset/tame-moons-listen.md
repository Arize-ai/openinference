---
"@arizeai/openinference-instrumentation-openai": patch
---

Detect Meta AI, Z.ai, and MiniMax from the request host, so an OpenAI client pointed at one of their OpenAI-compatible endpoints records the real `llm.provider` instead of falling back to `openai`. Adds `api.meta.ai` → `meta`, `api.z.ai` → `zai`, and `api.minimax.io` / `api.minimaxi.com` / `api.minimax.chat` → `minimax` to `HOST_SUFFIX_TO_PROVIDER`. Matching stays suffix-based and anchored at a label boundary, so subdomains of these hosts resolve too and unrelated hosts are unaffected.
