---
"@arizeai/openinference-core": patch
---

Honor the spec's OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS environment variable and replace hidden embedding vectors with "__REDACTED__" instead of dropping them, as the spec and the Python package do
