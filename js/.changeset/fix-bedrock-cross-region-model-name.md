---
"@arizeai/openinference-instrumentation-bedrock": patch
---

fix(bedrock): handle cross-region inference model IDs in extractModelName

Cross-region model IDs (e.g., `us.anthropic.claude-haiku-4-5-20251001-v1:0`) have a region prefix before the vendor, resulting in 3 dot-separated segments instead of 2. The previous code used `parts[1]` which returned the vendor name (`"anthropic"`) instead of the actual model name. Changed to `parts[parts.length - 1]` to always pick the last segment (the model name) regardless of prefix count.
