---
"@arizeai/openinference-instrumentation-bedrock": patch
---

Record Bedrock Nova `invoke_model` tool definitions verbatim on `tool.json_schema`, keeping the Converse tagged-union envelope (`toolSpec`/`systemTool`/`cachePoint`) instead of unwrapping to the inner `toolSpec` body and dropping non-`toolSpec` members. This matches the Converse extractor and the OpenAI/Anthropic instrumentors (which all record the whole provider-native tool element), preserves `systemTool`/`cachePoint` entries, and keeps the recorded schema replayable against the Converse API.
