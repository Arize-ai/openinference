---
"@arizeai/openinference-core": minor
---

Add `hideLLMTools` trace config option (and `OPENINFERENCE_HIDE_LLM_TOOLS` env var) to mask the tool definitions advertised to the LLM (`llm.tools.*`). These attributes are also hidden when `hideInputs` is enabled.
