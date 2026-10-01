---
"@arizeai/openinference-semantic-conventions": minor
---

Add `decision.model_name`, `decision.request.model_name`, `decision.response.model_name`, `decision.system`, and `decision.provider` attributes for identifying the model behind `DECISION` spans, plus `DecisionSystem` and `DecisionProvider` well-known value enums that alias the matching `LLMSystem` and `LLMProvider` values.
