---
"@arizeai/openinference-genai": minor
---

Map `gen_ai.operation.name` to OpenInference span kinds. Control-flow operations `"create_agent"`, `"invoke_agent"`, and `"plan"` map to `AGENT`, and `"invoke_workflow"` maps to `CHAIN`. Terminal (concrete leaf) operations are classified by operation name: `"chat"`, `"text_completion"`, and `"generate_content"` map to `LLM`; `"embeddings"` to `EMBEDDING`; `"retrieval"` to `RETRIEVER`; and `"execute_tool"` to `TOOL`.

An explicit agent identity (a `gen_ai.agent.*` attribute) classifies a span as `AGENT`, but only for control-flow or unrecognized operations — on terminal operations the agent identity is treated as context and does not change the kind. `gen_ai.agent.name` is now mapped to `agent.name` via the new `mapAgentAttributes` helper.

The minimum supported `@opentelemetry/semantic-conventions` peer dependency is now `1.41.1` (the first version exporting the `invoke_workflow` operation value). `"plan"` is matched by literal until a constant is published, since it was added in the OTel GenAI semantic conventions ([semantic-conventions-genai#97](https://github.com/open-telemetry/semantic-conventions-genai/pull/97)) but is not yet released.

This reclassifies some spans that previously fell through to `LLM` (e.g. an `invoke_agent` span without agent attributes now becomes `AGENT`). `"plan"` is a value added in the OTel GenAI semantic conventions ([semantic-conventions-genai#97](https://github.com/open-telemetry/semantic-conventions-genai/pull/97)) that is not yet released, so it is matched by literal until a constant is published. The minimum supported `@opentelemetry/semantic-conventions` peer dependency is now `1.41.1` (the first version exporting the `invoke_workflow` operation value).
