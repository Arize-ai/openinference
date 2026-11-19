# Context Attributes

Context attributes describe the request a span belongs to rather than the operation itself: which session and user it
serves, the application's metadata and tags, and the prompt template in use. Applications set them once, through the
instrumentation context API, and OpenInference instrumentation attaches them to every span it creates while that
context is active. They are not set by hand on each span.

Spans of every kind carry context attributes, including [LLM](../spans/llm.md),
[Embedding](../spans/embedding.md), and [Decision](../spans/decision.md) spans.

| Attribute | Type | Description |
| --- | --- | --- |
| `session.id` | String | Unique identifier for the session, such as a multi-turn conversation |
| `user.id` | String | Unique identifier for the user |
| `metadata` | JSON String | Key–value metadata associated with the request |
| `tag.tags` | List of strings | Tags for categorizing the span |
| `llm.prompt_template.template` | String | The prompt template used to generate the LLM input |
| `llm.prompt_template.variables` | JSON String | Key–value pairs applied to the prompt template |
| `llm.prompt_template.version` | String | Version identifier for the prompt template |

See the [Semantic Conventions](./semantic-conventions.md#context) for the full definitions.

## Setting Context Attributes

OpenInference SDKs provide an API for setting context attributes on the OpenTelemetry context. Instrumentation
copies the attributes in the active context onto each span when the span starts.
