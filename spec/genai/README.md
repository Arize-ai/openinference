# OpenInference ↔ OpenTelemetry GenAI Semantic Conventions

A comprehensive mapping between [OpenInference Semantic Conventions](../semantic_conventions.md) and [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). This document identifies where the two systems align, where they diverge, and what a convergence path looks like.

> **Status note:** The OTel GenAI semantic conventions are in [**Development** status](https://opentelemetry.io/docs/specs/semconv/gen-ai/) and subject to breaking changes. OpenInference is stable and widely deployed.

---

## Executive Summary

OpenInference and OTel GenAI solve the same problem — standardizing AI observability on top of OpenTelemetry — but they grew from different roots and make different trade-offs.

**Where they agree:** Both define span-based conventions for LLM inference, embeddings, retrieval, tool execution, and agent orchestration. Core concepts like message roles, tool calling fields, and token usage map cleanly between the two.

**Where they diverge:**
- **Message format** (high impact) — OpenInference flattens messages into indexed span attributes (`llm.input_messages.0.message.role`); GenAI stores structured JSON arrays (`gen_ai.input.messages`). This is the hardest gap to bridge.
- **Attribute naming** (medium impact) — Different namespaces and terminology (`llm.token_count.prompt` vs `gen_ai.usage.input_tokens`), but the mappings are mechanical.
- **Scope** — OpenInference covers cost tracking, reranking, guardrails, prompt templates, document models, and privacy configuration that GenAI does not define. GenAI covers metrics, streaming, structured events, and agent lifecycle that OpenInference does not define.

**Convergence outlook:** GenAI is the upstream OTel standard. OpenInference should adopt GenAI conventions where equivalent and contribute its unique capabilities upstream. Section 8 proposes a phased convergence plan.

---

## Table of Contents

1. [Span Kind / Operation Mapping](#1-span-kind--operation-mapping)
2. [Core Attribute Mapping](#2-core-attribute-mapping)
3. [Message Format Comparison](#3-message-format-comparison)
4. [Missing in OpenInference](#4-missing-in-openinference)
5. [Missing in GenAI](#5-missing-in-genai)
6. [Compatibility Issues](#6-compatibility-issues)
7. [Harmonious Values](#7-harmonious-values)
8. [Convergence Strategy](#8-convergence-strategy)

---

## 1. Span Kind / Operation Mapping

OpenInference uses a single [`openinference.span.kind`](../semantic_conventions.md#span-kinds) attribute with uppercase enum values. GenAI uses [`gen_ai.operation.name`](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) with lowercase operation strings and different span naming conventions.

| OpenInference Span Kind | GenAI `gen_ai.operation.name` | GenAI Span Name Pattern | OTel Span Kind | Notes |
|---|---|---|---|---|
| `LLM` | `chat` | `chat {gen_ai.request.model}` | `CLIENT` | OI uses one kind for all LLM calls; GenAI splits chat vs text_completion vs generate_content |
| `LLM` | `text_completion` | `text_completion {gen_ai.request.model}` | `CLIENT` | OI [`llm.prompts`](../llm_spans.md#completions) / `llm.choices` maps to text_completion |
| `LLM` | `generate_content` | `generate_content {gen_ai.request.model}` | `CLIENT` | Google-style multimodal generation |
| `EMBEDDING` | `embeddings` | `embeddings {gen_ai.request.model}` | `CLIENT` | Direct equivalent |
| `RETRIEVER` | `retrieval` | `retrieval {gen_ai.data_source.id}` | `CLIENT` | Similar concept; GenAI uses `gen_ai.data_source.id` in span name |
| `TOOL` | `execute_tool` | `execute_tool {gen_ai.tool.name}` | `INTERNAL` | Direct equivalent |
| `AGENT` | `invoke_agent` | `invoke_agent {gen_ai.agent.name}` | `CLIENT` | OI `AGENT` maps loosely; GenAI also has `create_agent` and agent-specific attributes |
| `CHAIN` | _(no equivalent)_ | | | OI-specific orchestration span |
| `RERANKER` | _(no equivalent)_ | | | OI-specific reranking span |
| `GUARDRAIL` | _(no equivalent)_ | | | OI-specific moderation span |
| `EVALUATOR` | _(no equivalent)_ | | | GenAI has [`gen_ai.evaluation.*`](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/) events but no span kind |
| `PROMPT` | _(no equivalent)_ | | | OI-specific prompt rendering span |
| _(no equivalent)_ | `create_agent` | `create_agent {gen_ai.agent.name}` | `CLIENT` | GenAI-specific agent creation operation |
| _(no equivalent)_ | `invoke_workflow` | `invoke_workflow {gen_ai.agent.name}` | `CLIENT` | GenAI-specific workflow invocation |

> **Sources:** OI span kinds are defined in [`semantic_conventions.md`](../semantic_conventions.md#span-kinds) and enumerated in [`SpanAttributes.OPENINFERENCE_SPAN_KIND`](../../python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py). GenAI operations are defined in the [GenAI span conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/).

---

## 2. Core Attribute Mapping

### 2.1 Model & Provider Identification

| OpenInference Attribute | GenAI Attribute | Notes |
|---|---|---|
| [`llm.model_name`](../semantic_conventions.md) | `gen_ai.request.model` | OI stores the model name; GenAI distinguishes request vs response model |
| _(no equivalent)_ | `gen_ai.response.model` | GenAI tracks the actual model that generated the response (important for aliases/fine-tuned models) |
| [`llm.system`](../semantic_conventions.md#system-and-model-identification) | `gen_ai.provider.name` | **Semantic overlap with differences.** OI `llm.system` = the AI product/vendor. GenAI `gen_ai.provider.name` = the provider. See [Provider Values](#73-provider-name-values) |
| [`llm.provider`](../semantic_conventions.md#system-and-model-identification) | _(partial overlap)_ | OI separates system (product) from provider (host). GenAI merges these into `gen_ai.provider.name` with composite values like `azure.ai.openai` |
| [`embedding.model_name`](../embedding_spans.md) | `gen_ai.request.model` | OI has a separate attribute for embedding models; GenAI uses the same `gen_ai.request.model` |
| [`reranker.model_name`](../semantic_conventions.md) | _(no equivalent)_ | OI-specific |

### 2.2 Token Usage

| OpenInference Attribute | GenAI Attribute | Notes |
|---|---|---|
| `llm.token_count.prompt` | `gen_ai.usage.input_tokens` | Direct mapping; different naming convention |
| `llm.token_count.completion` | `gen_ai.usage.output_tokens` | Direct mapping; different naming convention |
| `llm.token_count.total` | _(no equivalent)_ | GenAI does not define a total; consumers compute `input + output` |
| `llm.token_count.prompt_details.cache_read` | `gen_ai.usage.cache_read.input_tokens` | Direct mapping |
| `llm.token_count.prompt_details.cache_write` | `gen_ai.usage.cache_creation.input_tokens` | Direct mapping; different naming (`cache_write` vs `cache_creation`) |
| `llm.token_count.prompt_details.cache_input` | _(no equivalent)_ | OI tracks cached input tokens; GenAI does not |
| `llm.token_count.prompt_details.audio` | _(no equivalent)_ | GenAI does not break down token types beyond cache |
| `llm.token_count.completion_details.reasoning` | _(no equivalent)_ | GenAI does not track reasoning tokens separately |
| `llm.token_count.completion_details.audio` | _(no equivalent)_ | GenAI does not track audio output tokens separately |

> **Sources:** OI token attributes are defined in [`semantic_conventions.md`](../semantic_conventions.md#token-count-details). GenAI usage attributes are defined on [inference spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/).

### 2.3 Input/Output Values

| OpenInference Attribute | GenAI Attribute | Notes |
|---|---|---|
| [`input.value`](../semantic_conventions.md) | _(no equivalent)_ | OI captures a general serialized input for any span kind |
| `input.mime_type` | _(no equivalent)_ | OI tags the format of `input.value` |
| `output.value` | _(no equivalent)_ | OI captures a general serialized output for any span kind |
| `output.mime_type` | _(no equivalent)_ | OI tags the format of `output.value` |
| [`llm.input_messages`](../llm_spans.md) | `gen_ai.input.messages` | **Structural difference.** OI uses flattened span attributes; GenAI uses structured JSON. See [Message Format](#3-message-format-comparison) |
| `llm.output_messages` | `gen_ai.output.messages` | Same structural difference as above |
| _(no equivalent)_ | `gen_ai.system_instructions` | GenAI separates system messages from chat history |

### 2.4 Tool Calling

| OpenInference Attribute | GenAI Attribute | Notes |
|---|---|---|
| [`tool.name`](../semantic_conventions.md) | `gen_ai.tool.name` | Direct mapping |
| `tool.description` | `gen_ai.tool.description` | Direct mapping |
| `tool.json_schema` | _(in `gen_ai.tool.definitions`)_ | OI stores per-tool schema; GenAI stores all tool definitions as a single structured attribute |
| `tool.parameters` | _(in `gen_ai.tool.definitions`)_ | Same as above |
| [`tool.id`](../semantic_conventions.md) | `gen_ai.tool.call.id` | OI uses `tool.id` on TOOL spans; GenAI uses `gen_ai.tool.call.id` on execute_tool spans |
| [`tool_call.id`](../tool_calling.md) | `gen_ai.tool.call.id` | Direct mapping (in message context) |
| `tool_call.function.name` | tool_call `name` field in message | OI: flattened attribute; GenAI: field within structured message JSON |
| `tool_call.function.arguments` | tool_call `arguments` field in message | OI: flattened attribute; GenAI: field within structured message JSON |
| [`llm.tools`](../tool_calling.md#tool-definitions) | `gen_ai.tool.definitions` | OI: list of `{tool.json_schema}` flattened; GenAI: structured JSON array |
| _(no equivalent)_ | `gen_ai.tool.type` | GenAI distinguishes `function`, `extension`, `datastore` |
| _(no equivalent)_ | `gen_ai.tool.call.arguments` | GenAI has this as a span attribute on execute_tool spans |
| _(no equivalent)_ | `gen_ai.tool.call.result` | GenAI has this as a span attribute on execute_tool spans |

> **Sources:** OI tool calling is specified in [`tool_calling.md`](../tool_calling.md). GenAI tool conventions are in the [execute_tool span section](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/).

### 2.5 Embeddings

| OpenInference Attribute | GenAI Attribute | Notes |
|---|---|---|
| [`embedding.model_name`](../embedding_spans.md) | `gen_ai.request.model` | GenAI uses the same model attribute for all operations |
| `embedding.embeddings` | _(no equivalent)_ | GenAI does not capture embedding vectors |
| `embedding.text` | _(no equivalent)_ | GenAI does not capture embedding input text |
| `embedding.vector` | _(no equivalent)_ | GenAI does not capture embedding vectors |
| `embedding.invocation_parameters` | _(no equivalent)_ | GenAI uses `gen_ai.request.*` attributes instead |
| _(no equivalent)_ | `gen_ai.embeddings.dimension.count` | GenAI tracks output dimension count |
| _(no equivalent)_ | `gen_ai.request.encoding_formats` | GenAI tracks requested encoding formats |

### 2.6 Retrieval

| OpenInference Attribute | GenAI Attribute | Notes |
|---|---|---|
| [`retrieval.documents`](../semantic_conventions.md) | `gen_ai.retrieval.documents` | Both capture retrieved documents; different structural format |
| _(no equivalent)_ | `gen_ai.retrieval.query.text` | GenAI captures the query text |
| _(no equivalent)_ | `gen_ai.data_source.id` | GenAI identifies the data source |
| _(no equivalent)_ | `gen_ai.request.top_k` | GenAI has this as a first-class attribute on retrieval spans |

### 2.7 Session & User

| OpenInference Attribute | GenAI Attribute | Notes |
|---|---|---|
| [`session.id`](../semantic_conventions.md) | `gen_ai.conversation.id` | Similar concept — OI uses "session", GenAI uses "conversation". A session may span multiple conversations. |
| [`user.id`](../semantic_conventions.md) | _(no equivalent)_ | GenAI does not define a user identifier attribute |

### 2.8 LLM Request Parameters

| OpenInference Attribute | GenAI Attribute | Notes |
|---|---|---|
| [`llm.invocation_parameters`](../semantic_conventions.md) (JSON blob) | Individual `gen_ai.request.*` attributes | **Structural difference.** OI stores all params as a single JSON string; GenAI uses separate typed attributes |
| _(in JSON blob)_ | `gen_ai.request.temperature` | GenAI: first-class `double` attribute |
| _(in JSON blob)_ | `gen_ai.request.top_p` | GenAI: first-class `double` attribute |
| _(in JSON blob)_ | `gen_ai.request.top_k` | GenAI: first-class `double` attribute |
| _(in JSON blob)_ | `gen_ai.request.max_tokens` | GenAI: first-class `int` attribute |
| _(in JSON blob)_ | `gen_ai.request.frequency_penalty` | GenAI: first-class `double` attribute |
| _(in JSON blob)_ | `gen_ai.request.presence_penalty` | GenAI: first-class `double` attribute |
| _(in JSON blob)_ | `gen_ai.request.stop_sequences` | GenAI: first-class `string[]` attribute |
| _(in JSON blob)_ | `gen_ai.request.seed` | GenAI: first-class `int` attribute |
| _(in JSON blob)_ | `gen_ai.request.choice.count` | GenAI: first-class `int` attribute (number of completions) |

### 2.9 Response Metadata

| OpenInference Attribute | GenAI Attribute | Notes |
|---|---|---|
| _(no equivalent)_ | `gen_ai.response.id` | GenAI captures the response ID from the provider |
| _(no equivalent)_ | `gen_ai.response.model` | GenAI captures the actual model that generated the response |
| _(no equivalent)_ | `gen_ai.response.finish_reasons` | GenAI captures finish reasons as a `string[]` attribute |
| _(no equivalent)_ | `gen_ai.output.type` | GenAI captures the output type (`text`, `json`, `image`, `speech`) |

---

## 3. Message Format Comparison

This is the **most significant structural difference** between the two conventions.

### OpenInference: Flattened Span Attributes

OI represents messages as [flattened OTel span attributes](../llm_spans.md#attribute-flattening) with indexed prefixes:

```
llm.input_messages.0.message.role = "user"
llm.input_messages.0.message.content = "What's the weather?"
llm.input_messages.1.message.role = "assistant"
llm.input_messages.1.message.content = "Let me check."
llm.input_messages.1.message.tool_calls.0.tool_call.function.name = "get_weather"
llm.input_messages.1.message.tool_calls.0.tool_call.function.arguments = "{\"city\":\"Paris\"}"
llm.input_messages.1.message.tool_calls.0.tool_call.id = "call_123"
```

[Multimodal content](../multimodal_attributes.md) uses further nesting:

```
llm.input_messages.0.message.contents.0.message_content.type = "text"
llm.input_messages.0.message.contents.0.message_content.text = "Describe this image"
llm.input_messages.0.message.contents.1.message_content.type = "image"
llm.input_messages.0.message.contents.1.message_content.image.image.url = "https://..."
```

### GenAI: Structured JSON

GenAI represents messages as [structured JSON objects](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) stored as a single attribute or event body:

```json
[
  {
    "role": "user",
    "parts": [
      {"type": "text", "content": "What's the weather?"}
    ]
  },
  {
    "role": "assistant",
    "parts": [
      {
        "type": "tool_call",
        "id": "call_123",
        "name": "get_weather",
        "arguments": {"city": "Paris"}
      }
    ]
  },
  {
    "role": "tool",
    "parts": [
      {
        "type": "tool_call_response",
        "id": "call_123",
        "response": "Rainy, 57F"
      }
    ]
  }
]
```

### Key Structural Differences

| Aspect | OpenInference | GenAI |
|---|---|---|
| Storage format | Flattened span attributes with indexed keys | Structured JSON (attribute or event body) |
| Content model | `message.content` (string) or [`message.contents`](../multimodal_attributes.md) (typed array with `message_content.type`) | `parts` array with `type` discriminator |
| Text content key | `message.content` or `message_content.text` | `parts[].content` |
| Image content | `message_content.image.image.url` | `parts[].uri` or `parts[].content` (base64 blob) with `modality: "image"` |
| Audio content | Separate [`audio.url`](../semantic_conventions.md), `audio.mime_type`, `audio.transcript` | `parts[].content` (blob) or `parts[].uri` with `modality: "audio"` |
| Tool calls in messages | Nested under [`message.tool_calls.N.tool_call.*`](../tool_calling.md#tool-calls-in-messages) | Inline `parts` with `type: "tool_call"` |
| Tool results | Message with `role: "tool"` and [`message.tool_call_id`](../tool_calling.md#tool-results) | Message with `role: "tool"` and `parts[].type: "tool_call_response"` |
| System messages | Part of `llm.input_messages` with `role: "system"` | Separate `gen_ai.system_instructions` attribute |
| Finish reason | Not captured in messages | `finish_reason` field on each output message |
| Multiple choices | Not natively supported | Multiple objects in `gen_ai.output.messages` array |
| Reasoning content | Not supported | `parts[].type: "reasoning"` |
| Built-in tools | Not supported | `parts[].type: "server_tool_call"` and `"server_tool_call_response"` |
| Video content | Not supported | `parts[].modality: "video"` with URI or blob |
| File references | Not supported | `parts[].type: "file"` with `file_id` |

---

## 4. Missing in OpenInference

Conventions present in [GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) that OpenInference does not define.

### 4.1 Request Parameters as First-Class Attributes

GenAI defines individual typed attributes for common LLM request parameters. OpenInference stores these in a single [`llm.invocation_parameters`](../semantic_conventions.md) JSON string, which is flexible but not queryable at the attribute level.

| GenAI Attribute | Type | Description |
|---|---|---|
| `gen_ai.request.temperature` | double | Sampling temperature |
| `gen_ai.request.top_p` | double | Nucleus sampling parameter |
| `gen_ai.request.top_k` | double | Top-k sampling parameter |
| `gen_ai.request.max_tokens` | int | Maximum tokens to generate |
| `gen_ai.request.frequency_penalty` | double | Frequency penalty |
| `gen_ai.request.presence_penalty` | double | Presence penalty |
| `gen_ai.request.stop_sequences` | string[] | Stop sequences |
| `gen_ai.request.seed` | int | Seed for reproducibility |
| `gen_ai.request.choice.count` | int | Number of completions to generate |
| `gen_ai.request.encoding_formats` | string[] | Encoding formats for embeddings |

### 4.2 Response Metadata

| GenAI Attribute | Type | Description |
|---|---|---|
| `gen_ai.response.id` | string | Unique response identifier from the provider |
| `gen_ai.response.model` | string | Actual model that generated the response |
| `gen_ai.response.finish_reasons` | string[] | Why the model stopped generating |
| `gen_ai.output.type` | string | Output type: `text`, `json`, `image`, `speech` |

### 4.3 Metrics

GenAI defines [histogram metrics](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-metrics/) that OpenInference does not have:

| Metric | Unit | Description |
|---|---|---|
| `gen_ai.client.operation.duration` | seconds | End-to-end latency of GenAI operations |
| `gen_ai.client.token.usage` | tokens | Token usage histogram (by input/output type) |
| `gen_ai.client.operation.time_to_first_chunk` | seconds | Client-side time to first streaming chunk |
| `gen_ai.client.operation.time_per_output_chunk` | seconds | Client-side time per output chunk |
| `gen_ai.server.request.duration` | seconds | Server-side request duration |
| `gen_ai.server.time_per_output_token` | seconds | Time per output token (throughput) |
| `gen_ai.server.time_to_first_token` | seconds | Time to first token (TTFT) |

### 4.4 Events

| Event | Description |
|---|---|
| [`gen_ai.client.inference.operation.details`](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/) | Captures full request/response details as a structured event |
| `gen_ai.evaluation.result` | Captures evaluation results (score, label, explanation) |
| `gen_ai.client.operation.exception` | Captures exceptions during GenAI operations |

### 4.5 Content Types in Messages

GenAI's structured message format supports content types OI does not:

| Content Type | Description |
|---|---|
| `reasoning` | Model's intermediate reasoning/chain-of-thought |
| `server_tool_call` | Built-in tool execution (e.g., code interpreter) |
| `server_tool_call_response` | Response from built-in tool execution |
| `blob` | Binary content with modality and MIME type |
| `uri` | Reference to external content by URI |
| `file` | Reference to provider-hosted file by ID |

### 4.6 Agent Lifecycle Attributes

GenAI defines [agent-specific attributes](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) that OpenInference does not have:

| GenAI Attribute | Type | Description |
|---|---|---|
| `gen_ai.agent.id` | string | Unique agent identifier |
| `gen_ai.agent.name` | string | Agent name (used in span naming) |
| `gen_ai.agent.description` | string | Agent description |
| `gen_ai.agent.version` | string | Agent version |

GenAI also defines operations with no OI equivalent:

| Operation | Span Name Pattern | Description |
|---|---|---|
| `create_agent` | `create_agent {gen_ai.agent.name}` | Creating/configuring a GenAI agent |
| `invoke_agent` | `invoke_agent {gen_ai.agent.name}` | Invoking a previously created agent |
| `invoke_workflow` | `invoke_workflow {gen_ai.agent.name}` | Invoking a workflow |

### 4.7 Other Missing Attributes

| GenAI Attribute | Description |
|---|---|
| `gen_ai.conversation.id` | Conversation identifier (OI has `session.id` but semantics differ) |
| `gen_ai.data_source.id` | Data source identifier for retrieval operations |
| `gen_ai.tool.type` | Tool type classification (`function`, `extension`, `datastore`) |
| `gen_ai.tool.call.arguments` | Tool arguments as span attribute (on execute_tool spans) |
| `gen_ai.tool.call.result` | Tool result as span attribute (on execute_tool spans) |
| `gen_ai.embeddings.dimension.count` | Output embedding dimension count |
| `gen_ai.retrieval.query.text` | Query text for retrieval operations |
| `server.address` | GenAI server address |
| `server.port` | GenAI server port |
| `error.type` | Standardized error classification |

---

## 5. Missing in GenAI

Conventions present in [OpenInference](../semantic_conventions.md) that GenAI does not define.

### 5.1 Span Kinds with No GenAI Equivalent

| OI Span Kind | Description | Gap |
|---|---|---|
| [`CHAIN`](../traces.md#chain) | Orchestration / glue between steps | GenAI has no concept of a generic orchestration span |
| [`RERANKER`](../traces.md#reranker) | Document reranking | No reranking operation in GenAI |
| [`GUARDRAIL`](../traces.md#guardrail) | Input/output moderation | No moderation/guardrail operation in GenAI |
| [`PROMPT`](../traces.md#prompt) | Prompt template rendering | No prompt rendering operation in GenAI |

### 5.2 Cost Tracking

OI defines a [full cost breakdown](../semantic_conventions.md#token-count-details); GenAI has no cost attributes at all:

| OI Attribute | Description |
|---|---|
| `llm.cost.prompt` | Total cost of input tokens (USD) |
| `llm.cost.completion` | Total cost of output tokens (USD) |
| `llm.cost.total` | Total cost (USD) |
| `llm.cost.prompt_details.input` | Cost of standard input tokens |
| `llm.cost.prompt_details.cache_write` | Cost of cache write tokens |
| `llm.cost.prompt_details.cache_read` | Cost of cache read tokens |
| `llm.cost.prompt_details.cache_input` | Cost of cached input tokens |
| `llm.cost.prompt_details.audio` | Cost of audio input tokens |
| `llm.cost.completion_details.output` | Cost of standard output tokens |
| `llm.cost.completion_details.reasoning` | Cost of reasoning tokens |
| `llm.cost.completion_details.audio` | Cost of audio output tokens |

### 5.3 Reranker Attributes

| OI Attribute | Description |
|---|---|
| [`reranker.input_documents`](../semantic_conventions.md) | Documents input to the reranker |
| `reranker.output_documents` | Documents output by the reranker |
| `reranker.query` | Query string for reranking |
| `reranker.model_name` | Name of the reranker model |
| `reranker.top_k` | Top-K parameter |

### 5.4 Document Attributes

OI defines a rich [document model](../semantic_conventions.md) used across retrieval and reranker spans:

| OI Attribute | Description |
|---|---|
| `document.content` | Document text content |
| `document.id` | Document identifier |
| `document.metadata` | Document metadata (JSON) |
| `document.score` | Relevance score |

GenAI's `gen_ai.retrieval.documents` is opt-in and mentions IDs and relevance scores but does not define a formal document attribute schema.

### 5.5 Prompt Template System

| OI Attribute | Description |
|---|---|
| [`llm.prompt_template.template`](../semantic_conventions.md) | The template string (e.g., `"Weather for {city}"`) |
| `llm.prompt_template.variables` | Variable values applied to the template (JSON) |
| `llm.prompt_template.version` | Template version identifier |

### 5.6 Prompt Management

| OI Attribute | Description |
|---|---|
| [`prompt.vendor`](../semantic_conventions.md) | Prompt registry vendor (e.g., langsmith, portkey, arize-phoenix) |
| `prompt.id` | Vendor-specific prompt identifier |
| `prompt.url` | URL to the prompt in the vendor's UI |

### 5.7 Embedding Vectors

| OI Attribute | Description |
|---|---|
| [`embedding.embeddings`](../embedding_spans.md) | Full list of embedding objects (text + vector) |
| `embedding.text` | Text that was embedded |
| `embedding.vector` | The embedding vector (list of floats) |
| `embedding.invocation_parameters` | Embedding API parameters (JSON) |

GenAI tracks embedding dimensions and encoding formats but does not capture the actual vectors or input text.

### 5.8 Execution Graph

| OI Attribute | Description |
|---|---|
| [`graph.node.id`](../semantic_conventions.md) | Node ID in execution graph |
| `graph.node.name` | Human-readable node name |
| `graph.node.parent_id` | Parent node ID (for visualization) |

### 5.9 General-Purpose Attributes

| OI Attribute | Description |
|---|---|
| [`input.value`](../semantic_conventions.md) / `output.value` | Generic serialized input/output for any span kind |
| `input.mime_type` / `output.mime_type` | MIME type of the input/output value |
| `metadata` | Arbitrary metadata JSON for any span |
| `tag.tags` | List of string tags for categorization |
| `user.id` | User identifier |
| `agent.name` | Agent name |

### 5.10 Privacy / Redaction Configuration

OI defines [environment variables](../configuration.md) for controlling what data is captured:

| Environment Variable | Description |
|---|---|
| `OPENINFERENCE_HIDE_INPUTS` | Hide all input values and messages |
| `OPENINFERENCE_HIDE_OUTPUTS` | Hide all output values and messages |
| `OPENINFERENCE_HIDE_INPUT_MESSAGES` | Hide input messages only |
| `OPENINFERENCE_HIDE_OUTPUT_MESSAGES` | Hide output messages only |
| `OPENINFERENCE_HIDE_INPUT_IMAGES` | Hide images in input messages |
| `OPENINFERENCE_HIDE_INPUT_TEXT` | Hide text in input messages |
| `OPENINFERENCE_HIDE_OUTPUT_TEXT` | Hide text in output messages |
| `OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS` | Hide invocation parameters |
| `OPENINFERENCE_HIDE_PROMPTS` | Hide LLM prompts (completions API) |
| `OPENINFERENCE_HIDE_CHOICES` | Hide LLM choices (completions API outputs) |
| `OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS` | Redact embedding vectors |
| `OPENINFERENCE_HIDE_EMBEDDINGS_TEXT` | Redact embedding text |
| `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` | Cap base64 image encoding length |

GenAI recommends that sensitive data not be captured by default, but provides no standardized configuration mechanism — it is left to individual instrumentations.

---

## 6. Compatibility Issues

### 6.1 Flattened Attributes vs Structured JSON

**Impact: High** — This is the most fundamental difference and the biggest obstacle to convergence.

OI represents all structured data (messages, tool calls, documents) as flattened span attributes using indexed prefixes (`llm.input_messages.0.message.role`). GenAI uses structured JSON objects stored in a single attribute or event body (`gen_ai.input.messages`).

| Concern | Detail |
|---|---|
| Translation complexity | Converting between formats requires traversing and reconstructing nested structures |
| OTel backend support | Not all backends support structured/complex attribute values; OI's flattened approach has broader backend compatibility |
| Query patterns | OI's flattened attributes can be queried individually; GenAI's structured JSON requires JSON path queries |
| Attribute count | OI can generate hundreds of attributes for a single span (one per message field per index); GenAI uses fewer attributes |

### 6.2 System / Provider Attribute Semantics

**Impact: Medium**

OI separates AI product identity from hosting provider:
- [`llm.system`](../semantic_conventions.md#system-and-model-identification) = the AI product (e.g., `openai`, `anthropic`)
- [`llm.provider`](../semantic_conventions.md#system-and-model-identification) = the hosting provider (e.g., `azure`, `aws`, `google`)

GenAI combines these into [`gen_ai.provider.name`](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) with composite values:
- `openai` (direct)
- `azure.ai.openai` (Azure-hosted OpenAI)
- `gcp.vertex_ai` (Google Cloud Vertex AI)
- `aws.bedrock` (AWS Bedrock)

| OI `llm.system` | OI `llm.provider` | GenAI `gen_ai.provider.name` |
|---|---|---|
| `openai` | `openai` | `openai` |
| `openai` | `azure` | `azure.ai.openai` |
| `anthropic` | `anthropic` | `anthropic` |
| `anthropic` | `aws` | `aws.bedrock` |
| `vertexai` | `google` | `gcp.vertex_ai` |
| `cohere` | `cohere` | `cohere` |
| `mistralai` | `mistralai` | `mistral_ai` |
| `openai` | `groq` | `groq` |
| `deepseek` | `deepseek` | `deepseek` |

Note the naming differences: OI uses `mistralai`, GenAI uses `mistral_ai`. OI uses `vertexai`, GenAI uses `gcp.vertex_ai`.

GenAI also defines provider values with no direct OI mapping: `azure.ai.inference`, `gcp.gemini`, `gcp.gen_ai`, `ibm.watsonx.ai`.

### 6.3 Invocation Parameters: Opaque JSON vs Typed Attributes

**Impact: Medium**

OI stores all LLM parameters in a single `llm.invocation_parameters` JSON string. This is flexible (any parameter can be stored) but opaque to backends (can't filter/aggregate on `temperature` without parsing JSON).

GenAI defines individual typed attributes (`gen_ai.request.temperature`, `gen_ai.request.top_p`, etc.) which are directly queryable but must be extended for new parameters.

Translation from OI → GenAI requires parsing the JSON and extracting known fields. Translation from GenAI → OI requires serializing individual attributes into a JSON object.

### 6.4 Token Count Attribute Names

**Impact: Low**

The mapping is straightforward but the naming conventions differ:

| OI Pattern | GenAI Pattern |
|---|---|
| `llm.token_count.prompt` | `gen_ai.usage.input_tokens` |
| `llm.token_count.completion` | `gen_ai.usage.output_tokens` |
| `llm.token_count.prompt_details.cache_read` | `gen_ai.usage.cache_read.input_tokens` |
| `llm.token_count.prompt_details.cache_write` | `gen_ai.usage.cache_creation.input_tokens` |

OI uses "prompt/completion" terminology (from the OpenAI API). GenAI uses "input/output" terminology (provider-neutral).

### 6.5 Tool Definition Format

**Impact: Medium**

OI stores tool definitions as a [flattened list](../tool_calling.md#tool-definitions):
```
llm.tools.0.tool.json_schema = '{"type":"function","function":{"name":"get_weather",...}}'
```

GenAI stores tool definitions as a structured JSON array in `gen_ai.tool.definitions`. The internal schema of each tool definition may also differ.

### 6.6 Multimodal Content Representation

**Impact: Medium**

OI uses [`message_content.type`](../multimodal_attributes.md) with values `text`, `image`, `audio` and nested objects (`message_content.image.image.url`).

GenAI uses a `parts` array with richer type discrimination: `text`, `blob`, `uri`, `file`, `tool_call`, `tool_call_response`, `reasoning`, `server_tool_call`, `server_tool_call_response`, plus `modality` and `mime_type` fields.

GenAI supports video content and provider-hosted file references, which OI does not.

### 6.7 Embedding Span Scope

**Impact: Low**

OI captures the [full embedding data](../embedding_spans.md) (input text, output vectors) on embedding spans. GenAI captures only metadata (model, dimension count, token usage) and does not store the actual embeddings.

This is a philosophical difference: OI prioritizes debugging and data analysis; GenAI prioritizes lightweight observability.

---

## 7. Harmonious Values

Areas where the two conventions align well and translation is straightforward.

### 7.1 Core Concepts

Both conventions share the same foundational concepts:

| Concept | OI | GenAI | Alignment |
|---|---|---|---|
| LLM inference | `LLM` span | `chat` / `text_completion` operation | Same concept, different naming |
| Embeddings | `EMBEDDING` span | `embeddings` operation | Direct equivalent |
| Retrieval | `RETRIEVER` span | `retrieval` operation | Direct equivalent |
| Tool execution | `TOOL` span | `execute_tool` operation | Direct equivalent |
| Agent orchestration | `AGENT` span | `invoke_agent` operation | Close equivalent |

### 7.2 Message Roles

Both use the same role values:

| Role | OI | GenAI | Notes |
|---|---|---|---|
| `user` | Supported | Supported | Identical |
| `system` | Supported (in messages) | Supported (separate `gen_ai.system_instructions`) | Same concept, different placement |
| `assistant` | Supported | Supported | Identical |
| `tool` | Supported | Supported | Identical |
| `function` | Supported (legacy) | Not used | OI legacy; deprecated in favor of `tool` |

### 7.3 Provider Name Values

Many provider values are identical or trivially mappable:

| Provider | OI `llm.system` | OI `llm.provider` | GenAI `gen_ai.provider.name` | Status |
|---|---|---|---|---|
| OpenAI | `openai` | `openai` | `openai` | Identical |
| Anthropic | `anthropic` | `anthropic` | `anthropic` | Identical |
| Cohere | `cohere` | `cohere` | `cohere` | Identical |
| DeepSeek | `deepseek` | `deepseek` | `deepseek` | Identical |
| xAI | `xai` | `xai` | `x_ai` | Minor difference (underscore) |
| Mistral | `mistralai` | `mistralai` | `mistral_ai` | Minor difference (underscore) |
| Groq | — | `groq` | `groq` | Identical |
| Perplexity | — | `perplexity` | `perplexity` | Identical |
| Fireworks | — | `fireworks` | — | OI-only provider |
| Moonshot | — | `moonshot` | — | OI-only provider |
| Cerebras | — | `cerebras` | — | OI-only provider |
| Together | — | `together` | — | OI-only provider |
| Amazon | `amazon` | — | `aws.bedrock` | Different naming |
| Meta | `meta` | — | — | OI-only system |
| AI21 | `ai21` | — | — | OI-only system |
| IBM | — | — | `ibm.watsonx.ai` | GenAI-only provider |

> **Sources:** OI system values are defined in [`semantic_conventions.md`](../semantic_conventions.md) and enumerated in [`OpenInferenceLLMSystemValues`](../../python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py) / [`OpenInferenceLLMProviderValues`](../../python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py). GenAI provider values are defined in the [GenAI span conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/).

### 7.4 Tool Calling Core Fields

Both conventions capture the same essential tool call information:

| Field | OI | GenAI |
|---|---|---|
| Tool call ID | [`tool_call.id`](../tool_calling.md) | `gen_ai.tool.call.id` / message `id` field |
| Function name | `tool_call.function.name` | Message `name` field |
| Function arguments | `tool_call.function.arguments` | Message `arguments` field |
| Tool name | `tool.name` | `gen_ai.tool.name` |
| Tool description | `tool.description` | `gen_ai.tool.description` |

### 7.5 OTel Foundation

Both conventions are built on OpenTelemetry:
- Both use OTel spans as the primary trace unit
- Both use OTel span attributes for data
- Both are designed to work with OTel-compatible backends
- Both support context propagation

### 7.6 Sensitive Data Philosophy

Both conventions recognize the need to protect sensitive data:
- OI: opt-out via [`OPENINFERENCE_HIDE_*`](../configuration.md) environment variables (captures by default)
- GenAI: opt-in capture (sensitive data not recorded by default)

The default posture differs (OI captures by default with opt-out; GenAI does not capture by default with opt-in), but both provide mechanisms for controlling data sensitivity.

---

## 8. Convergence Strategy

GenAI is the upstream OTel standard. As it matures toward stability, OpenInference should converge toward it while preserving the capabilities that GenAI does not yet cover. This section outlines a phased approach.

### Design Principles

1. **GenAI is the target for shared concepts.** Where both conventions define the same concept (token usage, model name, tool calling), OpenInference should adopt GenAI attribute names.
2. **OpenInference extends, not forks.** OI-unique capabilities (cost tracking, reranker, guardrail, prompt templates, document model, privacy config) should be maintained as extensions and proposed upstream to GenAI.
3. **Migration must be non-breaking.** Existing OI backends and dashboards must continue to work throughout the transition. Dual-write periods are necessary.
4. **Flattened format has deployment gravity.** OI's flattened attribute format works with every OTel backend. The structured JSON format requires backend-specific support. Convergence on message format must account for this.

### Phase 1: Dual-Write Aliases (Non-Breaking)

**Goal:** Emit GenAI attributes alongside OI attributes on the same spans, giving backends time to adopt.

| Action | Detail |
|---|---|
| Emit `gen_ai.usage.input_tokens` alongside `llm.token_count.prompt` | Mechanical alias — both carry the same integer value |
| Emit `gen_ai.request.model` alongside `llm.model_name` | Same |
| Emit `gen_ai.request.temperature`, `gen_ai.request.top_p`, etc. alongside `llm.invocation_parameters` | Parse the JSON blob and emit individual typed attributes in addition |
| Emit `gen_ai.response.id`, `gen_ai.response.model`, `gen_ai.response.finish_reasons` | Net-new attributes — no OI equivalent to deprecate |
| Emit `gen_ai.operation.name` alongside `openinference.span.kind` | Map `LLM` → `chat`, `EMBEDDING` → `embeddings`, etc. |
| Emit `gen_ai.provider.name` alongside `llm.system` / `llm.provider` | Requires a mapping function that combines system + provider into composite values |

**What stays unchanged:** All OI-unique attributes (cost, reranker, guardrail, prompt templates, graph, privacy config) continue to be emitted as-is under their existing names.

### Phase 2: Structured Messages (Requires Backend Coordination)

**Goal:** Adopt GenAI's structured JSON message format while maintaining backward compatibility.

This is the highest-risk phase because it changes the fundamental data shape.

| Action | Detail |
|---|---|
| Emit `gen_ai.input.messages` and `gen_ai.output.messages` as structured JSON | Alongside (not replacing) the flattened `llm.input_messages.*` attributes |
| Emit `gen_ai.system_instructions` for system messages | Extract from the flattened message list |
| Emit `gen_ai.tool.definitions` as structured JSON | Alongside the flattened `llm.tools.*` attributes |

During this phase, backends must be updated to read from `gen_ai.*` attributes. The flattened attributes are retained for backward compatibility and can be deprecated once adoption is confirmed.

### Phase 3: Upstream Contribution

**Goal:** Propose OI-unique capabilities to the GenAI semconv working group.

| Capability | Rationale for Upstream |
|---|---|
| **Cost tracking** (`llm.cost.*`) | Token costs are a universal operational concern; every AI team tracks spend |
| **Reranker span kind & attributes** | RAG pipelines are ubiquitous; reranking is a first-class operation |
| **Guardrail span kind** | Safety/moderation is a growing requirement across the industry |
| **Document attribute schema** | GenAI's retrieval documents are underspecified compared to OI's `document.*` model |
| **Prompt template system** | Prompt versioning and variable tracking are standard in production AI systems |
| **Privacy configuration** | Standardized redaction config is better than per-instrumentation ad-hoc solutions |
| **General-purpose `input.value` / `output.value`** | Useful for non-LLM span kinds (chains, guardrails) where structured messages don't apply |

### Phase 4: Deprecation of Legacy Attributes

**Goal:** Once GenAI equivalents are widely supported by backends, deprecate and eventually remove the OI-specific names for shared concepts.

| Deprecated OI Attribute | Replacement |
|---|---|
| `llm.token_count.prompt` | `gen_ai.usage.input_tokens` |
| `llm.token_count.completion` | `gen_ai.usage.output_tokens` |
| `llm.model_name` | `gen_ai.request.model` |
| `llm.system` + `llm.provider` | `gen_ai.provider.name` |
| `llm.invocation_parameters` (JSON blob) | Individual `gen_ai.request.*` attributes |
| `llm.input_messages.*` (flattened) | `gen_ai.input.messages` (structured JSON) |
| `llm.output_messages.*` (flattened) | `gen_ai.output.messages` (structured JSON) |

Attributes with no GenAI equivalent (cost, reranker, guardrail, prompt templates, document model, graph, privacy config) remain under their current OI names until/unless adopted upstream.

### Open Questions

1. **Span kind taxonomy:** Should OI's richer span kinds (`CHAIN`, `RERANKER`, `GUARDRAIL`, `PROMPT`) be proposed as `gen_ai.operation.name` values or maintained as a separate `openinference.span.kind` attribute?
2. **Privacy default posture:** OI captures by default (opt-out); GenAI does not (opt-in). Which default should a converged convention use?
3. **Cost attribute namespace:** If cost tracking is accepted upstream, should it live under `gen_ai.cost.*` or retain the `llm.cost.*` prefix?
4. **Embedding vector capture:** OI captures full vectors for debugging; GenAI deliberately omits them. Should a converged convention support both modes?
5. **Message format transition timeline:** How long should the dual-write period last before flattened attributes are deprecated?
