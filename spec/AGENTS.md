# OpenInference Specification Guide

Language-agnostic conventions that all OpenInference instrumentors must follow. Defines the span kinds and attribute names used to represent AI/ML operations in OpenTelemetry traces.

---

## Span Kinds

`openinference.span.kind` is **REQUIRED** on every OpenInference span. It tells the tracing backend how to assemble and visualize the trace.

| Span Kind | Description |
|-----------|-------------|
| `LLM` | Call to a Large Language Model — e.g., OpenAI or Anthropic for chat completions or text generation |
| `EMBEDDING` | Call to an embedding model or service — e.g., OpenAI for ada embeddings used in retrieval |
| `CHAIN` | Starting point or link between application steps — e.g., the glue code passing context from a retriever to an LLM |
| `RETRIEVER` | Data retrieval step — e.g., a vector store or database query fetching relevant documents |
| `RERANKER` | Reranking a set of documents by relevance — e.g., a cross-encoder scoring documents against a query |
| `TOOL` | External tool invoked by an LLM or agent — e.g., a calculator, weather API, or function call |
| `AGENT` | Reasoning block that acts on tools using LLM guidance — encompasses LLM calls and tool invocations |
| `GUARDRAIL` | Safety component checking or modifying LLM output — e.g., content filtering, jailbreak detection |
| `EVALUATOR` | Function evaluating LLM output quality — e.g., relevance scoring, correctness checking |
| `PROMPT` | Rendering of a prompt template — e.g., filling variables into a template string |

---

## Flattened Array Format (critical implementation detail)

OpenTelemetry span attributes are flat key-value pairs; there are no nested objects or arrays. OpenInference represents lists of structured objects using **indexed flattened attributes**:

```
<prefix>.<zero-based-index>.<suffix>
```

**Examples:**

```
# Input messages to an LLM:
llm.input_messages.0.message.role    = "system"
llm.input_messages.0.message.content = "You are a helpful assistant."
llm.input_messages.1.message.role    = "user"
llm.input_messages.1.message.content = "What is the capital of France?"

# Output messages from an LLM:
llm.output_messages.0.message.role    = "assistant"
llm.output_messages.0.message.content = "The capital of France is Paris."

# Tool calls in an output message:
llm.output_messages.0.message.tool_calls.0.tool_call.function.name      = "get_weather"
llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments = "{\"city\": \"London\"}"
llm.output_messages.0.message.tool_calls.0.tool_call.id                 = "call_62136355"

# Available tools advertised to the LLM:
llm.tools.0.tool.json_schema = "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", ...}}"

# Retrieved documents:
retrieval.documents.0.document.id      = "doc-123"
retrieval.documents.0.document.content = "Paris is the capital of France..."
retrieval.documents.0.document.score   = 0.98
```

**Python implementation:**
```python
for i, msg in enumerate(messages):
    span.set_attribute(f"llm.input_messages.{i}.message.role", msg["role"])
    span.set_attribute(f"llm.input_messages.{i}.message.content", msg["content"])
```

**JavaScript implementation:**
```javascript
messages.forEach((msg, i) => {
  span.setAttribute(`llm.input_messages.${i}.message.role`, msg.role);
  span.setAttribute(`llm.input_messages.${i}.message.content`, msg.content);
});
```

Flatten recursively until all leaf values are primitives (`bool`, `string`, `int`, `float`) or simple arrays of primitives.

---

## Key Attributes Reference

### Always Required

| Attribute | Type | Description |
|-----------|------|-------------|
| `openinference.span.kind` | String | One of the 10 span kind values above. REQUIRED on every span. |

### Input and Output

| Attribute | Type | Description |
|-----------|------|-------------|
| `input.value` | String | The input to the operation |
| `input.mime_type` | String | MIME type of input: `text/plain` or `application/json` |
| `output.value` | String | The output of the operation |
| `output.mime_type` | String | MIME type of output: `text/plain` or `application/json` |

### LLM Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `llm.model_name` | String | Model identifier (e.g., `"gpt-4o"`, `"claude-3-opus-20240229"`) |
| `llm.system` | String | AI product/vendor: `openai`, `anthropic`, `vertexai`, `cohere`, `mistralai` |
| `llm.provider` | String | Hosting provider: `openai`, `anthropic`, `azure`, `aws`, `google` |
| `llm.input_messages` | Flattened array | Messages sent to the LLM (see array format above) |
| `llm.output_messages` | Flattened array | Messages received from the LLM |
| `llm.invocation_parameters` | JSON String | Model parameters (temperature, max_tokens, etc.) |
| `llm.token_count.prompt` | Integer | Tokens in the prompt |
| `llm.token_count.completion` | Integer | Tokens in the completion |
| `llm.token_count.total` | Integer | Total tokens |
| `llm.token_count.prompt_details.cache_read` | Integer | Cached prompt tokens (cache hits) |
| `llm.token_count.prompt_details.cache_write` | Integer | Prompt tokens written to cache (Anthropic) |
| `llm.token_count.completion_details.reasoning` | Integer | Reasoning tokens (chain-of-thought models) |
| `llm.cost.prompt` | Float | Cost of prompt tokens in USD |
| `llm.cost.completion` | Float | Cost of completion tokens in USD |
| `llm.cost.total` | Float | Total cost in USD |
| `llm.tools` | Flattened array | Tools advertised to the LLM |
| `llm.prompt_template.template` | String | Prompt template as a Python f-string |
| `llm.prompt_template.version` | String | Template version |
| `llm.prompt_template.variables` | JSON String | Variables applied to the template |

### Embedding Attributes

Note: `llm.system` and `llm.provider` are **not used** for EMBEDDING spans.

| Attribute | Type | Description |
|-----------|------|-------------|
| `embedding.model_name` | String | Embedding model name (e.g., `"text-embedding-3-small"`) |
| `embedding.embeddings` | Flattened array | List of embedding objects with text and vector |
| `embedding.invocation_parameters` | JSON String | Embedding API parameters |

### Document Attributes (RETRIEVER / RERANKER)

| Attribute | Type | Description |
|-----------|------|-------------|
| `retrieval.documents` | Flattened array | Retrieved documents |
| `reranker.input_documents` | Flattened array | Documents input to reranker |
| `reranker.output_documents` | Flattened array | Documents output by reranker |
| `reranker.query` | String | Query used for reranking |
| `reranker.model_name` | String | Reranker model name |
| `reranker.top_k` | Integer | Top K parameter |
| `document.id` | String | Document identifier |
| `document.content` | String | Document text content |
| `document.score` | Float | Relevance score |
| `document.metadata` | JSON String | Document metadata |

### Tool Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `tool.name` | String | Tool name |
| `tool.description` | String | Tool description |
| `tool.json_schema` | JSON String | Tool input schema |
| `tool.parameters` | JSON String | Parameters definition |
| `tool.id` | String | Tool call result identifier |
| `tool_call.function.name` | String | Function name in a tool call |
| `tool_call.function.arguments` | JSON String | Function arguments |
| `tool_call.id` | String | Tool call identifier |

### Session and User

| Attribute | Type | Description |
|-----------|------|-------------|
| `session.id` | String | Session identifier (groups multi-turn conversations) |
| `user.id` | String | User identifier |
| `metadata` | JSON String | Custom metadata |
| `tag.tags` | List of strings | Span tags for filtering |

### Agent and Graph

| Attribute | Type | Description |
|-----------|------|-------------|
| `agent.name` | String | Name of the agent |
| `graph.node.id` | String | Node ID in execution graph |
| `graph.node.name` | String | Human-readable node name |
| `graph.node.parent_id` | String | Parent node ID (empty = root) |

### Exception

| Attribute | Type | Description |
|-----------|------|-------------|
| `exception.type` | String | Exception class name |
| `exception.message` | String | Exception message |
| `exception.stacktrace` | String | Full stack trace |
| `exception.escaped` | Boolean | Whether exception escaped the span scope |

### Multimodal

| Attribute | Type | Description |
|-----------|------|-------------|
| `image.url` | String | Image URL or base64 encoding |
| `audio.url` | String | Audio file URL |
| `audio.mime_type` | String | Audio MIME type (e.g., `audio/mpeg`) |
| `audio.transcript` | String | Audio transcript |

---

## Spec Files Index

| File | Contents |
|------|---------|
| `semantic_conventions.md` | Complete attribute reference (primary reference) |
| `traces.md` | Trace and span structure overview |
| `llm_spans.md` | Detailed LLM span specification |
| `embedding_spans.md` | Embedding span specification (explains why llm.system is excluded) |
| `tool_calling.md` | Tool call attribute patterns |
| `multimodal_attributes.md` | Image and audio attribute specifications |
| `configuration.md` | Environment variables for TraceConfig settings |
