# OpenInference Specification Guide

Language-agnostic conventions that all OpenInference instrumentors must follow. Defines the span kinds and attribute names used to represent AI/ML operations in OpenTelemetry traces. [`semantic_conventions.md`](semantic_conventions.md) is the authoritative source for all attribute names, types, and descriptions.

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

## Attribute Reference

See [`semantic_conventions.md`](semantic_conventions.md) for the complete, authoritative list of all span attributes, types, and descriptions.

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
