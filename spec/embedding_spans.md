# Embedding Spans

Embedding spans capture operations that convert text or token IDs into dense float vectors for semantic search, clustering, and similarity comparison.

## Span Name

The span name MUST be `"CreateEmbeddings"` for embedding operations.

## Required Attributes

All embedding spans MUST include:
- `openinference.span.kind`: Set to `"EMBEDDING"`

## Common Attributes

Embedding spans typically include:

- `embedding.model_name`: Name of the embedding model (e.g., "text-embedding-3-small")
- `embedding.embeddings`: Nested structure for embedding objects in batch operations
- `embedding.invocation_parameters`: JSON string of parameters sent to the model (excluding input)
- `input.value`: The raw input as a JSON string (text strings or token ID arrays)
- `input.mime_type`: Usually "application/json"
- `output.value`: The raw output (embedding vectors as JSON or base64-encoded)
- `output.mime_type`: Usually "application/json"
- `llm.token_count.prompt`: Number of tokens in the input
- `llm.token_count.total`: Total number of tokens used

### Text Attributes

The `embedding.embeddings.N.embedding.text` attributes are populated ONLY when the input is already text (strings). These attributes are recorded during the request phase to ensure availability even on errors.

Token IDs (pre-tokenized integer arrays) are NOT decoded to text because:
- **Cross-provider incompatibility**: Same token IDs represent different text across tokenizers (OpenAI uses cl100k_base, Ollama uses BERT/WordPiece/etc.)
- **Runtime impossibility**: OpenAI-compatible APIs may serve any model with unknown tokenizers
- **Heavy dependencies**: Supporting all tokenizers would require libraries beyond tiktoken (which only supports OpenAI)

### Vector Attributes

The `embedding.embeddings.N.embedding.vector` attributes MUST contain float arrays, regardless of the API response format:

1. **Float response format**: Store vectors directly as float arrays
2. **Base64 response format**: MUST decode base64-encoded strings to float arrays before recording
   - Base64 encoding is ~25% more compact in transmission but must be decoded for consistency
   - Example: "AACAPwAAAEA=" → [1.5, 2.0]

## Attributes Not Used in Embedding Spans

The following attributes that are used in LLM spans are **not applicable** to embedding spans:

- `llm.system`: Not used for embedding spans
- `llm.provider`: Not used for embedding spans

### Rationale

The `llm.system` attribute is defined as "the AI product as identified by the client or server instrumentation." While this definition has been reserved for API providers in LLM spans (e.g., "openai", "anthropic"), it is ambiguous when applied to embedding operations.

In terms of conceptualization, `llm.system` describes the shape of the API, while `llm.provider` describes the owner of the physical hardware that runs those APIs. For observability products like Arize and Phoenix, these conventions are primarily consumed in playground features, allowing re-invocation of LLM calls.

For embedding operations:
- The `embedding.model_name` attribute provides sufficient identification of the embedding model being used
- The span kind `"EMBEDDING"` clearly identifies the operation type
- There is no concrete use case for playground re-invocation of embedding calls that would require `llm.system`
- Expanding the definition to include SDK/library names as systems does not facilitate any current observability use cases

Therefore, to avoid ambiguity and maintain clear semantic conventions, embedding spans use `embedding.model_name` rather than `llm.system` or `llm.provider`.

## Privacy Considerations

When `OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS` is set to true:
- The `embedding.embeddings.N.embedding.vector` attribute will contain `"__REDACTED__"`
- The actual vector data will not be included in traces

When `OPENINFERENCE_HIDE_EMBEDDINGS_TEXT` is set to true:
- The `embedding.embeddings.N.embedding.text` attribute will contain `"__REDACTED__"`
- The input text will not be included in traces

## Input/Output Structure

The response structure matches the input structure:
- Single input (text or token array) → `data[0]` with one embedding
- Array of N inputs → `data[0..N-1]` with N embeddings

Input formats (cannot mix text and tokens in one request):
- Single text: `"hello world"` → single embedding
- Text array: `["hello", "world"]` → array of embeddings
- Single token array: `[15339, 1917]` → single embedding
- Token array of arrays: `[[15339, 1917], [991, 1345]]` → array of embeddings

## Examples

### Text Input (Recorded in Traces)

A span for generating embeddings from text:

```json
{
    "name": "CreateEmbeddings",
    "span_kind": "SPAN_KIND_INTERNAL",
    "attributes": {
        "openinference.span.kind": "EMBEDDING",
        "embedding.model_name": "text-embedding-3-small",
        "embedding.invocation_parameters": "{\"model\": \"text-embedding-3-small\", \"encoding_format\": \"float\"}",
        "input.value": "{\"input\": \"hello world\", \"model\": \"text-embedding-3-small\", \"encoding_format\": \"float\"}",
        "input.mime_type": "application/json",
        "output.value": "{\"data\": [{\"embedding\": [0.1, 0.2, 0.3], \"index\": 0}], \"model\": \"text-embedding-3-small\", \"usage\": {\"prompt_tokens\": 2, \"total_tokens\": 2}}",
        "output.mime_type": "application/json",
        "embedding.embeddings.0.embedding.text": "hello world",
        "embedding.embeddings.0.embedding.vector": [0.1, 0.2, 0.3],
        "llm.token_count.prompt": 2,
        "llm.token_count.total": 2
    }
}
```

### Token Input (No Text Attributes)

When input consists of pre-tokenized integer arrays, text attributes are NOT recorded:

```json
{
    "name": "CreateEmbeddings",
    "span_kind": "SPAN_KIND_INTERNAL",
    "attributes": {
        "openinference.span.kind": "EMBEDDING",
        "embedding.model_name": "text-embedding-3-small",
        "embedding.invocation_parameters": "{\"model\": \"text-embedding-3-small\", \"encoding_format\": \"float\"}",
        "input.value": "{\"input\": [15339, 1917], \"model\": \"text-embedding-3-small\", \"encoding_format\": \"float\"}",
        "input.mime_type": "application/json",
        "output.value": "{\"data\": [{\"embedding\": [0.1, 0.2, 0.3], \"index\": 0}], \"model\": \"text-embedding-3-small\", \"usage\": {\"prompt_tokens\": 2, \"total_tokens\": 2}}",
        "output.mime_type": "application/json",
        "embedding.embeddings.0.embedding.vector": [0.1, 0.2, 0.3],
        "llm.token_count.prompt": 2,
        "llm.token_count.total": 2
    }
}
```

### Batch Text Input (Multiple Embeddings)

A span for generating embeddings from multiple text inputs:

```json
{
    "name": "CreateEmbeddings",
    "span_kind": "SPAN_KIND_INTERNAL",
    "attributes": {
        "openinference.span.kind": "EMBEDDING",
        "embedding.model_name": "text-embedding-ada-002",
        "embedding.invocation_parameters": "{\"model\": \"text-embedding-ada-002\"}",
        "input.value": "[\"hello\", \"world\", \"test\"]",
        "embedding.embeddings.0.embedding.text": "hello",
        "embedding.embeddings.0.embedding.vector": [0.1, 0.2, 0.3],
        "embedding.embeddings.1.embedding.text": "world",
        "embedding.embeddings.1.embedding.vector": [0.4, 0.5, 0.6],
        "embedding.embeddings.2.embedding.text": "test",
        "embedding.embeddings.2.embedding.vector": [0.7, 0.8, 0.9],
        "llm.token_count.prompt": 3,
        "llm.token_count.total": 3
    }
}
```
