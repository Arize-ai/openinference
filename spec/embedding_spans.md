# Embedding Spans

Embedding spans capture operations that generate vector embeddings from text, images, or other inputs.

## Required Attributes

All embedding spans MUST include:
- `openinference.span.kind`: Set to `"EMBEDDING"`
- `llm.system`: The AI system/product (e.g., "openai", "cohere")

## Common Attributes

Embedding spans typically include:
- `llm.model_name`: The specific embedding model used (e.g., "text-embedding-3-small")
- `llm.invocation_parameters`: JSON string of parameters sent to the model
- `input.value`: The raw input as a JSON string (may contain text, array of texts, or tokens)
- `input.mime_type`: Usually "application/json"
- `output.value`: The raw output (often base64-encoded vectors or array of vectors)
- `output.mime_type`: Usually "application/json"
- `embedding.model_name`: Name of the embedding model (may duplicate `llm.model_name`)
- `embedding.text`: The text being embedded (when hiding is not enabled)
- `embedding.vector`: The resulting embedding vector (when hiding is not enabled)
- `embedding.embeddings`: List of embedding objects for batch operations

## Privacy Considerations

When `OPENINFERENCE_HIDE_EMBEDDING_VECTORS` is set to true:
- The `embedding.vector` attribute will contain `"__REDACTED__"`
- The actual vector data will not be included in traces

When `OPENINFERENCE_HIDE_INPUT_TEXT` is set to true:
- The `embedding.text` attribute will contain `"__REDACTED__"`
- The input text will not be included in traces

## Example

A span for generating embeddings with OpenAI:

```json
{
    "name": "CreateEmbeddingResponse",
    "span_kind": "SPAN_KIND_INTERNAL",
    "attributes": {
        "openinference.span.kind": "EMBEDDING",
        "llm.system": "openai",
        "llm.model_name": "text-embedding-3-small",
        "input.value": "{\"input\": \"hello world\", \"model\": \"text-embedding-3-small\", \"encoding_format\": \"base64\"}",
        "input.mime_type": "application/json",
        "llm.invocation_parameters": "{\"model\": \"text-embedding-3-small\", \"encoding_format\": \"base64\"}",
        "embedding.model_name": "text-embedding-3-small",
        "embedding.text": "hello world",
        "embedding.vector": "[0.1, 0.2, 0.3, ...]"
    }
}
```

For batch embedding operations, the embeddings are flattened:

```json
{
    "attributes": {
        "embedding.embeddings.0.embedding.text": "first text",
        "embedding.embeddings.0.embedding.vector": "[0.1, 0.2, ...]",
        "embedding.embeddings.1.embedding.text": "second text", 
        "embedding.embeddings.1.embedding.vector": "[0.3, 0.4, ...]"
    }
}
```