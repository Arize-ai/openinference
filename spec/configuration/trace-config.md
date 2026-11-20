# Trace Configuration

Trace configuration controls how much an OpenInference instrumentation records. You may need to keep sensitive
information out of traces for security reasons, or limit the size of base64-encoded images to reduce payload size.

The specification defines a set of environment variables for these settings. The SDKs also accept the same settings
in code, as a `TraceConfig` object passed to an instrumentor.

## Settings

| Environment variable | Type | Default | Effect |
| --- | --- | --- | --- |
| `OPENINFERENCE_HIDE_INPUTS` | bool | `false` | Hides `input.value`, `input.images`, `input.documents`, all input messages, and the tool definitions advertised to the LLM |
| `OPENINFERENCE_HIDE_OUTPUTS` | bool | `false` | Hides `output.value`, `output.images`, `output.documents`, and all output messages |
| `OPENINFERENCE_HIDE_INPUT_MESSAGES` | bool | `false` | Hides all input messages (independent of `HIDE_INPUTS`) |
| `OPENINFERENCE_HIDE_OUTPUT_MESSAGES` | bool | `false` | Hides all output messages (independent of `HIDE_OUTPUTS`) |
| `OPENINFERENCE_HIDE_INPUT_IMAGES` | bool | `false` | Hides images from input messages and all `input.images` attributes |
| `OPENINFERENCE_HIDE_INPUT_TEXT` | bool | `false` | Hides text from input messages, including message document `document.content` |
| `OPENINFERENCE_HIDE_OUTPUT_TEXT` | bool | `false` | Hides text from output messages, including message document `document.content` |
| `OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS` | bool | `false` | Hides LLM invocation parameters (independent of input/output hiding) |
| `OPENINFERENCE_HIDE_LLM_TOOLS` | bool | `false` | Hides the tool definitions advertised to the LLM (`llm.tools.*`) |
| `OPENINFERENCE_HIDE_PROMPTS` | bool | `false` | Hides LLM prompts (completions API) |
| `OPENINFERENCE_HIDE_CHOICES` | bool | `false` | Hides LLM choices (completions API outputs) |
| `OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS` | bool | `false` | Replaces `embedding.embeddings.*.embedding.vector` values with `"__REDACTED__"` |
| `OPENINFERENCE_HIDE_EMBEDDING_VECTORS` | bool | `false` | Deprecated: use `OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS` |
| `OPENINFERENCE_HIDE_EMBEDDINGS_TEXT` | bool | `false` | Replaces `embedding.embeddings.*.embedding.text` values with `"__REDACTED__"` |
| `OPENINFERENCE_HIDE_RETRIEVAL_DOCUMENTS` | bool | `false` | Replaces `retrieval.documents.*.document.content` and `.document.metadata` values with `"__REDACTED__"` |
| `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` | int | `32000` | Limits the characters of a base64 data URI recorded in an image `image.url` or a document `document.url`; longer values are uploaded or redacted |
| `OPENINFERENCE_BLOB_UPLOADER` | str | unset | Names a blob uploader that receives oversized base64 images and documents. Experimental; see [Blob Upload](./blob-upload.md) |
| `OPENINFERENCE_ENABLE_GENAI_SEMCONV` | bool | `false` | Also emits OTel `gen_ai.*` attributes. Experimental; see [GenAI Export](./genai-semconv.md) |

[Data Masking](./data-masking.md) describes exactly which attributes each setting hides and how the settings interact.

## Precedence

Each setting is resolved in this order:

1. The value set in code, in the `TraceConfig`
2. The environment variable
3. The default value

Settings left out of a `TraceConfig` fall back to the environment variable, then the default.

## Usage

### Python

Pass a `TraceConfig` to the `instrument()` method of any instrumentor. The example uses the `OpenAIInstrumentor`:

```python
from openinference.instrumentation import TraceConfig
config = TraceConfig(
    hide_llm_invocation_parameters=...,
    hide_llm_tools=...,  # Hides tool definitions advertised to the LLM
    hide_inputs=...,
    hide_outputs=...,
    hide_input_messages=...,
    hide_output_messages=...,
    hide_input_images=...,
    hide_input_text=...,
    hide_output_text=...,
    hide_embeddings_vectors=...,
    hide_embeddings_text=...,
    hide_retrieval_documents=...,
    base64_image_max_length=...,
    blob_uploader=...,  # Uploads oversized base64 images and documents, records a URI
    hide_prompts=...,  # Hides LLM prompts (completions API)
    hide_choices=...,  # Hides LLM choices (completions API outputs)
    enable_genai_semconv=...,  # Also emits gen_ai.* attributes (experimental)
)

from openinference.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument(
    tracer_provider=tracer_provider,
    config=config,
)
```

### JavaScript

Pass a `traceConfig` to any OpenInference instrumentation. The example uses the `OpenAIInstrumentation`:

```typescript
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai"

/**
 * Everything left out of here will fallback to
 * environment variables then defaults
 */
const traceConfig = { hideInputs: true }

const instrumentation = new OpenAIInstrumentation({ traceConfig })
```
