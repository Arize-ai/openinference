# OpenInference Configuration

In some situations, you may need to modify the observability level of your tracing. For instance, you may want to keep sensitive information from being logged for security reasons, or you may want to limit the size of the base64 encoded images logged to reduced payload size.

The OpenInference Specification defines a set of environment variables you can configure to suit your observability needs. In addition, the OpenInference Instrumentation Python package also offers convenience functions to do this in code without having to set environment variables, if that's what you prefer.

The possible settings are:

| Environment Variable Name                    | Effect                                                                                                                         | Type | Default |
|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|------|---------|
| OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS | Hides LLM invocation parameters (independent of input/output hiding)                                                           | bool | False   |
| OPENINFERENCE_HIDE_LLM_TOOLS                 | Hides the tool definitions advertised to the LLM (`llm.tools.*`); also hidden when HIDE_INPUTS is true                         | bool | False   |
| OPENINFERENCE_HIDE_INPUTS                    | Hides input.value, all input messages, and the tool definitions advertised to the LLM (input messages are hidden if either HIDE_INPUTS OR HIDE_INPUT_MESSAGES is true) | bool | False   |
| OPENINFERENCE_HIDE_OUTPUTS                   | Hides output.value and all output messages (output messages are hidden if either HIDE_OUTPUTS OR HIDE_OUTPUT_MESSAGES is true) | bool | False   |
| OPENINFERENCE_HIDE_INPUT_MESSAGES            | Hides all input messages (independent of HIDE_INPUTS)                                                                          | bool | False   |
| OPENINFERENCE_HIDE_OUTPUT_MESSAGES           | Hides all output messages (independent of HIDE_OUTPUTS)                                                                        | bool | False   |
| OPENINFERENCE_HIDE_INPUT_IMAGES              | Hides images from input messages (only applies when input messages are not already hidden)                                     | bool | False   |
| OPENINFERENCE_HIDE_INPUT_AUDIO               | Hides audio from input messages (only applies when input messages are not already hidden)                                      | bool | False   |
| OPENINFERENCE_HIDE_OUTPUT_AUDIO              | Hides audio from output messages (only applies when output messages are not already hidden)                                    | bool | False   |
| OPENINFERENCE_HIDE_INPUT_FILES               | Hides files (e.g. PDF documents) from input messages (only applies when input messages are not already hidden)                 | bool | False   |
| OPENINFERENCE_HIDE_INPUT_TEXT                | Hides text from input messages (only applies when input messages are not already hidden)                                       | bool | False   |
| OPENINFERENCE_HIDE_PROMPTS                   | Hides LLM prompts (completions API)                                                                                            | bool | False   |
| OPENINFERENCE_HIDE_OUTPUT_TEXT               | Hides text from output messages (only applies when output messages are not already hidden)                                     | bool | False   |
| OPENINFERENCE_HIDE_CHOICES                   | Hides LLM choices (completions API outputs)                                                                                    | bool | False   |
| OPENINFERENCE_HIDE_EMBEDDING_VECTORS         | Deprecated: use OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS                                                                          | bool | False   |
| OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS        | Replaces embedding.embeddings.*.embedding.vector values with `"__REDACTED__"`                                                  | bool | False   |
| OPENINFERENCE_HIDE_EMBEDDINGS_TEXT           | Replaces embedding.embeddings.*.embedding.text values with `"__REDACTED__"`                                                    | bool | False   |
| OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH        | Limits characters of a base64 encoding of an image                                                                             | int  | 32,000  |
| OPENINFERENCE_BASE64_MEDIA_MAX_LENGTH        | Limits characters of a base64 data URI carrying non-image media (audio, files/documents, video)                                | int  | 32,000  |
| OPENINFERENCE_BLOB_UPLOADER                  | Experimental: names a `BlobUploader` registered under the `openinference_blob_uploader` entry-point group; base64 media larger than the max-length limits is handed to it and the span attribute records the returned URI instead of being redacted | str  | unset   |

## Redacted Content

When content is hidden due to privacy configuration settings, the value `"__REDACTED__"` is used as a placeholder. This constant value allows consumers of the trace data to identify that content was intentionally hidden rather than missing or empty.

## External Blob Upload (Experimental)

Large binary content (audio, PDF documents, images) captured as base64 data URIs can exceed span attribute and OTLP payload limits. Instead of redacting oversized media, an instrumentation MAY upload the decoded bytes to external storage at capture time and record only a reference URI in the span attribute. See [Multimodal Attributes](./multimodal_attributes.md#external-storage-for-large-media) for the attribute-level semantics and the design rationale.

OpenInference defines the interface and the offload policy but ships no uploader implementation — implementations come from applications, vendor SDKs (e.g. the Arize SDK), or a future upstream (OTel util-genai) byte uploader. In Python an uploader is supplied either in code, or zero-code via an entry point:

```python
from openinference.instrumentation import TraceConfig

config = TraceConfig(blob_uploader=my_uploader)  # any object satisfying BlobUploader
```

```toml
# the package providing the uploader registers it in its own packaging
# metadata under the "openinference_blob_uploader" entry-point group:
[project.entry-points.openinference_blob_uploader]
arize = "arize_otel.blob:ArizeBlobUploader"
```

```bash
export OPENINFERENCE_BLOB_UPLOADER=arize
```

The name is resolved when a `TraceConfig` is constructed: the entry point is imported, instantiated if it is a class or zero-argument factory, and validated against the `BlobUploader` protocol. Resolution failures log a warning and leave the uploader unset — oversized media then redacts exactly as with no uploader configured. A `blob_uploader` passed in code takes precedence over the environment variable. The uploader's own configuration (bucket, credentials, endpoint) is read by the uploader itself, typically from its own environment variables, so fully zero-code deployments stay possible.

Implementations MUST NOT block the instrumented call: return the destination URI immediately (content-addressed naming, e.g. SHA-256 of the bytes, makes it computable before any I/O) and move the bytes on a background worker. Returning `None` (backpressure, shutdown, policy) makes the caller fall back to the standard redaction behavior.

## Usage

To set up this configuration you can either:
- Set environment variables as specified above
- Define the configuration in code as shown below
- Do nothing and fall back to the default values
- Use a combination of the three, the order of precedence is:
  - Values set in the TraceConfig in code
  - Environment variables
  - Default values

### Python

If you are working in Python, and want to set up a configuration different than the default you can define the configuration in code as shown below, passing it to the `instrument()` method of your instrumentator (the example below demonstrates using the OpenAIInstrumentator)
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
    hide_input_audio=...,
    hide_output_audio=...,
    hide_input_files=...,
    hide_input_text=...,
    hide_output_text=...,
    hide_embeddings_vectors=...,
    hide_embeddings_text=...,
    base64_image_max_length=...,
    base64_media_max_length=...,
    blob_uploader=...,  # Experimental: uploads oversized base64 media, records a URI
    hide_prompts=...,  # Hides LLM prompts (completions API)
    hide_choices=...,  # Hides LLM choices (completions API outputs)
)

from openinference.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument(
    tracer_provider=tracer_provider,
    config=config,
)
```

### Javascript

If you are working in JavaScript, and want to set up a configuration different than the default you can define the configuration as shown below and pass it into any OpenInference instrumentation (the example below demonstrates using the OpenAIInstrumentation)

```typescript
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai"

/**
 * Everything left out of here will fallback to
 * environment variables then defaults
 */
const traceConfig = { hideInputs: true }

const instrumentation = new OpenAIInstrumentation({ traceConfig })
```
