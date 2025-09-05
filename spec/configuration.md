# OpenInference Configuration

In some situations, you may need to modify the observability level of your tracing. For instance, you may want to keep sensitive information from being logged for security reasons, or you may want to limit the size of the base64 encoded images logged to reduced payload size.

The OpenInference Specification defines a set of environment variables you can configure to suit your observability needs. In addition, the OpenInference Instrumentation Python package also offers convenience functions to do this in code without having to set environment variables, if that's what you prefer.

The possible settings are:

| Environment Variable Name                    | Effect                                                                                                                         | Type | Default |
|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|------|---------|
| OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS | Removes llm.invocation_parameters attribute entirely from spans                                                                | bool | False   |
| OPENINFERENCE_HIDE_INPUTS                    | Replaces input.value with `"__REDACTED__"` and removes input.mime_type                                                         | bool | False   |
| OPENINFERENCE_HIDE_OUTPUTS                   | Replaces output.value with `"__REDACTED__"` and removes output.mime_type                                                       | bool | False   |
| OPENINFERENCE_HIDE_INPUT_MESSAGES            | Removes all llm.input_messages attributes entirely from spans                                                                  | bool | False   |
| OPENINFERENCE_HIDE_OUTPUT_MESSAGES           | Removes all llm.output_messages attributes entirely from spans                                                                 | bool | False   |
| OPENINFERENCE_HIDE_INPUT_IMAGES              | Removes image URLs from llm.input_messages message content blocks                                                              | bool | False   |
| OPENINFERENCE_HIDE_INPUT_TEXT                | Replaces text content in llm.input_messages with `"__REDACTED__"`                                                              | bool | False   |
| OPENINFERENCE_HIDE_OUTPUT_TEXT               | Replaces text content in llm.output_messages with `"__REDACTED__"`                                                             | bool | False   |
| OPENINFERENCE_HIDE_EMBEDDINGS_VECTORS        | Replaces embedding.embeddings.*.embedding.vector values with `"__REDACTED__"`                                                  | bool | False   |
| OPENINFERENCE_HIDE_EMBEDDINGS_TEXT           | Replaces embedding.embeddings.*.embedding.text values with `"__REDACTED__"`                                                    | bool | False   |
| OPENINFERENCE_HIDE_PROMPTS                   | Replaces llm.prompts values with `"__REDACTED__"`                                                                              | bool | False   |
| OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH        | Truncates base64-encoded images to this length, replacing excess with `"__REDACTED__"`                                         | int  | 32,000  |

## Redacted Content

When content is hidden due to privacy configuration settings, the value `"__REDACTED__"` is used as a placeholder. This constant value allows consumers of the trace data to identify that content was intentionally hidden rather than missing or empty.

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
        hide_inputs=...,
        hide_outputs=...,
        hide_input_messages=...,
        hide_output_messages=...,
        hide_input_images=...,
        hide_input_text=...,
        hide_output_text=...,
        hide_embeddings_vectors=...,
        hide_embeddings_text=...,
        hide_prompts=...,
        base64_image_max_length=...,
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
