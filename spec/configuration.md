# OpenInference Configuration

In some situations, you may need to modify the observability level of your tracing. For instance, you may want to keep sensitive information from being logged for security reasons, or you may want to limit the size of the base64 encoded images logged to reduced payload size.

The OpenInference Specification defines a set of environment variables you can configure to suit your observability needs. In addition, the OpenInference Instrumentation Python package also offers convenience functions to do this in code without having to set environment variables, if that's what you prefer.

The possible settings are:

| Environment Variable Name                    | Effect                                                                                                                         | Type | Default |
|----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|------|---------|
| OPENINFERENCE_HIDE_LLM_INVOCATION_PARAMETERS | Hides LLM invocation parameters (independent of input/output hiding)                                                           | bool | False   |
| OPENINFERENCE_HIDE_INPUTS                    | Hides input.value and all input messages (input messages are hidden if either HIDE_INPUTS OR HIDE_INPUT_MESSAGES is true)      | bool | False   |
| OPENINFERENCE_HIDE_OUTPUTS                   | Hides output.value and all output messages (output messages are hidden if either HIDE_OUTPUTS OR HIDE_OUTPUT_MESSAGES is true) | bool | False   |
| OPENINFERENCE_HIDE_INPUT_MESSAGES            | Hides all input messages (independent of HIDE_INPUTS)                                                                          | bool | False   |
| OPENINFERENCE_HIDE_OUTPUT_MESSAGES           | Hides all output messages (independent of HIDE_OUTPUTS)                                                                        | bool | False   |
| OPENINFERENCE_HIDE_INPUT_IMAGES              | Hides images from input messages (only applies when input messages are not already hidden)                                     | bool | False   |
| OPENINFERENCE_HIDE_INPUT_TEXT                | Hides text from input messages (only applies when input messages are not already hidden)                                       | bool | False   |
| OPENINFERENCE_HIDE_PROMPTS                   | Hides LLM prompts (completions API)                                                                                            | bool | False   |
| OPENINFERENCE_HIDE_OUTPUT_TEXT               | Hides text from output messages (only applies when output messages are not already hidden)                                     | bool | False   |
| OPENINFERENCE_HIDE_CHOICES                   | Hides LLM choices (completions API outputs)                                                                                    | bool | False   |
| OPENINFERENCE_HIDE_EMBEDDING_VECTORS         | Hides embedding vectors                                                                                                        | bool | False   |
| OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH        | Limits characters of a base64 encoding of an image                                                                             | int  | 32,000  |

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
        hide_embedding_vectors=...,
        base64_image_max_length=...,
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
