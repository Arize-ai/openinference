# OpenInference Configuration

In some situations, you may need to modify the observability level of your tracing. For instance, you may want to keep sensitive information from being logged for security reasons, or you may want to limit the size of the base64 encoded images logged to reduced payload size.

The OpenInference Specification defines a set of environment variables you can configure to suit your observability needs. In addition, the OpenInference Instrumentation Python package also offers convenience functions to do this in code without having to set environment variables, if that's what you prefer.

The possible settings are:
| Environment Variable Name             | Effect                                             | Type | Default |
|---------------------------------------|----------------------------------------------------|------|---------|
| OPENINFERENCE_HIDE_INPUTS             | Hides input value, all input messages & embedding input text             | bool | False   |
| OPENINFERENCE_HIDE_OUTPUTS            | Hides output value & all output messages           | bool | False   |
| OPENINFERENCE_HIDE_INPUT_MESSAGES     | Hides all input messages & embedding input text                           | bool | False   |
| OPENINFERENCE_HIDE_OUTPUT_MESSAGES    | Hides all output messages                          | bool | False   |
| PENINFERENCE_HIDE_INPUT_IMAGES        | Hides images from input messages                   | bool | False   |
| OPENINFERENCE_HIDE_INPUT_TEXT         | Hides text from input messages & input embeddings                     | bool | False   |
| OPENINFERENCE_HIDE_OUTPUT_TEXT        | Hides text from output messages                    | bool | False   |
| OPENINFERENCE_HIDE_EMBEDDING_VECTORS  | Hides returned embedding vectors                   | bool | False   |
| OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH | Limits characters of a base64 encoding of an image | int  | 32,000  |

## Setup a Configuration in Python

If you are working in Python, and want to set up a configuration different than the default values you can:
- Create the environment variables with the desired values, or
- Define the configuration in code as shown below, passing it to the `instrument()` method of your instrumentator (the example below demonstrates using the OpenAiInstrumentator)
    ```python
    from openinference.instrumentation import TraceConfig
    config = TraceConfig(        
        hide_inputs=...,
        hide_outputs=...,
        hide_input_messages=...,
        hide_output_messages=...,
        hide_input_images=...,
        hide_input_text=...,
        hide_output_text=...,
        hide_embedding_vectors=...,
        base64_image_max_length=...,
    )

    from openinference.instrumentation.openai import OpenAIInstrumentor
    OpenAIInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=config,
    )
    ```
- Do nothing and the default values will be used.
