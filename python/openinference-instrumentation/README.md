# OpenInference Instrumentation

[![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation.svg)](https://pypi.python.org/pypi/openinference-instrumentation) 

Utility functions for OpenInference instrumentation.

## Installation

```shell
pip install openinference-instrumentation
```

## Customizing Spans

The `openinference-instrumentation` package offers utilities to track important application metadata such as sessions and metadata using Python context managers:

* `using_session`: to specifiy a session ID to track multi-turn conversations with a user
* `using_user`: to specifiy a user ID.
* `using_metadata`: to add custom metadata.
* `using_tag`: to add tags.
* `using_prompt_template`: to reflect the prompt template used.
* `using_attributes`: to specify all the above at once.
  
For example:
```python
from openinference.instrumentation import using_attributes
tags = ["tag_1", "tag_2", ...]
metadata = {
    "key-1": value_1,
    "key-2": value_2,
    ...
}
prompt_template = "Please describe the weather forecast for {city} on {date}"
prompt_template_variables = {"city": "Johannesburg", "date":"July 11"}
prompt_template_version = "v1.0"
with using_attributes(
    session_id="my-session-id",
    user_id="my-user-id",
    metadata=metadata,
    tags=tags,
    prompt_template=prompt_template,
    prompt_template_version=prompt_template_version,
    prompt_template_variables=prompt_template_variables,
):
    # Calls within this block will generate spans with the attributes:
    # "session.id" = "my-session-id"
    # "user.id" = "my-user-id"
    # "metadata" = "{\"key-1\": value_1, \"key-2\": value_2, ... }" # JSON serialized
    # "tag.tags" = "["tag_1","tag_2",...]"
    # "llm.prompt_template.template" = "Please describe the weather forecast for {city} on {date}"
    # "llm.prompt_template.variables" = "{\"city\": \"Johannesburg\", \"date\": \"July 11\"}" # JSON serialized
    # "llm.prompt_template.version " = "v1.0"
    ...
```
You can read more about this in our [docs](https://docs.arize.com/phoenix/tracing/how-to-tracing/customize-spans).


## Tracing Configuration


The core `openinference-instrumentation` package offers the `TraceConfig` class, which helps set up privacy configurations to mask specific fields in your spans. For instance, you may want to keep sensitive information from being logged for security reasons, or you may want to limit the size of the base64 encoded images logged to reduced payload size.

In addition, you an also use environment variables, read more [here](../../spec/configuration.md). The following is an example of using the `TraceConfig` object:

```python
from openinference.instrumentation import TraceConfig
config = TraceConfig(
    hide_inputs=hide_inputs,
    hide_outputs=hide_outputs,
    hide_input_messages=hide_input_messages,
    hide_output_messages=hide_output_messages,
    hide_input_images=hide_input_images,
    hide_input_text=hide_input_text,
    hide_output_text=hide_output_text,
    base64_image_max_length=base64_image_max_length,
)
tracer_provider=...
# This example uses the OpenAIInstrumentor, but it works with any of our auto instrumentors
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)
```