# OpenInference Instrumentation

[![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation.svg)](https://pypi.python.org/pypi/openinference-instrumentation) 

Utility functions for OpenInference instrumentation.

## Installation

```shell
pip install openinference-instrumentation
```

## Annotation and Evaluation Attributes

Use `get_annotation_attributes` and `get_evaluation_attributes` to turn typed
`Annotation` objects into flattened OpenInference span attributes. Both helpers
support `"span"` (the default), `"trace"`, and `"session"` scopes and assign
contiguous collection indices in input order.

```python
from openinference.instrumentation import (
    Annotation,
    get_annotation_attributes,
    get_evaluation_attributes,
)

span_annotations = get_annotation_attributes(
    annotations=[
        Annotation(
            name="hallucination",
            label="factual",
            explanation="Every claim is supported by the retrieved documents.",
            annotator_kind="LLM",
            identifier="judge-v2",
            metadata={"rubric_version": 2},
        )
    ]
)

trace_evaluations = get_evaluation_attributes(
    evaluations=[Annotation(name="correctness", score=0.95)],
    scope="trace",
)

span.set_attributes({**span_annotations, **trace_evaluations})
```

`Annotation` has these fields:

- `name` (required): criterion or metric name.
- At least one of `score`, `label`, or `explanation` (required by the helpers).
- `annotator_kind`: conventionally `"HUMAN"`, `"LLM"`, or `"CODE"`; custom values are allowed.
- `identifier`: stable producer-assigned result identifier.
- `metadata`: a dictionary to JSON-serialize, or an already serialized JSON object string.

The evaluation helper uses the same `Annotation` model because evaluation is an
alternative attribute terminology for annotations. It emits `evaluations.*`
instead of `annotations.*`. Trace and session scopes add the corresponding
`trace.` or `session.` prefix. Session-scoped annotations also require the
carrying span to have `session.id`; post-hoc span and trace annotations require
the target Span Link described in the
[annotation specification](../../spec/annotations.md).

## Customizing Spans

The `openinference-instrumentation` package offers utilities to track important application metadata such as sessions and metadata using Python context managers:

* `using_session`: to specify a session ID to track and group a multi-turn conversation with a user
* `using_user`: to specify a user ID to track different conversations with a given user
* `using_metadata`: to add custom metadata, that can provide extra information that supports a wide range of operational needs
* `using_tag`: to add tags, to help filter on specific keywords
* `using_prompt_template`: to reflect the prompt template used, with its version and variables. This is useful for prompt template management
* `using_attributes`: it helps handling multiple of the previous options at once in a concise manner
  
For example:

```python
from openinference.instrumentation import using_attributes
tags = ["business_critical", "simple", ...]
metadata = {
    "country": "United States",
    "topic":"weather",
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

Each helper also works as a decorator. The attributes stay attached for the whole call,
including across `await` suspension points of `async def` functions and while the body of a
generator or `async def` generator runs (without leaking into the code that consumes it):

```python
from openinference.instrumentation import using_session, using_user

@using_session("my-session-id")
@using_user("my-user-id")
async def answer(question: str) -> str:
    # Spans created here, and by any awaited instrumented call, carry
    # "session.id" = "my-session-id" and "user.id" = "my-user-id"
    ...
```

When combining them with span-creating decorators such as `tracer.agent` or `tracer.tool`, put
the `using_*` decorators on top: the attributes are copied onto a span when it starts, so they
have to be attached before the span-creating decorator runs.

See [`examples/async_context_attribute_decorators.py`](examples/async_context_attribute_decorators.py) for a runnable example that exports to a local Phoenix server.

You can read more about this in our [docs](https://docs.arize.com/phoenix/tracing/how-to-tracing/customize-spans).

## Tracing Configuration

This package contains the central `TraceConfig` class, which lets you specify a tracing configuration that lets you control settings like data privacy and payload sizes. For instance, you may want to keep sensitive information from being logged for security reasons, or you may want to limit the size of the base64 encoded images logged to reduced payload size.

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
tracer_provider = ...
# This example uses the OpenAIInstrumentor, but it works with any of our auto instrumentors
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)
```
