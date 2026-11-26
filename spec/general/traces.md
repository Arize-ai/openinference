# Traces and Spans

OpenInference uses the [OpenTelemetry trace data model](https://opentelemetry.io/docs/concepts/signals/traces/)
unchanged. Every OpenInference trace is a valid OTLP trace; the conventions in this specification give its span
attributes AI-specific meaning. This page describes the parts of the data model that OpenInference relies on.

## Traces

A trace records the full path a request takes through an AI application, from the user's input through every LLM
call, tool invocation, and retrieval step to the final response. A trace is a tree of spans that share a trace ID and
are connected by parent–child relationships.

Consider a request that is handled by a `CHAIN` span, which calls an LLM. The root span (shown here in a logical JSON
form) has a trace ID but no parent:

```json
{
    "name": "query",
    "context": {
        "trace_id": "ed7b336d-e71a-46f0-a334-5f2e87cb6cfc",
        "span_id": "f89ebb7c-10f6-4bf8-8a74-57324d2556ef"
    },
    "span_kind": "SPAN_KIND_INTERNAL",
    "parent_id": null,
    "start_time": "2023-09-07T12:54:47.293922-06:00",
    "end_time": "2023-09-07T12:54:49.322066-06:00",
    "status_code": "OK",
    "status_message": "",
    "attributes": {
        "openinference.span.kind": "CHAIN",
        "input.value": "Is anybody there?",
        "input.mime_type": "text/plain",
        "output.value": "Yes, I am here.",
        "output.mime_type": "text/plain"
    },
    "events": []
}
```

The LLM call is a child span. It shares the root span's trace ID, and its `parent_id` is the root span's span ID:

```json
{
    "name": "llm",
    "context": {
        "trace_id": "ed7b336d-e71a-46f0-a334-5f2e87cb6cfc",
        "span_id": "ad67332a-38bd-428e-9f62-538ba2fa90d4"
    },
    "span_kind": "SPAN_KIND_INTERNAL",
    "parent_id": "f89ebb7c-10f6-4bf8-8a74-57324d2556ef",
    "start_time": "2023-09-07T12:54:47.597121-06:00",
    "end_time": "2023-09-07T12:54:49.321811-06:00",
    "status_code": "OK",
    "status_message": "",
    "attributes": {
        "openinference.span.kind": "LLM",
        "llm.input_messages": [
            {
                "message.role": "system",
                "message.content": "You are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided context information, and not prior knowledge."
            },
            {
                "message.role": "user",
                "message.content": "Hello?"
            }
        ],
        "output.value": "assistant: Yes I am here",
        "output.mime_type": "text/plain"
    },
    "events": []
}
```

Each span reads like a structured log with context, correlation, and hierarchy built in. Because the spans can come
from different parts of the stack (an agent framework, an LLM client, a vector store), the trace gives an end-to-end
view of the system.

## Spans

A span represents one unit of work: one LLM call, one tool execution, one retrieval query. A span carries:

- A name
- A span context (trace ID and span ID)
- A parent span ID, empty for a root span
- Start and end timestamps
- Attributes
- Span events
- Span links
- A status

### Span Context

The span context is an immutable object on every span holding the ID of the trace the span belongs to and the span's
own ID. Span links refer to other spans by their span context.

### Attributes

Attributes are key–value pairs that describe the operation a span tracks: for an LLM call, the model name, the
invocation parameters, the messages, and the token counts. Keys are non-null strings; values are non-null strings,
booleans, floating point numbers, integers, or arrays of one of these.

OpenInference attributes are the payload of this specification. See [Attribute Conventions](../attributes/conventions.md)
for how they are named and encoded, and the [Semantic Conventions](../attributes/semantic-conventions.md) for every attribute.

### Span Events

A span event is a structured log message on a span that marks a meaningful, singular point in time during the span's
duration. An LLM call's execution time is best represented by the span itself, since it has a start and an end; the
moment the first token arrives is best represented by an event.

### Span Links

A span link connects a span to another span, possibly in another trace, without making it a parent. OpenInference uses
links to attach [post-hoc feedback](../feedback/annotations.md#post-hoc-feedback) to the span it describes.

### Span Status

A span's status is `UNSET`, `OK`, or `ERROR`. Instrumentation sets `ERROR` when the operation fails, for example when an
exception is raised and not handled, and records the exception as an OpenTelemetry `exception` span event carrying
the [`exception.*` attributes](../attributes/semantic-conventions.md#exception).

### Span Kind

OpenTelemetry's `SpanKind` (`INTERNAL`, `CLIENT`, `SERVER`, `PRODUCER`, `CONSUMER`) describes a span's position in a
network call. OpenInference defines its own span kinds, such as `LLM`, `TOOL`, and `RETRIEVER`, that describe the role
of the operation in an AI application. Because the names would conflict, the OpenInference span kind is stored in the
`openinference.span.kind` attribute. See [Span Kinds](./span-kinds.md).

## Instrumentation Components

- A **tracer** creates spans for the operations an application performs.
- A **span processor** and **exporter** send finished spans to a consumer: standard output during development, or an
  OTLP endpoint such as a collector or an observability backend.

OpenInference instrumentation uses the standard OpenTelemetry tracer provider, processors, and exporters.
