# Traces

Traces give us the big picture of what happens when a request is made to an LLM application. Whether your application is an agent or a chatbot a, traces are essential to understanding the full "path" a request takes in your application.

Let's explore this with three units of work, represented as Spans:

query span:

```json
{
    "name": "query",
    "context": {
        "trace_id": "ed7b336d-e71a-46f0-a334-5f2e87cb6cfc",
        "span_id": "f89ebb7c-10f6-4bf8-8a74-57324d2556ef"
    },
    "span_kind": "CHAIN",
    "parent_id": null,
    "start_time": "2023-09-07T12:54:47.293922-06:00",
    "end_time": "2023-09-07T12:54:49.322066-06:00",
    "status_code": "OK",
    "status_message": "",
    "attributes": {
        "input.value": "Is anybody there?",
        "input.mime_type": "text/plain",
        "output.value": "Yes, I am here.",
        "output.mime_type": "text/plain"
    },
    "events": []
}
```

This is the root span, denoting the beginning and end of the entire operation. Note that it has a trace_id field indicating the trace, but has no parent_id. That's how you know it's the root span.

LLM span:

```json
{
    "name": "llm",
    "context": {
        "trace_id": "ed7b336d-e71a-46f0-a334-5f2e87cb6cfc",
        "span_id": "ad67332a-38bd-428e-9f62-538ba2fa90d4"
    },
    "span_kind": "LLM",
    "parent_id": "f89ebb7c-10f6-4bf8-8a74-57324d2556ef",
    "start_time": "2023-09-07T12:54:47.597121-06:00",
    "end_time": "2023-09-07T12:54:49.321811-06:00",
    "status_code": "OK",
    "status_message": "",
    "attributes": {
        "llm.input_messages": [
            {
                "message.role": "system",
                "message.content": "You are an expert Q&A system that is trusted around the world.\nAlways answer the query using the provided context information, and not prior knowledge.\nSome rules to follow:\n1. Never directly reference the given context in your answer.\n2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines."
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

This span encapsulates a sub task, like invoking an LLM, and its parent is the hello span. Note that it shares the same trace_id as the root span, indicating it's a part of the same trace. Additionally, it has a parent_id that matches the span_id of the query span.

These two blocks of JSON all share the same trace_id, and the parent_id field represents a hierarchy. That makes it a Trace!

Another thing you'll note is that each Span looks like a structured log. That's because it kind of is! One way to think of Traces is that they're a collection of structured logs with context, correlation, hierarchy, and more baked in. However, these "structured logs" can come from different parts of your application stack such as a vector store retriever or a langchain tool. This is what allows tracing to represent an end-to-end view of any system.

To understand how tracing in OpenInference works, let's look at a list of components that will play a part in instrumenting our code.

Tracer
A Tracer creates spans containing more information about what is happening for a given operation, such as a request in a service.

Trace Exporters
Trace Exporters send traces to a consumer. This consumer can be standard output for debugging and development-time or a OpenInference Collector.

## Spans

A span represents a unit of work or operation. Spans are the building blocks of Traces. In OpenInference, they include the following information:

-   Name
-   Parent span ID (empty for root spans)
-   Start and End Timestamps
-   Span Context
-   Attributes
-   Span Events
-   Span Status
-   Sample span:

```json
{
    "name": "query",
    "context": {
        "trace_id": "ed7b336d-e71a-46f0-a334-5f2e87cb6cfc",
        "span_id": "f89ebb7c-10f6-4bf8-8a74-57324d2556ef"
    },
    "span_kind": "CHAIN",
    "parent_id": null,
    "start_time": "2023-09-07T12:54:47.293922-06:00",
    "end_time": "2023-09-07T12:54:49.322066-06:00",
    "status_code": "OK",
    "status_message": "",
    "attributes": {
        "input.value": "Hello?",
        "input.mime_type": "text/plain",
        "output.value": "I am here.",
        "output.mime_type": "text/plain"
    },
    "events": []
}
```

Spans can be nested, as is implied by the presence of a parent span ID: child spans represent sub-operations. This allows spans to more accurately capture the work done in an application.

### Span Context

Span context is an immutable object on every span that contains the following:

-   The Trace ID representing the trace that the span is a part of
-   The span's Span ID

Because Span Context contains the Trace ID, it is used when creating Span Links.

### Attributes

Attributes are key-value pairs that contain metadata that you can use to annotate a Span to carry information about the operation it is tracking.

For example, if a span invokes an LLM, you can capture the model name, the invocation parameters, the token count, and so on.

Attributes have the following rules:

-   Keys must be non-null string values
-   Values must be a non-null string, boolean, floating point value, integer, or an array of these values
    Additionally, there are Semantic Attributes, which are known naming conventions for metadata that is typically present in common operations. It's helpful to use semantic attribute naming wherever possible so that common kinds of metadata are standardized across systems. See [semantic conventions](./semantic_conventions.md) for more information.

### Span Events

A Span Event can be thought of as a structured log message (or annotation) on a Span, typically used to denote a meaningful, singular point in time during the Span's duration.

For example, consider two scenarios with an LLM:

-   Tracking a LLM execution time
-   Denoting when the first token is sent

A Span is best used to the first scenario because it's an operation with a start and an end.

A Span Event is best used to track the second scenario because it represents a meaningful, singular point in time.

### Span Status

A status will be attached to a span. Typically, you will set a span status when there is a known error in the application code, such as an exception. A Span Status will be tagged as one of the following values:

-   Unset
-   Ok
-   Error
-   When an exception is handled, a Span status can be set to Error.

### Span Kind

When a span is created, it is one of Chain, Retriever, Reranker, LLM, Embedding, Agent, or Tool. This span kind provides a hint to the tracing backend as to how the trace should be assembled.

Note that `span_kind` is a OpenTelemetry concept and thus conflicts with the OpenInference concept of `span_kind`. When OTLP is used as the transport, the OpenInference `span_kind` is stored in the `openinference.span.kind` attribute.

#### Chain

A Chain is a starting point or a link between different LLM application steps. For example, a Chain span could be used to represent the beginning of a request to an LLM application or the glue code that passes context from a retriever to and LLM call.

#### Retriever

A Retriever is a span that represents a data retrieval step. For example, a Retriever span could be used to represent a call to a vector store or a database.

#### Reranker

A Reranker is a span that represents the reranking of a set of input documents. For example, a cross-encoder may be used to compute the input documents' relevance scores with respect to a user query, and the top K documents with the highest scores are then returned by the Reranker.

#### LLM

An LLM is a span that represents a call to an LLM. For example, an LLM span could be used to represent a call to OpenAI or Llama.

#### Embedding

An Embedding is a span that represents a call to an LLM for an embedding. For example, an Embedding span could be used to represent a call OpenAI to get an ada-2 embedding for retrieval.

#### Tool

A Tool is a span that represents a call to an external tool such as a calculator or a weather API.

#### Agent

A span that encompasses calls to LLMs and Tools. An agent describes a reasoning block that acts on tools using the guidance of an LLM.

#### Guardrail

A span that represents calls to a component to protect against jailbreak user input prompts by taking action to modify or reject an LLM's response if it contains undesirable content. For example, a Guardrail span could involve checking if an LLM's output response contains inappropriate language, via a custom or external guardrail library, and then amending the LLM response to remove references to the inappropriate language.
