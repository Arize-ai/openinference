# OpenInference Tracing Specification

This specification covers the OpenInference Tracing specification for capturing and storing LLM application executions. It is designed to be a category of telemetry data that is used to understand the execution of LLMs and the surrounding application context such as retrieval from vector stores and the usage of external tools such as search engines or APIs.

## Spans

A span represents a unit of work or operation. It tracks specific operations that a request makes, painting a picture of what happened during the time in which that operation was executed.

A span contains name, time-related data, structured log messages, and other metadata (that is, Attributes) to provide information about the operation it tracks.

## Traces

A trace records the paths taken by requests (made by an application or end-user) as they propagate through multiple steps.

Without tracing, it is challenging to pinpoint the cause of performance problems in a system.

It improves the visibility of our application or system’s health and lets us debug behavior that is difficult to reproduce locally. Tracing is essential for LLM applications, which commonly have nondeterministic problems or are too complicated to reproduce locally.

Tracing makes debugging and understanding LLM applications less daunting by breaking down what happens within a request as it flows through a system.

A trace is made of one or more spans. The first span represents the root span. Each root span represents a request from start to finish. The spans underneath the parent provide a more in-depth context of what occurs during a request (or what steps make up a request).

## Specifications

-   [Traces](./traces.md)
-   [Semantic Conventions](./semantic_conventions.md)

## Notation Conventions and Compliance

The keywords "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD",
"SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in the
specification are to be interpreted as described in [BCP
14](https://tools.ietf.org/html/bcp14)
[[RFC2119](https://tools.ietf.org/html/rfc2119)]
[[RFC8174](https://tools.ietf.org/html/rfc8174)] when, and only when, they
appear in all capitals, as shown here.

An implementation of the specification is not compliant if it fails to
satisfy one or more of the "MUST", "MUST NOT", "REQUIRED", "SHALL", or "SHALL
NOT" requirements defined in the specification. Conversely, an
implementation of the specification is compliant if it satisfies all the
"MUST", "MUST NOT", "REQUIRED", "SHALL", and "SHALL NOT" requirements defined in
the specification.

## Project Naming

-   The official project name is "OpenInference Tracing" (with no space between "Open" and
    "Inference").
