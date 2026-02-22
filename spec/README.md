# OpenInference Tracing Specification

OpenInference Tracing is a set of conventions for capturing observability data from AI/ML applications. It is built on top of [OpenTelemetry](https://opentelemetry.io/) and extends it with AI-specific semantic conventions — standardized attribute names and span kinds that describe LLM calls, embeddings, retrievals, tool use, and other operations common in modern AI systems.

## Why OpenInference?

OpenTelemetry provides a general-purpose, vendor-neutral framework for distributed tracing. OpenInference adds a semantic layer on top of it specifically for AI/ML workloads:

- **Standardized span kinds** — `LLM`, `EMBEDDING`, `RETRIEVER`, `CHAIN`, `AGENT`, `TOOL`, `RERANKER`, `GUARDRAIL`, `EVALUATOR`, and `PROMPT` describe the role of each operation in an AI pipeline.
- **Standardized attributes** — Consistent names for model parameters, token counts, input/output messages, tool calls, retrieved documents, and more.
- **Privacy controls** — Built-in configuration for masking sensitive data (prompts, inputs, outputs, images) before export.
- **Interoperability** — Any tracing backend that understands OTLP can consume OpenInference traces; observability platforms add richer AI-aware analysis on top.

## How to Use This Spec

Start with the Core Specifications to understand the tracing model, then refer to the Span Type Specifications and Attribute Conventions for the details relevant to your implementation.

## Specifications

### Core Specifications
- [Traces](./traces.md) - Core tracing concepts and structure
- [Semantic Conventions](./semantic_conventions.md) - Complete list of attributes and their meanings
- [Configuration](./configuration.md) - Environment variables and privacy settings

### Span Type Specifications
- [LLM Spans](./llm_spans.md) - Large Language Model operation spans
- [Embedding Spans](./embedding_spans.md) - Vector embedding generation spans

### Attribute Conventions
- [Tool Calling](./tool_calling.md) - Function/tool calling conventions
- [Multimodal Attributes](./multimodal_attributes.md) - Image, audio, and multimodal content representation


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
