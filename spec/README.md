# OpenInference Specification

OpenInference is a semantic convention specification for AI application observability, built on [OpenTelemetry](https://opentelemetry.io/). It standardizes how LLM calls, agent reasoning steps, tool invocations, retrieval operations, and other AI-specific workloads are represented as distributed traces.

## Motivation

OpenTelemetry defines a universal wire format and SDK model for distributed tracing, but its attribute model is intentionally generic. AI applications present a distinct set of observability requirements that general-purpose conventions do not address:

- **Structured inputs and outputs** — LLM calls carry multi-turn message arrays, system prompts, tool definitions, and multimodal content; a single string `input.value` is insufficient.
- **Token economics** — Prompt and completion token counts, along with cached and reasoning token breakdowns, are first-class operational metrics, not afterthoughts.
- **Agentic control flow** — Modern AI systems route through reasoning loops, delegate to sub-agents, invoke tools, and query retrieval systems. Each hop needs a consistent identity and span-kind taxonomy for the trace to be interpretable.
- **Privacy sensitivity** — Prompts and completions frequently contain personal information and must be maskable before export, with per-field granularity.
- **Nondeterminism** — LLM outputs are stochastic; traces must carry enough context to reproduce — or at least explain — a particular execution.

OpenInference solves these problems by defining a concrete attribute schema and span-kind taxonomy on top of OpenTelemetry spans. Every OpenInference trace is a valid OTLP trace; the conventions give attribute names their AI-specific meaning.

## Data Model

### Traces

A trace records the full execution path of a request — from the user's initial input through every LLM call, tool invocation, and retrieval step to the final response. Traces are trees of spans connected by parent–child relationships. The root span typically represents an agent turn or pipeline invocation; child spans represent individual operations within it.

### Spans

A span is the atomic unit of work: one LLM call, one tool execution, one retrieval query, one embedding generation. Every span carries:

| Field | Description |
|---|---|
| Name | Human-readable operation name (e.g., `ChatCompletion`, `web_search`) |
| Start / end time | Wall-clock timestamps with nanosecond precision |
| `openinference.span.kind` | The role of this operation in the pipeline (see [Span Kinds](#span-kinds)) |
| Attributes | Typed key/value pairs capturing inputs, outputs, configuration, and cost |
| Status | `OK`, `ERROR`, or `UNSET` |

### Span Kinds

The `openinference.span.kind` attribute classifies what an operation does, enabling observability platforms to render traces with AI-aware visualizations and aggregations:

| Kind | Description |
|---|---|
| `LLM` | A call to a language model API. Carries input messages, model parameters, output messages, and token counts. |
| `AGENT` | A reasoning step in an autonomous agent. May spawn child spans for tool calls, retrievals, or nested LLM calls. |
| `CHAIN` | A deterministic sequence of operations such as prompt formatting, post-processing, or orchestration logic. |
| `TOOL` | Execution of a function or external API called by a language model. |
| `RETRIEVER` | A query to a vector store, search engine, or knowledge base. |
| `RERANKER` | A reranking model that reorders a candidate set of documents by relevance. |
| `EMBEDDING` | Generation of vector embeddings from text or other content. |
| `GUARDRAIL` | An input or output moderation check. |
| `EVALUATOR` | An automated evaluation of a model response (e.g., LLM-as-judge). |
| `PROMPT` | A named prompt template invocation. |

### Attributes

Attributes are typed key/value pairs attached to spans following a structured naming convention. They are the primary payload of OpenInference: they carry the prompt, the response, the model name, the retrieved documents, the tool arguments, and everything else needed to understand and reproduce a given execution.

Attribute names use dot-separated namespaces (e.g., `llm.input_messages`, `llm.token_count.prompt`). List-valued attributes use zero-based integer indices in flattened form (e.g., `llm.input_messages.0.message.role`).

The [Semantic Conventions](./semantic_conventions.md) document is the authoritative reference for all attribute names, types, and meanings.

## Specifications

### Core

- [Traces](./traces.md) — Trace structure, span hierarchy, and context propagation
- [Semantic Conventions](./semantic_conventions.md) — Complete attribute reference
- [Configuration](./configuration.md) — Environment variables, privacy controls, and data masking

### Span Types

- [LLM Spans](./llm_spans.md) — Attributes for language model calls: messages, token counts, model parameters, and tool definitions
- [Embedding Spans](./embedding_spans.md) — Attributes for vector embedding generation

### Attribute Conventions

- [Tool Calling](./tool_calling.md) — Function/tool call and result representation
- [Multimodal Attributes](./multimodal_attributes.md) — Image, audio, and mixed-content messages

## Notation Conventions and Compliance

The keywords "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this specification are to be interpreted as described in [BCP 14](https://tools.ietf.org/html/bcp14) [[RFC2119](https://tools.ietf.org/html/rfc2119)] [[RFC8174](https://tools.ietf.org/html/rfc8174)] when, and only when, they appear in all capitals, as shown here.

An implementation is **compliant** if it satisfies all "MUST", "MUST NOT", "REQUIRED", "SHALL", and "SHALL NOT" requirements defined in this specification. An implementation that fails to satisfy any such requirement is **not compliant**.

## Project Naming

The official project name is "OpenInference" (no space between "Open" and "Inference").
