# OpenInference Specification

OpenInference is a semantic convention specification for AI application observability, built on [OpenTelemetry](https://opentelemetry.io/). It standardizes how LLM calls, agent reasoning steps, tool invocations, retrieval operations, and other AI-specific workloads are represented as distributed traces.

## Motivation

OpenTelemetry defines a universal wire format and SDK model for distributed tracing, but its attribute model is intentionally generic. AI applications present a distinct set of observability requirements that general-purpose conventions do not address:

- **Structured inputs and outputs** — LLM calls carry multi-turn message arrays, system prompts, tool definitions, and multimodal content; a single string `input.value` is insufficient.
- **Token economics** — Prompt and completion token counts, along with cached and reasoning token breakdowns, are first-class operational metrics, not afterthoughts.
- **Agentic control flow** — Modern AI systems route through reasoning loops, delegate to sub-agents, invoke tools, and query retrieval systems. Each hop needs a consistent identity and span-kind taxonomy for the trace to be interpretable.
- **Privacy sensitivity** — Prompts and completions frequently contain personal information and must be maskable before export, with per-field granularity.
- **Nondeterminism** — LLM outputs are stochastic; traces must carry enough context to reproduce — or at least explain — a particular execution.
- **Quality feedback** — Human reviewers, LLM judges, and code-based evaluators produce scores, labels, and explanations that must remain associated with the operation they assess as telemetry moves between systems.

OpenInference solves these problems by defining a concrete attribute schema and span-kind taxonomy on top of OpenTelemetry spans. Every OpenInference trace is a valid OTLP trace; the conventions give attribute names their AI-specific meaning.

## Data Model

OpenInference adds no new signal or wire format. It defines conventions for OpenTelemetry traces:

- A **trace** records the full execution path of a request, from the user's input through every LLM call, tool
  invocation, and retrieval step to the final response. It is a tree of spans connected by parent–child
  relationships; the root span typically represents an agent turn or pipeline invocation.
- A **span** is the atomic unit of work: one LLM call, one tool execution, one retrieval query, one embedding
  generation. See [Traces and Spans](./general/traces.md).
- The **span kind**, recorded in the `openinference.span.kind` attribute, classifies what an operation does (`LLM`,
  `AGENT`, `TOOL`, `RETRIEVER`, and so on), so observability platforms can render traces with AI-aware
  visualizations and aggregations. See [Span Kinds](./general/span-kinds.md).
- **Attributes** are typed key–value pairs on a span. They carry the prompt, the response, the model name, the
  retrieved documents, the tool arguments, and everything else needed to understand and reproduce a given execution.
  Names use dot-separated namespaces (`llm.token_count.prompt`), and lists are flattened with zero-based indices
  (`llm.input_messages.0.message.role`). See [Attribute Conventions](./attributes/conventions.md).

## Specification

### General

- [Traces and Spans](./general/traces.md) — How OpenInference uses the OpenTelemetry trace data model
- [Span Kinds](./general/span-kinds.md) — The `openinference.span.kind` values and what each represents

### Attributes

- [Attribute Conventions](./attributes/conventions.md) — Naming, value types, list flattening, and requirement levels
- [Semantic Conventions](./attributes/semantic-conventions.md) — Every attribute, grouped by namespace, with well-known values
- [Semantic Convention Registry](./attributes/registry.md) — The same conventions as a machine-readable, versioned
  [Weaver](https://github.com/open-telemetry/weaver) registry: where it is published and how to pull it
- [Context Attributes](./attributes/context.md) — Session, user, metadata, tag, and prompt template attributes
  propagated to every span

### Span Conventions

- [LLM Spans](./spans/llm.md) — Language model calls: model identification, messages, token counts, and cost
- [Embedding Spans](./spans/embedding.md) — Vector embedding generation
- [Decision Spans](./spans/decision.md) — Decision model calls: model, system, and provider identification

### Messages and Content

- [Messages](./content/messages.md) — Input and output messages, content items, reasoning content, and the
  completions API
- [Multimodal Content](./content/multimodal.md) — Image, audio, video, and document content
- [Tool Calling](./content/tool-calling.md) — Tool definitions, tool calls, and tool results

### Feedback

- [Annotations and Evaluations](./feedback/annotations.md) — Span-, trace-, and session-scoped feedback from human,
  LLM, and code judges

### Configuration

- [Trace Configuration](./configuration/trace-config.md) — Environment variables and `TraceConfig` settings
- [Data Masking](./configuration/data-masking.md) — What each privacy setting hides
- [Blob Upload](./configuration/blob-upload.md) — Uploading oversized media to external storage (experimental)
- [GenAI Semantic Conventions Export](./configuration/genai-semconv.md) — Also emitting OTel `gen_ai.*` attributes (experimental)

## Notation Conventions and Compliance

The keywords "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this specification are to be interpreted as described in [BCP 14](https://tools.ietf.org/html/bcp14) [[RFC2119](https://tools.ietf.org/html/rfc2119)] [[RFC8174](https://tools.ietf.org/html/rfc8174)] when, and only when, they appear in all capitals, as shown here.

An implementation is **compliant** if it satisfies all "MUST", "MUST NOT", "REQUIRED", "SHALL", and "SHALL NOT" requirements defined in this specification. An implementation that fails to satisfy any such requirement is **not compliant**.

## Project Naming

The official project name is "OpenInference" (no space between "Open" and "Inference").
