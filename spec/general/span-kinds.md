# Span Kinds

The `openinference.span.kind` attribute is **required** on every OpenInference span. It names the role the operation
plays in an AI application, which tells a tracing backend how to assemble and present the trace. It is distinct from
OpenTelemetry's own `SpanKind`; see [Traces and Spans](./traces.md#span-kind).

<!-- weaver . as $root | {span_kinds: [$root.registry.attributes[] | select(.key == "openinference.span.kind") | .type.members[] | . as $member | ($root.registry.spans[] | select(.type == ("openinference." + $member.id))) as $span | {value: $member.value, brief: $member.brief, spec: ($span.annotations.openinference.spec // "")}]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Value | Represents | Conventions |
| --- | --- | --- |
| [`LLM`](#llm) | A call to a large language model. | [LLM Spans](../spans/llm.md) |
| [`CHAIN`](#chain) | A link between application steps, or the start of a request. |  |
| [`TOOL`](#tool) | Execution of a tool or function. | [Tool Calling](../content/tool-calling.md) |
| [`RETRIEVER`](#retriever) | A query to a vector store, search engine, or database. |  |
| [`RERANKER`](#reranker) | Reordering of candidate documents by relevance. |  |
| [`EMBEDDING`](#embedding) | Generation of vector embeddings. | [Embedding Spans](../spans/embedding.md) |
| [`AGENT`](#agent) | A reasoning step that acts on tools using an LLM's guidance. |  |
| [`GUARDRAIL`](#guardrail) | An input or output moderation check. |  |
| [`EVALUATOR`](#evaluator) | An evaluation of a model's output. | [Annotations and Evaluations](../feedback/annotations.md) |
| [`PROMPT`](#prompt) | The rendering of a prompt template. |  |
| [`DECISION`](#decision) | A call to a decision model. | [Decision Spans](../spans/decision.md) |

<!-- endweaver -->

Each section below lists the attributes that characterize spans of that kind. Any span may also carry the
[general attributes](../attributes/semantic-conventions.md#input-and-output) `input.value` and `output.value` and the
[context attributes](../attributes/context.md).

## LLM

A call to a Large Language Model, for example a call to OpenAI or Llama for chat completions or text generation.

<!-- weaver . as $root | [$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | [.key, .requirement_level]] as $common | {characteristic: ([$root.registry.spans[] | select(.type == "openinference.llm") | .attributes[] | select([.key, .requirement_level] as $pair | $common | index([$pair]) | not) | .key | split(".") | if length > 2 then (.[0:2] | join(".")) + ".*" else join(".") end] | unique)} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

Characteristic attributes: `llm.choices`, `llm.cost.*`, `llm.finish_reason`, `llm.input_messages`, `llm.invocation_parameters`, `llm.model_name`, `llm.output_messages`, `llm.prompts`, `llm.provider`, `llm.request.*`, `llm.response.*`, `llm.system`, `llm.token_count.*`, `llm.tools`.

<!-- endweaver -->

See [LLM Spans](../spans/llm.md).

## EMBEDDING

A call to an LLM or embedding service to generate embeddings, for example a call to OpenAI for an ada embedding used in
retrieval.

<!-- weaver . as $root | [$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | [.key, .requirement_level]] as $common | {characteristic: ([$root.registry.spans[] | select(.type == "openinference.embedding") | .attributes[] | select([.key, .requirement_level] as $pair | $common | index([$pair]) | not) | .key | split(".") | if length > 2 then (.[0:2] | join(".")) + ".*" else join(".") end] | unique)} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

Characteristic attributes: `embedding.embeddings`, `embedding.invocation_parameters`, `embedding.model_name`, `llm.token_count.*`.

<!-- endweaver -->

See [Embedding Spans](../spans/embedding.md).

## CHAIN

A starting point or a link between different LLM application steps, for example the beginning of a request to an LLM
application, or the glue code that passes context from a retriever to an LLM call.

<!-- weaver . as $root | [$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | [.key, .requirement_level]] as $common | {characteristic: ([$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | select([.key, .requirement_level] as $pair | $common | index([$pair]) | not) | .key | split(".") | if length > 2 then (.[0:2] | join(".")) + ".*" else join(".") end] | unique)} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

Characteristic attributes: `input.value`, `output.value`.

<!-- endweaver -->

## RETRIEVER

A data retrieval step, for example a call to a vector store or a database to fetch documents or information.

<!-- weaver . as $root | [$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | [.key, .requirement_level]] as $common | {characteristic: ([$root.registry.spans[] | select(.type == "openinference.retriever") | .attributes[] | select([.key, .requirement_level] as $pair | $common | index([$pair]) | not) | .key | split(".") | if length > 2 then (.[0:2] | join(".")) + ".*" else join(".") end] | unique)} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

Characteristic attributes: `retrieval.documents`.

<!-- endweaver -->

`input.value` usually holds the query.

## RERANKER

The reranking of a set of input documents. For example, a cross-encoder computes the input documents' relevance scores
with respect to a user query, and the top K documents with the highest scores are returned.

<!-- weaver . as $root | [$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | [.key, .requirement_level]] as $common | {characteristic: ([$root.registry.spans[] | select(.type == "openinference.reranker") | .attributes[] | select([.key, .requirement_level] as $pair | $common | index([$pair]) | not) | .key | split(".") | if length > 2 then (.[0:2] | join(".")) + ".*" else join(".") end] | unique)} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

Characteristic attributes: `reranker.input_documents`, `reranker.model_name`, `reranker.output_documents`, `reranker.query`, `reranker.top_k`.

<!-- endweaver -->

## TOOL

A call to an external tool such as a calculator or a weather API, or any function execution invoked by an LLM or agent.

<!-- weaver . as $root | [$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | [.key, .requirement_level]] as $common | {characteristic: ([$root.registry.spans[] | select(.type == "openinference.tool") | .attributes[] | select([.key, .requirement_level] as $pair | $common | index([$pair]) | not) | .key | split(".") | if length > 2 then (.[0:2] | join(".")) + ".*" else join(".") end] | unique)} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

Characteristic attributes: `tool.description`, `tool.id`, `tool.name`, `tool.parameters`.

<!-- endweaver -->

See [Tool Calling](../content/tool-calling.md) for how the LLM side of a tool call is recorded.

## AGENT

A span that encompasses calls to LLMs and tools. An agent describes a reasoning block that acts on tools using the
guidance of an LLM.

<!-- weaver . as $root | [$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | [.key, .requirement_level]] as $common | {characteristic: ([$root.registry.spans[] | select(.type == "openinference.agent") | .attributes[] | select([.key, .requirement_level] as $pair | $common | index([$pair]) | not) | .key | split(".") | if length > 2 then (.[0:2] | join(".")) + ".*" else join(".") end] | unique)} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

Characteristic attributes: `agent.name`, `graph.node.*`.

<!-- endweaver -->

The `graph.node.*` attributes apply when the agent is part of an execution graph.

## GUARDRAIL

A call to a component that protects against jailbreak prompts or undesirable output, by modifying or rejecting an LLM's
input or response. For example, a guardrail span could check whether an LLM's response contains inappropriate language,
through a custom or external guardrail library, and amend the response to remove it.

<!-- weaver . as $root | [$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | [.key, .requirement_level]] as $common | {characteristic: ([$root.registry.spans[] | select(.type == "openinference.guardrail") | .attributes[] | select([.key, .requirement_level] as $pair | $common | index([$pair]) | not) | .key | split(".") | if length > 2 then (.[0:2] | join(".")) + ".*" else join(".") end] | unique)} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

Characteristic attributes: `input.value`, `output.value`.

<!-- endweaver -->

## EVALUATOR

A call to a function or process that evaluates a language model's outputs, for example assessing the relevance,
correctness, or helpfulness of its answers.

<!-- weaver . as $root | [$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | [.key, .requirement_level]] as $common | {characteristic: ([$root.registry.spans[] | select(.type == "openinference.evaluator") | .attributes[] | select([.key, .requirement_level] as $pair | $common | index([$pair]) | not) | .key | split(".") | if length > 2 then (.[0:2] | join(".")) + ".*" else join(".") end] | unique)} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

Characteristic attributes: `annotations`, `evaluations`.

<!-- endweaver -->

An `EVALUATOR` span is the recommended carrier for
[post-hoc feedback](../feedback/annotations.md#post-hoc-feedback).

## PROMPT

The rendering of a prompt template, for example substituting variables into a template to produce the final prompt sent
to an LLM.

<!-- weaver . as $root | [$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | [.key, .requirement_level]] as $common | {characteristic: ([$root.registry.spans[] | select(.type == "openinference.prompt") | .attributes[] | select([.key, .requirement_level] as $pair | $common | index([$pair]) | not) | .key | split(".") | if length > 2 then (.[0:2] | join(".")) + ".*" else join(".") end] | unique)} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

Characteristic attributes: `llm.prompt_template.*`, `prompt.id`, `prompt.url`, `prompt.vendor`.

<!-- endweaver -->

The `prompt.*` attributes apply when the prompt comes from a prompt management system.

## DECISION

A call to a decision model, which scores or selects among candidate options provided in the request rather than
generating free-form text. For example, a decision span could represent choosing a route, judging a condition, or
scoring an item against a rubric. Decision models include
[TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

<!-- weaver . as $root | [$root.registry.spans[] | select(.type == "openinference.chain") | .attributes[] | [.key, .requirement_level]] as $common | {characteristic: ([$root.registry.spans[] | select(.type == "openinference.decision") | .attributes[] | select([.key, .requirement_level] as $pair | $common | index([$pair]) | not) | .key | split(".") | if length > 2 then (.[0:2] | join(".")) + ".*" else join(".") end] | unique)} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

Characteristic attributes: `decision.model_name`, `decision.provider`, `decision.request.*`, `decision.response.*`, `decision.system`, `decision.token_count.*`.

<!-- endweaver -->

See [Decision Spans](../spans/decision.md).
