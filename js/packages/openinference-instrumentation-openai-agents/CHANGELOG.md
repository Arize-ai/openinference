# @arizeai/openinference-instrumentation-openai-agents

## 0.2.4

### Patch Changes

- Updated dependencies [145e3c6]
  - @arizeai/openinference-semantic-conventions@2.6.0
  - @arizeai/openinference-core@2.4.1

## 0.2.3

### Patch Changes

- 622d20f: Bump @opentelemetry/core to ^2.8.0 for OpenAI Agents instrumentation to address the W3C Baggage denial-of-service security advisory.

## 0.2.2

### Patch Changes

- Updated dependencies [d0f5a88]
  - @arizeai/openinference-core@2.4.0

## 0.2.1

### Patch Changes

- Updated dependencies [1fe7927]
  - @arizeai/openinference-core@2.3.0

## 0.2.0

### Minor Changes

- ad1bbf4: Initial release of OpenInference instrumentation for the OpenAI Agents SDK (`@openai/agents`).

  - Bridges the SDK's native `TracingProcessor` interface to OpenTelemetry without monkey-patching.
  - Emits OpenInference-compliant spans for every SDK span type: agent, generation, response, function (tool), handoff, mcp_tools, guardrail, and custom.
  - Captures full LLM telemetry — model name, invocation parameters, input/output messages, tool calls, and token counts (including `cache_read` and `reasoning` details) — for both the `chat_completions` and `responses` transports.
  - Records multi-agent handoff relationships via `graph.node.id` / `graph.node.parent_id` so flows can be visualised as a graph.
  - Supports OpenInference `TraceConfig` for masking sensitive inputs/outputs.
  - Provides exclusive (default) and additive registration via `instrument({ exclusiveProcessor })`, mirroring the Python instrumentor's `exclusive_processor` argument, plus an `uninstrument` teardown.
