# OpenInference Instrumentation for OpenAI Agents SDK (Node.js)

OpenTelemetry-based instrumentation for the [OpenAI Agents SDK](https://www.npmjs.com/package/@openai/agents) (`@openai/agents`). Bridges the SDK's native tracing events to OpenTelemetry spans following the [OpenInference semantic conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md), so agent runs can be observed in any OpenTelemetry-compatible backend such as [Arize Phoenix](https://github.com/Arize-ai/phoenix), Arize AX, Jaeger, or your collector of choice.

## Installation

```bash
npm install @arizeai/openinference-instrumentation-openai-agents @arizeai/openinference-semantic-conventions @openai/agents
```

## Quickstart

```typescript
import * as agents from "@openai/agents";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources";
import { BatchSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

import { OpenAIAgentsInstrumentation } from "@arizeai/openinference-instrumentation-openai-agents";

// 1. Configure OpenTelemetry.
const provider = new NodeTracerProvider({
  resource: new Resource({
    [ATTR_SERVICE_NAME]: "my-agent-app",
  }),
  spanProcessors: [
    new BatchSpanProcessor(new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" })),
  ],
});
provider.register();

// 2. Register the OpenInference processor with the agents SDK.
const instrumentation = new OpenAIAgentsInstrumentation({ tracerProvider: provider });
instrumentation.manuallyInstrument(agents);

// 3. Use the agents SDK as usual.
const agent = new agents.Agent({
  name: "Assistant",
  instructions: "You are a helpful assistant.",
});

const result = await agents.run(agent, "What is the capital of France?");
console.log(result.finalOutput);
```

## How it works

Unlike most OpenInference instrumentations, this package does **not** monkey-patch the SDK. The agents SDK exposes a first-class `TracingProcessor` interface; this package implements it and registers via the SDK's `setTraceProcessors` / `addTraceProcessor` APIs.

| Mode                | Call                                                                                                                        | Behaviour                                                                                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Exclusive (default) | `instrument()` (CommonJS) or `manuallyInstrument(agents)` (ESM)                                                             | Replaces every existing trace processor with the OpenInference one. Use when OpenTelemetry is your sole tracing destination.                                                      |
| Additive            | `instrument({ exclusiveProcessor: false })` (CommonJS) or `manuallyInstrument(agents, { exclusiveProcessor: false })` (ESM) | Adds the OpenInference processor alongside any existing processors (e.g. the SDK's default OpenAI tracing exporter). Use when you want OpenInference _and_ OpenAI native tracing. |

When the module is patched automatically (via `registerInstrumentations` or NodeSDK), choose the mode through the constructor instead: `new OpenAIAgentsInstrumentation({ exclusiveProcessor: false })`.

To stop tracing, call `instrumentation.uninstrument()`. In additive mode, the agents SDK does not expose a single-processor removal API, so `uninstrument()` disables the OpenInference processor in place without clearing other SDK processors.

For ESM-only environments, pass the imported SDK namespace explicitly:

```typescript
import * as agents from "@openai/agents";

const instrumentation = new OpenAIAgentsInstrumentation({ tracerProvider: provider });
instrumentation.manuallyInstrument(agents, { exclusiveProcessor: false });
```

## Configuration

```typescript
new OpenAIAgentsInstrumentation({
  // Optional: an OTel TracerProvider. Defaults to the global provider.
  tracerProvider,

  // Optional: processor registration mode (see "How it works" above).
  // This is the only way to choose additive mode when the module is patched
  // automatically (registerInstrumentations / NodeSDK). Defaults to true.
  exclusiveProcessor: true,

  // Optional: OpenInference trace configuration for masking/redacting
  // sensitive data on emitted spans.
  // See https://github.com/Arize-ai/openinference/blob/main/js/packages/openinference-core
  traceConfig: {
    hideInputs: true,
    hideOutputs: true,
  },

  // Optional: maximum number of root trace spans kept in memory while traces
  // are in flight. If the limit is exceeded, the oldest root span is ended
  // and evicted. Defaults to 1000.
  maxRootSpansInFlight: 1000,
});
```

## Span coverage

Each agents SDK span type is mapped to an OpenInference span kind:

| SDK span     | `openinference.span.kind` | Captured attributes                                                                                                                                                                                                                      |
| ------------ | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `agent`      | `AGENT`                   | `graph.node.id`, `graph.node.parent_id` (on handoff destination)                                                                                                                                                                         |
| `generation` | `LLM`                     | `llm.model_name`, `llm.invocation_parameters`, `llm.input_messages.*`, `llm.output_messages.*`, `llm.token_count.{prompt,completion,total}`, `llm.token_count.prompt_details.cache_read`, `llm.token_count.completion_details.reasoning` |
| `response`   | `LLM`                     | All of the above plus `llm.tools.*`, system instructions as input message 0                                                                                                                                                              |
| `function`   | `TOOL`                    | `tool.name`, `input.value`, `output.value`                                                                                                                                                                                               |
| `handoff`    | `TOOL`                    | Span name `handoff to <to_agent>`; the destination agent receives `graph.node.parent_id` linking back to the source                                                                                                                      |
| `mcp_tools`  | `TOOL`                    | `output.value` (JSON list of tool names)                                                                                                                                                                                                 |
| `guardrail`  | `GUARDRAIL`               | `tool.name`, `guardrail.triggered`                                                                                                                                                                                                       |
| `custom`     | `CHAIN`                   | `output.value` (JSON-serialised user data)                                                                                                                                                                                               |

Both the **chat_completions** and **responses** transports are supported. In chat_completions mode the SDK stores raw response objects in `output[]`; this instrumentation extracts messages from `choices[].message` and accumulates token usage across all responses.

## Examples

```bash
cd js/packages/openinference-instrumentation-openai-agents
pnpm install
pnpm -r build

OPENAI_API_KEY=sk-... npx tsx examples/chat.ts     # single agent + tool call
OPENAI_API_KEY=sk-... npx tsx examples/handoff.ts  # multi-agent handoff
OPENAI_API_KEY=sk-... npx tsx examples/streaming.ts # streamed agent run
OPENAI_API_KEY=sk-... npx tsx examples/guardrail.ts # input guardrail
```

The shared OTel setup lives in `examples/instrumentation.ts` — modify it to swap in an OTLP exporter or any other span processor.

## License

Apache-2.0
