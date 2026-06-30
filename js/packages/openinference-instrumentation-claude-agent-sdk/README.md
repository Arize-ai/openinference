# OpenInference Instrumentation for Claude Agent SDK

OpenTelemetry-based instrumentation for the [Claude Agent SDK](https://www.npmjs.com/package/@anthropic-ai/claude-agent-sdk) (`@anthropic-ai/claude-agent-sdk`). Produces **AGENT** and **TOOL** spans following [OpenInference semantic conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md).

## Installation

```bash
npm install @arizeai/openinference-instrumentation-claude-agent-sdk
```

## Usage

### Auto-instrumentation (CommonJS)

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { ClaudeAgentSDKInstrumentation } from "@arizeai/openinference-instrumentation-claude-agent-sdk";

const provider = new NodeTracerProvider();
provider.register();

const instrumentation = new ClaudeAgentSDKInstrumentation();
instrumentation.setTracerProvider(provider);
```

### Manual instrumentation (ESM)

```typescript
import * as ClaudeAgentSDK from "@anthropic-ai/claude-agent-sdk";
import { ClaudeAgentSDKInstrumentation } from "@arizeai/openinference-instrumentation-claude-agent-sdk";

const instrumentation = new ClaudeAgentSDKInstrumentation({
  tracerProvider: provider,
});
instrumentation.manuallyInstrument(ClaudeAgentSDK);
```

### With trace configuration (mask sensitive data)

```typescript
const instrumentation = new ClaudeAgentSDKInstrumentation({
  tracerProvider: provider,
  traceConfig: {
    hideInputs: true,
    hideOutputs: true,
  },
});
```

## Instrumented APIs

### V1: `query()`

The `query()` function is wrapped to produce:

- An **AGENT** span for the entire query lifecycle
- **TOOL** child spans for each tool call (via hook injection)

### V2 (unstable): `unstable_v2_prompt()`, `unstable_v2_createSession()`, `unstable_v2_resumeSession()`

- `unstable_v2_prompt()` produces an **AGENT** span
- Session methods produce per-turn **AGENT** spans with **TOOL** child spans

## Span Attributes

### AGENT spans

| Attribute                    | Description            |
| ---------------------------- | ---------------------- |
| `openinference.span.kind`    | `"AGENT"`              |
| `input.value`                | Prompt text            |
| `output.value`               | Result text            |
| `session.id`                 | SDK session identifier |
| `llm.model_name`             | Model used             |
| `llm.token_count.prompt`     | Input token count      |
| `llm.token_count.completion` | Output token count     |
| `llm.token_count.total`      | Total token count      |
| `llm.cost.total`             | Total cost in USD      |

### TOOL spans

| Attribute                 | Description          |
| ------------------------- | -------------------- |
| `openinference.span.kind` | `"TOOL"`             |
| `tool.name`               | Tool name            |
| `tool.parameters`         | Tool input (JSON)    |
| `input.value`             | Tool input (JSON)    |
| `output.value`            | Tool response (JSON) |
