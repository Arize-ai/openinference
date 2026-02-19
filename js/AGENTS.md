# JavaScript/TypeScript Workspace Guide

TypeScript instrumentation packages for OpenInference. Uses pnpm workspaces, TypeScript strict mode, Vitest for testing, and publishes to the `@arizeai/` npm organization. Requires Node.js v20.

---

## Setup and Build

```bash
cd js
nvm use                               # uses .nvmrc for Node v20
pnpm install --frozen-lockfile -r    # MUST be pnpm, not npm or yarn
pnpm run -r prebuild                 # REQUIRED: generates version files + cross-package symlinks
pnpm run -r build
```

> **Why prebuild matters**: The prebuild script generates instrumentation version files and creates symlinks for cross-package dependencies. Cross-package imports will fail at test time if prebuild has not been run. Re-run it whenever you change packages in the repo.

Global pnpm version must be `>=9.7.0` (managed via `package.json`).

---

## Testing and Quality

```bash
pnpm run -r test
pnpm run type:check
pnpm run lint
pnpm run prettier:check
pnpm run prettier:write
```

---

## Package Inventory

| Package | Description |
|---------|-------------|
| `openinference-core` | Base utilities: `OITracer`, `OISpan`, `TraceConfig`, context attributes |
| `openinference-semantic-conventions` | Standard attribute constants for AI observability |
| `openinference-instrumentation-openai` | OpenAI SDK instrumentation |
| `openinference-instrumentation-anthropic` | Anthropic SDK instrumentation |
| `openinference-instrumentation-bedrock` | AWS Bedrock instrumentation |
| `openinference-instrumentation-bedrock-agent-runtime` | AWS Bedrock Agent Runtime instrumentation |
| `openinference-instrumentation-langchain` | LangChain.js instrumentation |
| `openinference-instrumentation-langchain-v0` | LangChain.js v0 instrumentation |
| `openinference-instrumentation-mcp` | MCP (Model Context Protocol) instrumentation |
| `openinference-instrumentation-beeai` | BeeAI framework instrumentation |
| `openinference-vercel` | Vercel AI SDK integration |
| `openinference-mastra` | Mastra framework support |
| `openinference-genai` | GenAI utilities |

---

## The Three Required Features

Every instrumentor must implement these three features.

### OITracer setup (handles features 2 and 3 automatically)

Use `OITracer` from core as a private property on your instrumentor class. Use `this.oiTracer` — never `this.tracer` — to create spans.

```typescript
import { OITracer, TraceConfigOptions } from "@arizeai/openinference-core";
import { InstrumentationBase, InstrumentationConfig } from "@opentelemetry/instrumentation";

export class MyInstrumentation extends InstrumentationBase<typeof moduleToInstrument> {
  private oiTracer: OITracer;

  constructor({
    instrumentationConfig,
    traceConfig,
  }: {
    instrumentationConfig?: InstrumentationConfig;
    traceConfig?: TraceConfigOptions;
  } = {}) {
    super(
      "@arizeai/openinference-instrumentation-<name>",
      VERSION,
      Object.assign({}, instrumentationConfig),
    );
    this.oiTracer = new OITracer({ tracer: this.tracer, traceConfig });
  }

  // Always use this.oiTracer, not this.tracer:
  protected patch(): void {
    const span = this.oiTracer.startSpan("my-span");
    // ...
    span.end();
  }
}
```

### Feature 1 — Suppress Tracing

Check the OTel context suppression flag before creating any span.

```typescript
import { context } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";

function patchedFunction(original: (...args: unknown[]) => unknown) {
  return function (...args: unknown[]) {
    if (isTracingSuppressed(context.active())) {
      return original.apply(this, args);  // skip tracing
    }
    // ... tracing logic ...
  };
}
```

Also implement the `unpatch()` method to restore original functions when instrumentation is disabled.

### Feature 2 — Context Attribute Propagation

Handled automatically by `OITracer`. See `packages/openinference-core/src/trace/contextAttributes.ts` for the underlying implementation.

### Feature 3 — Trace Configuration

Handled automatically by `OITracer` and `OISpan`. Pass `traceConfig` to `OITracer` at construction time (shown in the OITracer setup snippet above).

---

## The Module Mocking Pattern (critical, non-obvious)

Vitest's auto-mocking feature breaks instrumentation timing because instrumentation must run before the library is imported. Use manual module assignment instead:

```typescript
import { describe, it, expect, beforeAll } from "vitest";
import * as myModule from "my-module";

describe("MyInstrumentation", () => {
  const instrumentation = new MyInstrumentation();

  beforeAll(() => {
    // Manually assign the module exports to bypass auto-mock timing issues:
    instrumentation._modules[0].moduleExports = myModule;
  });

  it("creates a span", async () => {
    // test body
  });
});
```

---

## Creating a New JS Instrumentor

1. **Scaffold a new package** by copying an existing instrumentor:
   ```bash
   cp -r js/packages/openinference-instrumentation-openai \
         js/packages/openinference-instrumentation-<name>
   ```

2. **Required files**:
   - `package.json` — package name `@arizeai/openinference-instrumentation-<name>`, scripts
   - `src/instrumentation.ts` — instrumentor class extending `InstrumentationBase`
   - `src/index.ts` — public exports
   - `src/version.ts` — `VERSION` constant (generated by prebuild)
   - `tsconfig.json` — extending `../../tsconfig.base.json`
   - `vitest.config.ts` — test configuration

3. **Register the package**:
   - Add to `js/pnpm-workspace.yaml` under `packages:`
   - Run `pnpm install` to link it

4. **Create a changeset** (required before PR merge):
   ```bash
   pnpm changeset
   # Select the new package, choose semver bump type, add summary
   ```

---

## Semantic Conventions

```typescript
import {
  SemanticConventions,
  OpenInferenceSpanKind,
} from "@arizeai/openinference-semantic-conventions";

// Required on every span:
span.setAttribute(
  SemanticConventions.OPENINFERENCE_SPAN_KIND,
  OpenInferenceSpanKind.LLM,
);

// Common LLM attributes:
span.setAttribute(SemanticConventions.LLM_MODEL_NAME, "gpt-4o");
span.setAttribute(SemanticConventions.INPUT_VALUE, promptText);
span.setAttribute(SemanticConventions.OUTPUT_VALUE, responseText);
span.setAttribute(SemanticConventions.LLM_TOKEN_COUNT_PROMPT, promptTokens);
span.setAttribute(SemanticConventions.LLM_TOKEN_COUNT_COMPLETION, completionTokens);

// Flattened message arrays (zero-based index):
messages.forEach((msg, i) => {
  span.setAttribute(`llm.input_messages.${i}.message.role`, msg.role);
  span.setAttribute(`llm.input_messages.${i}.message.content`, msg.content);
});
```

---

## Version Management (Changesets)

A changeset must be created for **every PR that touches `js/`** before it can be merged.

```bash
pnpm changeset
# Interactive prompt:
# 1. Select which packages are changed
# 2. Choose semver bump type (patch / minor / major) per package
# 3. Write a summary of the changes
# Commit the generated .changeset/<id>.md file with your PR
```

On PR merge, GitHub Actions creates a release PR. When the release PR is merged, new versions are published to npm under the `@arizeai` organization.

---

## Publishing

Automated via GitHub Actions + changesets workflow. For manual publishing:

```bash
pnpm changeset          # create changeset
pnpm changeset version  # bump package versions
pnpm -r build           # build all packages
pnpm -r publish         # publish to npm (requires @arizeai org membership)
```
