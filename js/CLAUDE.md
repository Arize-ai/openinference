# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is the JavaScript/TypeScript workspace of OpenInference, providing OpenTelemetry-based instrumentation for AI/ML applications. The workspace contains core utilities and framework-specific instrumentors for popular AI libraries (OpenAI, LangChain, Bedrock, etc.).

## Essential Commands

### Package Management and Setup

```bash
# Install dependencies (MUST use pnpm, not npm)
pnpm install --frozen-lockfile -r

# Build all packages (includes prebuild for cross-package dependencies)
pnpm run -r build

# Run prebuild only (generates version files and symlinks)
pnpm run -r prebuild
```

### Testing and Quality

```bash
# Run tests for all packages
pnpm run -r test

# Type checking across all packages
pnpm run type:check

# Linting
pnpm run lint

# Code formatting
pnpm run prettier:check
pnpm run prettier:write
```

### Release Management

```bash
# Create changeset for version bumping
pnpm changeset

# Version bump and publish (CI handles this)
pnpm ci:version
pnpm ci:publish
```

## Architecture Overview

### Core Package Structure

- **`openinference-core`**: Base utilities including `OITracer`, `OISpan`, trace configuration, and context attribute management
- **`openinference-semantic-conventions`**: Standard attribute definitions for AI observability
- **Framework Instrumentors**: Individual packages for each AI framework (OpenAI, LangChain, Bedrock, etc.)
- **Specialized Packages**: Vercel AI SDK integration, Mastra framework support

### Key Patterns

#### Instrumentor Implementation

All instrumentors must extend `InstrumentationBase` and use `OITracer` from core:

```typescript
export class MyInstrumentation extends InstrumentationBase {
  private oiTracer: OITracer;

  constructor({ traceConfig }: { traceConfig?: TraceConfigOptions } = {}) {
    super(/* instrumentation config */);
    this.oiTracer = new OITracer({ tracer: this.tracer, traceConfig });
  }
}
```

#### Required Features

1. **Suppress Tracing**: Must respect `isTracingSuppressed()` from `@opentelemetry/core`
2. **Context Attribute Propagation**: Use `OITracer` to automatically handle context attributes (session ID, user ID, etc.)
3. **Trace Configuration**: Support masking sensitive data via `TraceConfig`

### Testing Considerations

- Uses Vite with manual module mocking due to instrumentation timing requirements
- Pattern: `instrumentation._modules[0].moduleExports = module` for manual mocks
- Must test suppress tracing, context propagation, and trace configuration features

### Workspace Configuration

- **pnpm workspaces**: All packages defined in `pnpm-workspace.yaml`
- **Cross-package dependencies**: Handled via prebuild script with symlinks
- **Changesets**: Version management system for coordinated releases
- **Publishing**: Automated via GitHub Actions to `@arizeai` npm organization

## Development Workflow

1. **Prerequisites**: Node.js v20, pnpm (global version >=9.7.0)
2. **Setup**: `pnpm install --frozen-lockfile -r`
3. **Build**: Always run `pnpm run -r build` after changes (includes prebuild)
4. **Testing**: Comprehensive tests including instrumentation-specific features
5. **Versioning**: Create changesets with `pnpm changeset` before merging PRs

## Important Notes

- Always use `OITracer` instead of raw OpenTelemetry tracer for consistent behavior
- Cross-package dependencies require prebuild step to generate symlinks
- Manual module mocking in tests is required due to instrumentation timing
- Each instrumentor must implement the minimum feature set (suppress tracing, context attributes, trace config)
- Packages publish to `@arizeai` organization on npm
