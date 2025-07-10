# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Additional Memory Files

This project maintains additional memory files that should be referenced for complete context:

- **SESSION_SUMMARY.md**: Detailed session history, current project state, and TDD progress for AWS Bedrock instrumentation
- **BEST_PRACTICES.md**: Code review standards, development methodology preferences, and critical architectural decisions

These files contain essential context about ongoing work, established patterns, and user preferences that inform development decisions.

## Code Quality Tools

- **ESLint**: Workspace-level linting (`pnpm run lint` from `/js/`)
- **Prettier**: Code formatting (`pnpm run prettier:write` from `/js/`)
- **TypeScript**: Type checking (`pnpm run type:check` from `/js/`)

## Architecture Overview

OpenInference is a comprehensive observability platform for AI applications, providing instrumentation libraries and tools for monitoring LLM interactions, embeddings, and other AI operations.

### Core Components

- **Instrumentation Libraries**: Auto-instrumentation for popular AI frameworks (OpenAI, LangChain, LlamaIndex, Anthropic, AWS Bedrock, etc.)
- **Semantic Conventions**: Standardized attributes and span structures for AI observability
- **Core Utilities**: Shared tracing functionality and configuration management
- **Phoenix Integration**: Native integration with Phoenix for trace collection and analysis

### Project Structure

```
openinference/
├── js/                           # JavaScript/TypeScript implementations
│   ├── packages/
│   │   ├── openinference-core/                    # Core utilities and tracing
│   │   ├── openinference-semantic-conventions/    # Semantic conventions
│   │   ├── openinference-instrumentation-openai/ # OpenAI instrumentation
│   │   ├── openinference-instrumentation-langchain/ # LangChain instrumentation
│   │   ├── openinference-instrumentation-bedrock/   # AWS Bedrock instrumentation
│   │   └── ...
│   ├── package.json              # Root package.json with workspace scripts
│   └── tsconfig.json             # TypeScript configuration
├── python/                       # Python implementations
│   ├── openinference-instrumentation-openai/
│   ├── openinference-instrumentation-langchain/
│   └── ...
└── README.md
```

### JavaScript Package Directory Reference

For efficient navigation and search, here's a comprehensive guide to all JS packages:

#### Core Infrastructure Packages
- **`js/packages/openinference-core/`** (`@arizeai/openinference-core`)
  - Shared utilities, tracing configuration, span management
  - Key files: `src/trace/`, `src/utils/`
  - Used by: All instrumentation packages

- **`js/packages/openinference-semantic-conventions/`** (`@arizeai/openinference-semantic-conventions`)
  - Standardized attribute definitions for AI observability
  - Key files: `src/trace/SemanticConventions.ts`, `src/resource/`
  - Used by: All instrumentation and utility packages

#### AI Framework Instrumentation Packages
- **`js/packages/openinference-instrumentation-openai/`** (`@arizeai/openinference-instrumentation-openai`)
  - Auto-instrumentation for OpenAI SDK
  - Key files: `src/instrumentation.ts`, `src/responsesAttributes.ts`
  - Peer dependency: `openai`

- **`js/packages/openinference-instrumentation-langchain/`** (`@arizeai/openinference-instrumentation-langchain`)
  - Auto-instrumentation for LangChain.js framework
  - Key files: `src/instrumentation.ts`, `src/instrumentationUtils.ts`
  - Peer dependency: `@langchain/core`
  - Multi-version support: v0.2, v0.3

- **`js/packages/openinference-instrumentation-bedrock/`** (`@arizeai/openinference-instrumentation-bedrock`)
  - Auto-instrumentation for AWS Bedrock Runtime
  - Key files: `src/instrumentation.ts`, `src/attributes/`
  - Peer dependency: `@aws-sdk/client-bedrock-runtime`
  - Features: VCR testing, tool calling support

- **`js/packages/openinference-instrumentation-beeai/`** (`@arizeai/openinference-instrumentation-beeai`)
  - Auto-instrumentation for BeeAI framework
  - Key files: `src/instrumentation.ts`, `src/middleware.ts`, `src/helpers/`
  - Peer dependency: `beeai-framework`

- **`js/packages/openinference-instrumentation-mcp/`** (`@arizeai/openinference-instrumentation-mcp`)
  - Auto-instrumentation for Model Context Protocol (MCP)
  - Key files: `src/mcp.ts`
  - Dev dependency: `@modelcontextprotocol/sdk`

#### Platform Integration Packages
- **`js/packages/openinference-vercel/`** (`@arizeai/openinference-vercel`)
  - Utilities for Vercel AI SDK span processing
  - Key files: `src/OpenInferenceSpanProcessor.ts`, `src/utils.ts`
  - Exports: Main, utils, types
  - Platform: Vercel AI SDK, Next.js

- **`js/packages/openinference-mastra/`** (`@arizeai/openinference-mastra`)
  - Utilities for Mastra agent framework span ingestion
  - Key files: `src/OpenInferenceTraceExporter.ts`, `src/attributes.ts`
  - Build: ESM-only, uses Vitest
  - Platform: Mastra agent framework

#### Package Navigation Quick Reference

**Search by Technology:**
- OpenAI → `openinference-instrumentation-openai/`
- LangChain → `openinference-instrumentation-langchain/`
- AWS Bedrock → `openinference-instrumentation-bedrock/`
- BeeAI → `openinference-instrumentation-beeai/`
- MCP → `openinference-instrumentation-mcp/`
- Vercel AI → `openinference-vercel/`
- Mastra → `openinference-mastra/`

**Search by Function:**
- Core utilities → `openinference-core/`
- Semantic conventions → `openinference-semantic-conventions/`
- Auto-instrumentation → `openinference-instrumentation-*/`
- Platform integration → `openinference-vercel/`, `openinference-mastra/`

**Search by File Type:**
- Instrumentation logic → `src/instrumentation.ts`
- Attribute extraction → `src/attributes/`, `src/responsesAttributes.ts`
- Test files → `test/`, `src/**/*.test.ts`
- Examples → `examples/`
- VCR recordings → `test/recordings/` (Bedrock only)

### Development Patterns

- **Monorepo Architecture**: Uses pnpm workspaces for JavaScript packages
- **Consistent API**: All instrumentation libraries follow similar patterns
- **OpenTelemetry Integration**: Built on OpenTelemetry for tracing infrastructure
- **Semantic Conventions**: Standardized attribute naming across all instrumentations
- **Test-Driven Development**: Comprehensive test coverage with VCR-style testing

#### Common File Patterns Across Packages

**Standard Package Structure:**
```
src/
├── index.ts              # Main entry point, exports instrumentation
├── instrumentation.ts    # Core instrumentation logic (most packages)
├── version.ts           # Auto-generated version info
├── types.ts             # TypeScript type definitions
└── utils.ts             # Shared utilities

test/
├── *.test.ts            # Jest test files
├── fixtures.ts          # Test data and mocks (some packages)
└── recordings/          # VCR test recordings (Bedrock only)
```

**Package-Specific Patterns:**
- **Bedrock**: `src/attributes/` (request/response attribute extraction), `scripts/` (VCR management)
- **BeeAI**: `src/helpers/` (trace building utilities), `src/middleware.ts`
- **LangChain**: `src/instrumentationUtils.ts`, `src/tracer.ts` (custom tracer logic)
- **OpenAI**: `src/responsesAttributes.ts` (response attribute extraction)
- **Vercel**: Multi-export structure with utils and types
- **Mastra**: ESM-only build, Vitest instead of Jest

#### Testing & Build Patterns

- **Jest**: Standard testing framework (most packages)
- **Vitest**: Modern testing (Mastra only)
- **VCR Testing**: HTTP recording/replay (Bedrock only)
- **TypeScript**: Triple build targets (CommonJS, ESM, ESNext)
- **Dual Exports**: CommonJS + ESM support

## Development Commands

```bash
# From /js/ directory (workspace level)
pnpm install                 # Install dependencies
pnpm test                    # Run all tests
pnpm run lint                # Lint entire workspace
pnpm run type:check          # Type check all packages
pnpm run prettier:write      # Fix formatting

# From individual package directory
npm test                     # Run package tests
npm run type:check           # Type check single package
```

## Key Technologies

- **OpenTelemetry**: Core tracing infrastructure
- **TypeScript**: Primary language for JavaScript packages
- **pnpm**: Package manager and workspace tool
- **Jest/Vitest**: Testing frameworks
- **Changesets**: Version and release management

