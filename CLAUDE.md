# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Additional Memory Files

This project maintains additional memory files that should be referenced for complete context:

- **SESSION_SUMMARY.md**: Detailed session history, current project state, and TDD progress for AWS Bedrock instrumentation
- **BEST_PRACTICES.md**: Code review standards, development methodology preferences, and critical architectural decisions

These files contain essential context about ongoing work, established patterns, and user preferences that inform development decisions.

## Development Insights on Code Quality Tools

### JavaScript/TypeScript Quality Tools
- **Purpose**: Ensure consistent code style, catch potential errors early, and improve overall code maintainability
- **ESLint**: Workspace-level linting with TypeScript support for static code analysis
  - Configuration at `/js/` workspace root applies to all packages
  - Helps identify potential bugs, style inconsistencies, and anti-patterns
  - Run with `pnpm run lint` from `/js/` directory
- **Prettier**: Workspace-level code formatting for consistency across all packages
  - Automatically formats code to a standard style across the entire monorepo
  - Reduces formatting debates and maintains consistency
  - Run `pnpm run prettier:write` from `/js/` directory to fix formatting
- **TypeScript**: Static type checking at both workspace and package levels
  - Workspace: `pnpm run type:check` (from `/js/`) checks all packages
  - Package: `npm run type:check` (from individual package) checks single package
  - Catches type-related errors before runtime and serves as documentation
- **Configuration Strategy**: Centralized configuration at workspace level reduces maintenance overhead
- **CI Integration**: All code quality checks run in parallel during continuous integration

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

#### Testing Framework Usage

**Jest (Most Packages):**
- Standard unit testing framework
- Configuration: `jest.config.js`
- Run: `npm test` or `jest .`

**Vitest (Mastra only):**
- Modern alternative to Jest
- Configuration: `vitest.config.ts`
- Run: `npm test` or `vitest`

**VCR Testing (Bedrock only):**
- HTTP recording/replay with Nock
- Recording mode: `BEDROCK_RECORD_MODE=record npm test`
- Management scripts: `npm run test:clear-recordings`

#### Build Configuration Patterns

**TypeScript Configs (Standard across packages):**
- `tsconfig.json` - CommonJS build
- `tsconfig.esm.json` - ES modules build  
- `tsconfig.esnext.json` - Latest JS features

**Package.json Scripts (Standard):**
```json
{
  "scripts": {
    "prebuild": "rimraf dist && pnpm run version:update",
    "build": "tsc --build [all three configs] && tsc-alias",
    "postbuild": "echo module marker && rimraf dist/test",
    "type:check": "tsc --noEmit",
    "test": "jest ." // or "vitest" for Mastra
  }
}
```

**Exports Pattern (Standard):**
```json
{
  "exports": {
    ".": {
      "import": "./dist/esm/index.js",
      "require": "./dist/src/index.js"
    }
  }
}
```

## Development Commands

### JavaScript Development

```bash
# Install dependencies (from js/ directory)
pnpm install

# Run all tests
pnpm test

# Run tests for specific package
pnpm test --filter @arizeai/openinference-instrumentation-bedrock

# Build all packages
pnpm run build
```

### Code Quality Tools

The project uses different levels for code quality checks:

**Workspace Level (run from `/js/` directory):**
```bash
# Type checking - runs across all packages
pnpm run type:check

# Linting - covers entire workspace
pnpm run lint

# Prettier formatting
pnpm run prettier:check  # Check formatting
pnpm run prettier:write  # Fix formatting
```

**Individual Package Level (run from specific package directory):**
```bash
# Type checking for single package
npm run type:check

# Note: lint and prettier are managed at workspace level only
# Individual packages don't have separate lint/prettier scripts
```

**Best Practices:**
- **Always run code quality tools from workspace level** (`/js/`) for consistency
- **Use `pnpm run type:check`** to check all packages at once
- **Use `pnpm run lint`** to check entire codebase for style issues
- **Use `pnpm run prettier:write`** to automatically fix formatting across all packages
- Individual package `type:check` is available but workspace-level is preferred

**Quick Reference:**
```bash
# Navigate to workspace root first
cd /path/to/openinference/js

# Code quality commands (run from /js/ directory)
pnpm run lint                # Lint entire workspace
pnpm run type:check          # Type check all packages
pnpm run prettier:check      # Check formatting
pnpm run prettier:write      # Fix formatting

# Individual package commands (run from package directory)
cd packages/openinference-instrumentation-bedrock
npm run type:check           # Type check single package
npm test                     # Run package tests
```

### Testing Framework

- **Jest**: Unit testing framework with comprehensive coverage
- **Nock**: HTTP mocking for VCR-style testing (used in Bedrock instrumentation)
- **Test Organization**: Individual packages contain their own test suites

### Release Process

- **Changesets**: Automated versioning and changelog generation
- **Semantic Versioning**: Follows semver for all packages
- **Automated Publishing**: CI/CD pipeline handles releases

## Key Technologies

- **OpenTelemetry**: Core tracing infrastructure
- **TypeScript**: Primary language for JavaScript packages
- **Jest**: Testing framework
- **pnpm**: Package manager and workspace tool
- **Changesets**: Version and release management

## VCR Testing Methodology

### Overview
The AWS Bedrock instrumentation uses a sophisticated VCR (Video Cassette Recorder) testing approach that allows recording real API calls and replaying them for deterministic testing.

### Recording vs Replay Modes

```bash
# Record mode - makes real API calls and saves responses
BEDROCK_RECORD_MODE=record npm test

# Replay mode (default) - uses saved recordings
npm test

# Record specific test
BEDROCK_RECORD_MODE=record npm test -- --testNamePattern="should handle tool calling"
```

### Test Cycle Workflow

1. **Initial Development**: Write failing test with expected span attributes
2. **Recording**: Run with `BEDROCK_RECORD_MODE=record` to capture real API responses
3. **Sanitization**: Auth headers automatically sanitized for security
4. **Refinement**: Iterate on instrumentation logic using recorded responses
5. **Validation**: Verify spans match expected semantic conventions using Jest inline snapshots

### Credentials Sanitization

The VCR system automatically sanitizes sensitive authentication data:

```javascript
const MOCK_AUTH_HEADERS = {
  authorization: "AWS4-HMAC-SHA256 Credential=AKIATEST1234567890AB/20250626/us-east-1/bedrock/aws4_request, SignedHeaders=accept;content-length;content-type;host;x-amz-date, Signature=fake-signature-for-vcr-testing",
  "x-amz-security-token": "FAKE-SESSION-TOKEN-FOR-VCR-TESTING-ONLY",
  "x-amz-date": "20250626T120000Z"
};
```

### Recording Management Scripts

#### Clear Recordings
```bash
# Interactive cleanup
npm run test:clear-recordings

# Remove all recordings
npm run test:clear-recordings -- --all --force

# Remove specific directory
npm run test:clear-recordings -- --dir recordings --force

# Dry run to see what would be removed
npm run test:clear-recordings -- --dry-run
```

### Nock Integration

Uses Nock for HTTP mocking with VCR-style recording:

```javascript
// Recording mode: capture real requests
nock.recorder.rec({
  output_objects: true,
  enable_reqheaders_recording: true,
});

// Replay mode: create mock from recording
nock("https://bedrock-runtime.us-east-1.amazonaws.com")
  .post(`/model/${encodeURIComponent(TEST_MODEL_ID)}/invoke`)
  .reply(200, mockResponse);
```

### Test Structure Patterns

#### Test Helper Functions
- `setupTestRecording(testName)`: Configures VCR for specific test
- `createTestClient()`: Creates BedrockRuntimeClient with appropriate credentials
- `verifySpanBasics()`: Validates core span structure
- `verifyResponseStructure()`: Validates API response format

#### Jest Inline Snapshots
Uses Jest's `toMatchInlineSnapshot()` for precise span attribute validation:

```javascript
expect(span.attributes).toMatchInlineSnapshot(`
{
  "input.mime_type": "application/json",
  "input.value": "Hello, how are you?",
  "llm.input_messages.0.message.content": "Hello, how are you?",
  "llm.input_messages.0.message.role": "user",
  "llm.model_name": "anthropic.claude-3-5-sonnet-20240620-v1:0",
  "llm.provider": "aws",
  "llm.system": "bedrock",
  "llm.token_count.completion": 35,
  "llm.token_count.prompt": 13,
  "llm.token_count.total": 48,
  "openinference.span.kind": "LLM",
  "output.value": "Hello! As an AI language model, I don't have feelings, but I'm functioning well and ready to assist you. How can I help you today?"
}
`);
```

### TDD Development Cycle

#### Phase 1: Foundation Testing
- Basic InvokeModel instrumentation
- Request/response attribute extraction
- Token count handling
- Error handling scenarios

#### Phase 2: Advanced Features
- Tool calling support
- Multi-modal messages (text + images)
- Streaming responses
- Complex conversation flows

#### Phase 3: Comprehensive Coverage
- Converse API support
- System prompts
- Error edge cases
- Performance optimization

### Advanced Tooling

#### Package Scripts
```json
{
  "scripts": {
    "test": "jest .",
    "test:record": "BEDROCK_RECORD_MODE=record jest .",
    "test:clear-recordings": "node scripts/clear-recordings.js",
    "test:watch": "jest . --watch",
    "test:coverage": "jest . --coverage"
  }
}
```

#### Recording File Structure
```
test/recordings/
├── should-create-spans-for-invokemodel-calls.json
├── should-handle-tool-calling-with-function-definitions.json
└── should-handle-api-errors-gracefully.json
```

### Best Practices

1. **Test Naming**: Use descriptive test names that become recording filenames
2. **Recording Hygiene**: Clear recordings when changing test logic
3. **Credential Safety**: Never commit real credentials; use mock values
4. **Span Validation**: Use inline snapshots for precise attribute checking
5. **Error Handling**: Test both success and failure scenarios
6. **Tool Testing**: Verify tool calling attributes match semantic conventions

### Security Considerations

- Real AWS credentials only used during recording (local development)
- All auth headers sanitized before saving recordings
- Mock credentials used for replay mode
- No sensitive data committed to repository