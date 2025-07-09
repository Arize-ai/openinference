# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Additional Memory Files

This project maintains additional memory files that should be referenced for complete context:

- **SESSION_SUMMARY.md**: Detailed session history, current project state, and TDD progress for AWS Bedrock instrumentation
- **BEST_PRACTICES.md**: Code review standards, development methodology preferences, and critical architectural decisions

These files contain essential context about ongoing work, established patterns, and user preferences that inform development decisions.

## Development Insights on Code Quality Tools

### Linting, Formatting, and Type Checking
- **Purpose of Code Quality Tools**: Ensure consistent code style, catch potential errors early, and improve overall code maintainability
- **Prettier**: Used for consistent code formatting across both Python and JavaScript/TypeScript projects
  - Automatically formats code to a standard style
  - Reduces bike-shedding about code formatting
  - Integrated into CI/CD pipelines to enforce formatting
- **Linting Strategies**:
  - Python uses `ruff` for fast, comprehensive linting
  - JavaScript/TypeScript uses ESLint for static code analysis
  - Linters help identify potential bugs, style inconsistencies, and anti-patterns
- **Type Checking**:
  - Python uses `mypy` for static type checking
  - TypeScript provides built-in type checking
  - Type checking catches type-related errors before runtime
  - Improves code reliability and serves as a form of documentation
- **CI Integration**: All code quality checks are run in parallel during continuous integration to catch issues early
- **Configuration Approach**: Maintain minimal, consistent configuration across packages to reduce maintenance overhead

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

### Development Patterns

- **Monorepo Architecture**: Uses pnpm workspaces for JavaScript packages
- **Consistent API**: All instrumentation libraries follow similar patterns
- **OpenTelemetry Integration**: Built on OpenTelemetry for tracing infrastructure
- **Semantic Conventions**: Standardized attribute naming across all instrumentations
- **Test-Driven Development**: Comprehensive test coverage with VCR-style testing

## Development Commands

### JavaScript Development

```bash
# Install dependencies (from js/ directory)
pnpm install

# Run all tests
pnpm test

# Run tests for specific package
pnpm test --filter @arizeai/openinference-instrumentation-bedrock

# Type checking
pnpm run type:check

# Linting
pnpm run lint

# Formatting
pnpm run prettier:check
pnpm run prettier:write

# Build all packages
pnpm run build
```

### Quality Assurance

- **ESLint**: Static code analysis with TypeScript support
- **Prettier**: Consistent code formatting
- **TypeScript**: Static type checking
- **Jest**: Unit testing framework
- **Nock**: HTTP mocking for VCR-style testing

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