# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Repository Overview

OpenInference is a multi-language monorepo providing OpenTelemetry-based instrumentation for AI/ML applications:

- **Python**: Instrumentation for OpenAI, LangChain, LlamaIndex, DSPy, etc.
- **JavaScript/TypeScript**: Node.js instrumentations with pnpm workspaces
- **Java**: Instrumentation for LangChain4j and Spring AI
- **Specification**: OpenInference semantic conventions in `spec/` - language-agnostic definitions of span attributes, span kinds (LLM, Chain, Tool, Agent, Retriever, Embedding, Reranker, Guardrail, Evaluator), and tracing standards for AI observability

## Repository Structure

```
.
├── python/                    # Python packages and instrumentations
│   ├── instrumentation/       # Individual instrumentor packages
│   ├── openinference-instrumentation/
│   └── openinference-semantic-conventions/
├── js/                        # JavaScript/TypeScript workspace
│   ├── packages/              # pnpm workspace packages
│   └── examples/
├── java/                      # Java packages
│   ├── instrumentation/
│   ├── openinference-instrumentation/
│   └── openinference-semantic-conventions/
├── spec/                      # OpenInference specification
└── scripts/                   # Repository automation scripts
```

## Core Architecture

### Shared Instrumentation Pattern

All instrumentors (Python, JavaScript, Java) follow the same architectural pattern:

1. **OITracer Wrapper**: Each language has an `OITracer` class that wraps the native OpenTelemetry tracer to provide consistent behavior across:
   - Automatic context attribute propagation (session ID, user ID, metadata, tags)
   - Trace configuration support (hiding sensitive data, payload size limits)
   - OpenInference semantic convention enforcement

2. **Required Features**: Every instrumentor must implement:
   - **Suppress Tracing**: Respect suppression context to pause tracing temporarily
   - **Context Attributes**: Automatically attach context attributes (session, user, metadata, tags) to spans
   - **Trace Configuration**: Support masking sensitive data and controlling payload sizes via `TraceConfig`

3. **Python Namespace Package**: `openinference-instrumentation` is a namespace package composed using symlinks. After making changes to core packages, run `tox run -e add_symlinks` to update symlinks.

4. **Cross-Package Dependencies**:
   - **Python**: Uses symlinks via `tox run -e add_symlinks`
   - **JavaScript**: Uses pnpm workspace with prebuild script (`pnpm run -r prebuild`) to generate symlinks
   - **Java**: Gradle multi-project build

## Essential Commands by Language

### Python

**Package Management:**
```bash
# Install development dependencies
pip install -r python/dev-requirements.txt

# Install tox-uv for testing automation
pip install tox-uv==1.11.2

# Compose namespace package
tox run -e add_symlinks

# Install in editable mode
pip install -e python/openinference-instrumentation
```

**Testing and Quality:**
```bash
# Run all CI checks in parallel
tox run-parallel

# Run tests for specific package (e.g., openai)
tox run -e test-openai

# Run a specific test file
tox run -e test-openai -- tests/test_specific.py

# Run linting and formatting
tox run -e ruff-openai

# Run type checking
tox run -e mypy-openai

# Run all checks for a package
tox run -e ruff-mypy-test-openai

# Multiple environments (comma-separated)
tox run -e ruff-openai,ruff-semconv
```

**Understanding tox Factor System:**
- Environment strings like `ruff-mypy-test-openai` are hyphen-delimited lists of "factors"
- Each factor (e.g., `ruff`, `mypy`, `test`, `openai`) is defined in `tox.ini` and specifies actions to perform
- tox creates a virtual environment and runs all specified factor actions in that environment
- This allows flexible composition: `ruff-openai` runs only linting, `test-openai` runs only tests, `ruff-mypy-test-openai` runs all three

**Key Tools:**
- tox with tox-uv for automation
- ruff for formatting and linting
- mypy for type checking

### JavaScript/TypeScript

**IMPORTANT: Read `js/CLAUDE.md` for critical JavaScript-specific architecture details including `OITracer` patterns, testing with manual module mocking, and required instrumentor features.**

**Package Management:**
```bash
# MUST use pnpm, not npm
pnpm install --frozen-lockfile -r

# Build all packages (REQUIRED after changes - includes prebuild)
# Prebuild generates version files and cross-package symlinks
pnpm run -r build

# Run prebuild only (generates symlinks for cross-package dependencies)
pnpm run -r prebuild
```

**Critical**: Always run `pnpm run -r build` after making changes. The prebuild step creates symlinks between packages (e.g., core utilities to instrumentors), similar to Python's `add_symlinks`.

**Testing and Quality:**
```bash
# Run tests
pnpm run -r test

# Type checking
pnpm run type:check

# Linting
pnpm run lint

# Formatting
pnpm run prettier:check
pnpm run prettier:write
```

**Version Management:**
```bash
# Create changeset for version bumping
pnpm changeset
```

### Java

**Build and Test:**
```bash
# Build all Java packages
cd java && ./gradlew build

# Run tests
./gradlew test

# Publish to Maven Local
./gradlew publishToMavenLocal
```

## Development Workflow

1. Explore the codebase to understand existing patterns
2. **Check language-specific guides** (contains critical architecture details):
   - **Python**: `python/DEVELOPMENT.md` - minimal feature set requirements (suppress tracing, context attributes, trace config)
   - **JavaScript**: `js/CLAUDE.md` - OITracer patterns, testing with manual mocks, instrumentor implementation
   - **Java**: `java/README.md` - manual and auto-instrumentation patterns
3. Run tests using the appropriate command for your language
4. Ensure code passes all quality checks before committing

### Critical Development Patterns

**Cross-Package Changes:**
- **Python**: After modifying core packages, run `tox run -e add_symlinks` to update namespace package symlinks
- **JavaScript**: Always run `pnpm run -r build` (includes prebuild) to regenerate symlinks between packages
- **Java**: Gradle handles dependencies automatically via multi-project build

**Testing Before Commits:**
- Python: `tox run -e ruff-mypy-test-<package>`
- JavaScript: `pnpm run -r test && pnpm run type:check && pnpm run lint`
- Java: `./gradlew build test`

**Version Management:**
- Versions are managed by `release-please` based on conventional commits
- Configuration: `.release-please-manifest.json` (version mappings) and `release-please-config.json` (package configs)
- JavaScript also uses changesets: `pnpm changeset` before merging PRs

### Creating New Instrumentors

See language-specific documentation:
- **Python**: `python/DEVELOPMENT.md` - includes minimal feature set requirements and setup instructions
- **JavaScript**: `js/CLAUDE.md` - includes OITracer patterns and testing considerations
- **Java**: Existing patterns in `java/instrumentation/`

## OpenInference Specification

The `spec/` directory contains the OpenInference semantic conventions specification:

- **Span Attributes**: Standardized attribute names for AI/ML operations (e.g., `llm.model_name`, `llm.token_count.prompt`)
- **Span Kinds**: Defines semantic meaning of operations (LLM, Chain, Tool, Agent, Retriever, Embedding, Reranker, Guardrail, Evaluator)
- **Language-Agnostic**: These conventions are implemented across Python, JavaScript, and Java
- **Transport-Agnostic**: Designed to work with JSON, ProtoBuf, DataFrames, etc.

Each language's `openinference-semantic-conventions` package provides constants/enums for these conventions.

## Release Management

This repository uses release-please for automated version management.

**Configuration:**
- `.release-please-manifest.json` - maps package paths to current versions
- `release-please-config.json` - defines package configurations and release behavior

**Process:**
1. Release-please monitors conventional commits on `main`
2. Creates PRs with version bumps and changelogs based on commit types (feat, fix, etc.)
3. Merge the PR to trigger releases to PyPI, npm, and Maven Central
4. **JavaScript**: Also uses changesets (`pnpm changeset`) before merging PRs

## Contributing

See `CONTRIBUTING` for detailed guidelines. Key points:
- Keep PRs small and focused (easier to review, less likely to introduce bugs)
- **PR titles must follow conventional commit format** (e.g., `feat:`, `fix:`, `chore:`) - used to generate release notes
- Include tests for new features and bug fixes
- Run all quality checks before submitting:
  - Python: `tox run -e ruff-mypy-test-<package>`
  - JavaScript: `pnpm run -r test && pnpm run type:check && pnpm run lint`
  - Java: `./gradlew build test`
- JavaScript PRs should include a changeset: `pnpm changeset`

## Requirements

### All Instrumentors Must (Across All Languages):

1. **Support Suppressing Tracing**:
   - Python: Check `_SUPPRESS_INSTRUMENTATION_KEY` from OpenTelemetry context
   - JavaScript: Use `isTracingSuppressed()` from `@opentelemetry/core`
   - Java: Implement `.uninstrument()` method

2. **Propagate Context Attributes**: Automatically attach to spans:
   - Session ID (for multi-turn conversations)
   - User ID (for tracking conversations per user)
   - Metadata (custom operational information)
   - Tags (filtering keywords)
   - Prompt template information (template management)

3. **Respect Trace Configuration**: Support `TraceConfig` for:
   - Hiding input/output messages
   - Masking sensitive data (PII)
   - Controlling payload sizes (e.g., base64 image sizes)

4. **Follow OpenInference Semantic Conventions**:
   - Use proper span kinds (LLM, Chain, Tool, Agent, Retriever, Embedding, Reranker, Guardrail, Evaluator)
   - Set standard attributes as defined in `spec/`

5. **Include Comprehensive Tests**:
   - Test suppress tracing functionality
   - Test context attribute propagation
   - Test trace configuration masking
   - Test instrumentation-specific features

### Implementation Pattern

All instrumentors should use `OITracer` wrapper instead of raw OpenTelemetry tracer:
- **Python**: `from openinference.instrumentation import OITracer`
- **JavaScript**: `import { OITracer } from '@arizeai/openinference-core'`
- **Java**: `import io.openinference.instrumentation.OITracer`

### Security
- Never commit sensitive data (.env files, credentials)
- Use trace configuration for masking PII
- Test that sensitive data can be properly masked

## Additional Documentation

- **JavaScript/TypeScript**: `js/CLAUDE.md`
- **Python**: `python/DEVELOPMENT.md`
- **Java**: `java/README.md`
