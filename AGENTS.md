# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Repository Overview

OpenInference is a multi-language monorepo providing OpenTelemetry-based instrumentation for AI/ML applications:

- **Python**: Instrumentation for OpenAI, LangChain, LlamaIndex, DSPy, etc.
- **JavaScript/TypeScript**: Node.js instrumentations with pnpm workspaces
- **Java**: Instrumentation for LangChain4j and Spring AI
- **Specification**: OpenInference semantic conventions in `spec/`
- **Semantic convention registry**: Machine-readable (OpenTelemetry Weaver) model of the conventions in `semantic-conventions/`

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
├── semantic-conventions/      # Machine-readable registry of the conventions (Weaver YAML)
└── scripts/                   # Repository automation scripts
```

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

# Run linting and formatting
tox run -e ruff-openai

# Run type checking
tox run -e mypy-openai

# Run all checks for a package
tox run -e ruff-mypy-test-openai
```

**Key Tools:**
- tox for automation
- ruff for formatting and linting
- mypy for type checking

### JavaScript/TypeScript

**See `js/CLAUDE.md` for detailed JavaScript-specific guidance.**

**Package Management:**
```bash
# MUST use pnpm, not npm
pnpm install --frozen-lockfile -r

# Build all packages (includes prebuild)
pnpm run -r build

# Run prebuild only
pnpm run -r prebuild
```

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

### Semantic Convention Registry

**See `semantic-conventions/README.md` for the model layout and modeling rules.**

The YAML under `semantic-conventions/model` is the source of truth for attribute keys, types, and well-known values.
Generated from it, and never edited by hand: the attribute tables in `spec/attributes/semantic-conventions.md` and the
constants in the Python, Go, TypeScript, and Java `openinference-semantic-conventions` packages.
To add or change an attribute, edit the model and regenerate.

```bash
# Run everything CI runs (needs weaver or docker/podman, and uv)
make -C semantic-conventions check-all

# Regenerate the registry docs, the spec tables, and the language packages after changing the model
make -C semantic-conventions generate
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
2. Check language-specific guides (`js/CLAUDE.md`, `python/DEVELOPMENT.md`)
3. Run tests using the appropriate command for your language
4. Ensure code passes all quality checks before committing

### Creating New Instrumentors

See language-specific documentation:
- **Python**: `python/DEVELOPMENT.md`
- **JavaScript**: `js/CLAUDE.md`
- **Java**: Existing patterns in `java/instrumentation/`

## Release Management

This repository uses release-please for automated version management.

**Configuration:** `.release-please-manifest.json` and `release-please-config.json`

**Process:**
1. Release-please creates PRs with version bumps and changelogs
2. Merge the PR to trigger releases to PyPI, npm, and Maven Central

## Contributing

See `CONTRIBUTING` for detailed guidelines. Key points:
- Keep PRs small and focused
- Follow conventional commit format for PR titles
- Include tests for new features and bug fixes
- Run all quality checks before submitting

## Requirements

### All Instrumentors Must:
1. Support suppressing tracing
2. Propagate context attributes (session ID, user ID, metadata, tags)
3. Respect trace configuration for masking sensitive data
4. Follow OpenInference semantic conventions
5. Include comprehensive tests

### Security
- Never commit sensitive data (.env files, credentials)
- Use trace configuration for masking PII

## Additional Documentation

- **JavaScript/TypeScript**: `js/CLAUDE.md`
- **Python**: `python/DEVELOPMENT.md`
- **Java**: `java/README.md`
