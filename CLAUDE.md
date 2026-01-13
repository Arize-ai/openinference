# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

OpenInference is a multi-language monorepo providing OpenTelemetry-based instrumentation for AI/ML applications. The repository contains:

- **Python**: Instrumentation for popular frameworks (OpenAI, LangChain, LlamaIndex, DSPy, etc.)
- **JavaScript/TypeScript**: Node.js instrumentations with pnpm workspaces
- **Java**: Instrumentation for LangChain4j and Spring AI
- **Specification**: OpenInference semantic conventions in the `spec/` directory

Each language workspace has its own development workflow and tools. See language-specific sections below.

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
- [tox](https://github.com/tox-dev/tox) for automation
- [ruff](https://github.com/astral-sh/ruff) for formatting and linting
- [mypy](https://github.com/python/mypy) for type checking

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

### Working on Issues

1. **Understand the Issue**: Read the issue description and any linked documentation
2. **Explore the Codebase**: Use appropriate tools (Grep, Glob, Read) to understand existing patterns
3. **Follow Language Conventions**: Check the language-specific guide (especially `js/CLAUDE.md` for JavaScript work)
4. **Test Your Changes**: Run tests using the appropriate command for your language
5. **Format and Lint**: Ensure code passes all quality checks before committing

### Creating New Instrumentors

Each language has specific patterns for creating instrumentors:

**Python**: See `python/DEVELOPMENT.md` for detailed guidance on:
- Minimal feature set (suppress tracing, customizing spans, trace configuration)
- Using `OITracer` wrapper
- Testing with tox
- Publishing to PyPI and Conda-Forge

**JavaScript**: See `js/CLAUDE.md` for:
- Extending `InstrumentationBase`
- Using `OITracer` from core
- Required features (suppress tracing, context attributes, trace config)
- Testing patterns with Vite

**Java**: Follow existing patterns in `java/instrumentation/` directory

## Release Management

This repository uses [release-please](https://github.com/googleapis/release-please) for automated version management and releases.

**Configuration Files:**
- `.release-please-manifest.json`: Package versions
- `release-please-config.json`: Release configuration

**Adding New Packages:**
Add your package to both config files to enable automated releases.

**Release Process:**
1. Release-please creates PRs with version bumps and changelogs
2. Merge the PR to trigger a GitHub release
3. Packages are automatically published to their respective registries (PyPI, npm, Maven Central)

## Contributing Guidelines

See `CONTRIBUTING` for detailed contribution guidelines, including:

- Code of conduct
- Branch organization (submit to `main`)
- Pull request requirements and best practices
- Code review standards
- CLA requirements

**Key Points:**
- Keep PRs small and focused
- Follow conventional commit format for PR titles
- Reference issues in PR descriptions
- Include tests for new features and bug fixes
- Run all quality checks before submitting

## Important Notes

### Cross-Language Consistency

All instrumentors across languages should:
1. Support suppressing tracing
2. Propagate context attributes (session ID, user ID, metadata, tags, etc.)
3. Respect trace configuration for masking sensitive data
4. Follow OpenInference semantic conventions
5. Include comprehensive tests

### Testing Requirements

- **Python**: Use tox for consistent environments
- **JavaScript**: Use Vite with manual module mocking for instrumentation timing
- **Java**: Use Gradle test tasks

### Documentation

- Update README files when adding new instrumentors
- Include examples in the appropriate `examples/` directory
- Update the main README.md table when adding new packages

### Security

- Never commit sensitive data (.env files, credentials, etc.)
- Be cautious with PII and use trace configuration for masking
- Follow security best practices for instrumentation code

## Helpful Resources

- [OpenInference Specification](https://arize-ai.github.io/openinference/spec/)
- [JavaScript Documentation](https://arize-ai.github.io/openinference/js/)
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) - Native OpenInference backend
- [Community Slack](https://join.slack.com/t/arize-ai/shared_invite/zt-3lqwr2oc3-7rhdyYEh82zJL_UhPKrb0A)

## Language-Specific Details

For more detailed language-specific guidance:
- **JavaScript/TypeScript**: See `js/CLAUDE.md`
- **Python**: See `python/DEVELOPMENT.md`
- **Java**: See `java/README.md`
