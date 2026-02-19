# OpenInference Repository Guide

OpenInference is an OpenTelemetry-based AI/ML instrumentation library — a multi-language monorepo published to PyPI, `@arizeai` npm, and Maven Central. Owned by Arize AI.

- **Python**: 33 instrumentation packages for OpenAI, LangChain, LlamaIndex, DSPy, and more
- **JavaScript/TypeScript**: 13 packages (core + instrumentors) with pnpm workspaces
- **Java**: 4 packages covering LangChain4j and Spring AI
- **Specification**: Language-agnostic OpenInference semantic conventions in `spec/`

---

## Repository Structure

```
openinference/
├── python/
│   ├── openinference-instrumentation/        # Core: OITracer, TraceConfig, suppress_tracing
│   ├── openinference-semantic-conventions/   # Python semconv constants
│   └── instrumentation/                      # 33 individual instrumentor packages
├── js/
│   └── packages/                             # 13 packages (core + 8 instrumentors + specialized)
├── java/
│   ├── openinference-instrumentation/        # Core: OITracer, TraceConfig
│   ├── openinference-semantic-conventions/   # Java semconv constants
│   └── instrumentation/                      # LangChain4j, Spring AI
├── spec/                                     # Language-agnostic semantic conventions
└── scripts/                                  # Repository automation scripts
```

---

## Universal Instrumentor Requirements

Every instrumentor in **every language** must implement these three features:

### 1. Suppress Tracing

Check the suppression key before creating any spans and skip tracing when suppressed.

- **Python**: check `context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)`
- **JS/Java**: call `isTracingSuppressed(context.active())` / OITracer handles this automatically

### 2. Context Attribute Propagation

Read session ID, user ID, metadata, and tags from OTel context and attach them to each span.

- **Python**: call `get_attributes_from_context()` at span creation time
- **JS/Java**: `OITracer` handles propagation automatically

### 3. Trace Configuration

Allow masking of sensitive data via `TraceConfig`. Always wrap the raw OTel tracer with `OITracer`.

---

## Development Commands

```bash
# Python — see python/AGENTS.md for full tox command reference
pip install tox-uv==1.11.2 && pip install -r python/dev-requirements.txt
tox run -e add_symlinks && tox run-parallel

# JavaScript — see js/AGENTS.md for full command reference (MUST use pnpm, not npm)
pnpm install --frozen-lockfile -r && pnpm run -r prebuild && pnpm run -r build

# Java — see java/AGENTS.md for full command reference
cd java && ./gradlew build && ./gradlew test
```

---

## OpenInference Semantic Conventions

`openinference.span.kind` is **REQUIRED** on all OpenInference spans. Valid values: `LLM`, `EMBEDDING`, `CHAIN`, `RETRIEVER`, `RERANKER`, `TOOL`, `AGENT`, `GUARDRAIL`, `EVALUATOR`, `PROMPT`.

Key attribute namespaces: `input.*`, `output.*`, `llm.*`, `embedding.*`, `document.*`, `retrieval.*`, `session.*`, `user.*`, `tag.*`, `metadata`, `tool.*`

See `spec/AGENTS.md` for the full attribute reference including flattened array format.

---

## Code Style

| Language | Format | Lint | Types | Test | Line Length |
|----------|--------|------|-------|------|-------------|
| Python | ruff | ruff | mypy strict | pytest (`asyncio_mode=auto`) | 100 chars |
| JavaScript | Prettier | ESLint | TypeScript strict | Vitest | — |
| Java | Palantir via Spotless | — | — | JUnit 5 | — |

---

## Release Management

- **Python + JS**: automated via release-please; PRs trigger PyPI and npm releases
- **JS**: `pnpm changeset` is required before any PR touching `js/` is merged
- **Java**: JReleaser publishes to Maven Central
- **Config files**: `.release-please-manifest.json`, `release-please-config.json`

---

## Contributing

- Follow conventional commit format for PR titles (e.g., `feat(openai): add streaming support`)
- Keep PRs small and focused
- Include tests for new features and bug fixes
- Run all quality checks before submitting
- Never commit sensitive data (.env files, credentials); use `TraceConfig` for masking PII

---

## See Also

- `python/AGENTS.md` — Python workspace details, tox reference, instrumentor patterns
- `js/AGENTS.md` — JavaScript workspace details, pnpm setup, TypeScript patterns
- `java/AGENTS.md` — Java workspace details, Gradle setup, instrumentor patterns
- `spec/AGENTS.md` — Full semantic conventions attribute reference
