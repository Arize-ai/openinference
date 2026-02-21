# OpenInference Repository Guide

OpenInference is an OpenTelemetry-based AI/ML instrumentation library — a multi-language monorepo published to PyPI, `@arizeai` npm, and Maven Central. Owned by Arize AI.

- **Python**: instrumentation packages for OpenAI, LangChain, LlamaIndex, DSPy, and more
- **JavaScript/TypeScript**: packages (core + instrumentors) with pnpm workspaces
- **Java**: packages covering LangChain4j and Spring AI
- **Specification**: Language-agnostic OpenInference semantic conventions in `spec/`

---

## Repository Structure

```
openinference/
├── python/
│   ├── openinference-instrumentation/        # Core: OITracer, TraceConfig, suppress_tracing
│   ├── openinference-semantic-conventions/   # Python semconv constants
│   └── instrumentation/                      # individual instrumentor packages
├── js/
│   └── packages/                             # core + instrumentors + specialized packages
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

1. **Suppress Tracing** — check the suppression key **per request** before creating any spans; skip tracing when suppressed
2. **Context Attribute Propagation** — read session ID, user ID, metadata, and tags from OTel context and attach to each span
3. **Trace Configuration** — wrap the raw OTel tracer with `OITracer` to support `TraceConfig` masking

See each workspace's `AGENTS.md` for language-specific implementation details.

---

## Development Commands

```bash
# Python — see python/AGENTS.md for full tox command reference
pip install tox-uv==1.11.2 && pip install -r python/dev-requirements.txt
tox run -e add_symlinks && tox run-parallel

# JavaScript — see js/AGENTS.md for full command reference (MUST use pnpm, not npm or yarn)
pnpm install --frozen-lockfile -r && pnpm run -r build

# Java — see java/AGENTS.md for full command reference
cd java && ./gradlew build && ./gradlew test
```

---

## OpenInference Semantic Conventions

`openinference.span.kind` is **REQUIRED** on all OpenInference spans. Valid values: `LLM`, `EMBEDDING`, `CHAIN`, `RETRIEVER`, `RERANKER`, `TOOL`, `AGENT`, `GUARDRAIL`, `EVALUATOR`, `PROMPT`.

Key attribute namespaces: `input.*`, `output.*`, `llm.*`, `embedding.*`, `document.*`, `retrieval.*`, `session.*`, `user.*`, `tag.*`, `metadata`, `tool.*`

See `spec/semantic_conventions.md` for the full attribute reference. The flattened array
format (e.g. `llm.input_messages.0.message.role`) is documented in `spec/llm_spans.md`
and `spec/tool_calling.md`.

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
- `spec/AGENTS.md` — Spec files index (see `spec/semantic_conventions.md` for the full attribute reference)
