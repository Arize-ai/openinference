# Code Review Guidelines

The keywords "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this specification are to be interpreted as described in BCP 14 [RFC2119] [RFC8174] when, and only when, they appear in all capitals, as shown here.

An implementation is compliant if it satisfies all "MUST", "MUST NOT", "REQUIRED", "SHALL", and "SHALL NOT" requirements defined in this specification. An implementation that fails to satisfy any such requirement is not compliant.

## Always Check

- Instrumentors MUST support suppressing tracing, MUST propagate context attributes (session ID, user ID, metadata, tags), and MUST respect TraceConfig masking on every path the change touches (`AGENTS.md` § Requirements, `python/AGENTS.md` § Required Features, `js/AGENTS.md` § Instrumentor Pattern)
- Span attribute names MUST come from the OpenInference semantic conventions packages, never string literals, and MUST follow the spec in `spec/`
- Every span an instrumentor starts MUST be ended, including when the wrapped call raises and when a stream is abandoned or closed early
- Instrumentation MUST NOT raise into or change the behavior of the instrumented library; errors in the instrumentor itself MUST be caught and logged, through stdlib `logging` in Python and `diag` (never `console.*`) in TypeScript (`python/AGENTS.md` § Logging and Diagnostics, `js/AGENTS.md` § Instrumentor Pattern)
- Wrappers MUST target methods that exist with the signatures they assume, across the library versions the package supports (its pinned test version and `-latest`)
- Tests MUST fail when the behavior they cover is removed — tests that only exercise mocks or restate the implementation MUST NOT be added. Tests SHOULD drive the real SDK client through the instrumentor against a cassette or a stubbed HTTP layer and assert on the spans an in-memory exporter collects (`python/AGENTS.md` § Testing and Quality, `js/AGENTS.md` § Testing)
- Cassettes MUST NOT contain API keys or other credentials
- The public API MUST stay backwards compatible: CONTRIBUTING § Branch Organization allows no breaking changes on `main`
- Changes to packages under `js/packages/` MUST include a changeset
- Package versions and `CHANGELOG.md` files MUST NOT be edited by hand: release-please and changesets own them
- Production dependencies MUST NOT be pinned to a pre-release of the instrumented library. OpenTelemetry's beta-only packages (`opentelemetry-instrumentation`, `opentelemetry-semantic-conventions`, versioned `0.NNbN`) are exempt

## Style

- Python changes MUST follow `python/AGENTS.md` § Key Patterns (attribute helpers, required features, logging, instrumentor conventions); TypeScript changes MUST follow `js/AGENTS.md` and the `openinference-typescript` skill
- Variable names SHOULD be descriptive; in TypeScript, single-letter names MUST NOT be used (`openinference-typescript` skill § Naming)
- Composition SHOULD be preferred over inheritance
- Comments SHOULD explain why, not what — comments that narrate the code, address the reviewer, or describe how the change was made MUST NOT be added
- Python instrumentor tests SHOULD assert exhaustively on span attributes (`python-code-reviewer` skill § 2.3: pop each expected attribute, then assert none remain)

## Patterns to Flag

- Catching broad exception types without logging or re-raising MUST NOT occur in code paths that hide the instrumented library's own errors
- Instrumentors MUST NOT keep state that survives `uninstrument()`
- Existing helpers in `openinference-instrumentation` and `@arizeai/openinference-core` MUST be reused rather than reimplemented
- Abstractions with a single call site SHOULD NOT be introduced
- Public API parameters, config options and exports the change does not need MUST NOT be added
- Code and exports the change leaves unused MUST be removed

## Security

- Secrets (API keys, tokens, auth headers) MUST NOT be recorded as span attributes, logged, or committed in cassettes
- Attributes MUST be set through the `OITracer` span, which applies TraceConfig masking, never on a raw OpenTelemetry span; data a hide flag covers MUST NOT leak through another attribute (for example, messages that `hide_input_messages` hides showing up in `input.value`)

## Reporting Findings

- Each finding MUST be verified against the code before it is reported: quote the line, and for a bug name the input or state that triggers it
- Only problems the change introduces SHOULD be reported — pre-existing issues and anything a linter, formatter or type checker catches MUST NOT be
- Each root cause MUST be reported once, listing every location
- Each finding MUST carry a severity: **Blocking** (wrong behavior, security, data loss, or a broken MUST) or **Non-blocking** (a broken SHOULD). Nits and praise SHOULD NOT be posted
- A convention finding MUST cite the rule it breaks in this file, `python/AGENTS.md`, `js/AGENTS.md`, or a skill under `.claude/skills/`
- A finding SHOULD fit in five lines: what is wrong, the evidence, and a concrete fix

## Skip

- Cassettes and recorded fixtures, except to check them for credentials
- Generated files under `__generated__` directories
- Files under `node_modules/`, `dist/`, or `build/`
