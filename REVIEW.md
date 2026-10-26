# Code Review Guidelines

The keywords "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this specification are to be interpreted as described in BCP 14 [RFC2119] [RFC8174] when, and only when, they appear in all capitals, as shown here.

An implementation is compliant if it satisfies all "MUST", "MUST NOT", "REQUIRED", "SHALL", and "SHALL NOT" requirements defined in this specification. An implementation that fails to satisfy any such requirement is not compliant.

## Always Check

- Instrumentors MUST support suppressing tracing, MUST propagate context attributes (session ID, user ID, metadata, tags), and MUST respect TraceConfig masking on every path the change touches
- Span attribute names MUST come from the OpenInference semantic conventions packages, never string literals, and MUST follow the spec in `spec/`
- Every span an instrumentor starts MUST be ended, including when the wrapped call raises and when a stream is abandoned or closed early
- Instrumentation MUST NOT raise into or change the behavior of the instrumented library; errors in the instrumentor itself MUST be caught and logged
- Wrappers MUST target methods that exist with the signatures they assume, across the library versions the package supports (its pinned test version and `-latest`)
- Tests MUST fail when the behavior they cover is removed — tests that only exercise mocks or restate the implementation MUST NOT be added. Tests SHOULD drive the real SDK client through the instrumentor against a cassette or mock transport and assert on the spans an in-memory exporter collects
- Cassettes MUST NOT contain API keys or other credentials
- Changes to packages under `js/packages/` MUST include a changeset
- Package versions and `CHANGELOG.md` files MUST NOT be edited by hand: release-please and changesets own them
- Production dependencies MUST NOT be pinned to pre-release versions

## Style

- Variable names MUST be descriptive — short names like `x` or `cv` MUST NOT be used
- Functions MUST be named with precise action verbs — e.g. `get_attributes()` rather than `attributes()`
- Keyword or object arguments SHOULD be preferred over positional arguments for clarity — in TypeScript, functions with two or more parameters SHOULD take a single destructured object
- Python functions MUST have type annotations, and TypeScript MUST NOT use `any` where `unknown` and a type guard would do
- Composition SHOULD be preferred over inheritance
- Comments SHOULD explain why, not what — comments that narrate the code, address the reviewer, or describe how the change was made MUST NOT be added

## Patterns to Flag

- Catching broad exception types without logging or re-raising MUST NOT occur in code paths that hide the instrumented library's own errors
- Instrumentors MUST NOT keep state that survives `uninstrument()`
- Existing helpers in `openinference-instrumentation` and `@arizeai/openinference-core` MUST be reused rather than reimplemented
- Abstractions with a single call site SHOULD NOT be introduced
- Public API parameters, config options and exports the change does not need MUST NOT be added
- Code and exports the change leaves unused MUST be removed

## Security

- Secrets (API keys, tokens, auth headers) MUST NOT be recorded as span attributes, logged, or committed in cassettes
- Masking that TraceConfig asks for (inputs, outputs, messages, images) MUST apply before attributes are set on the span, not after

## Reporting Findings

- Each finding MUST be verified against the code before it is reported: quote the line, and for a bug name the input or state that triggers it
- Only problems the change introduces SHOULD be reported — pre-existing issues and anything a linter, formatter or type checker catches MUST NOT be
- Each root cause MUST be reported once, listing every location
- Each finding MUST carry a severity: **Blocking** (wrong behavior, security, data loss, or a broken MUST) or **Non-blocking** (a broken SHOULD). Nits and praise SHOULD NOT be posted
- A convention finding MUST cite the rule it breaks in this file or in `.agents/skills/`
- A finding SHOULD fit in five lines: what is wrong, the evidence, and a concrete fix

## Skip

- Cassettes and recorded fixtures, except to check them for credentials
- Generated files under `__generated__` directories
- Files under `node_modules/`, `dist/`, or `build/`
