# OpenInference Semantic Conventions Registry

The machine-readable definition of the OpenInference semantic conventions: every attribute, its type and well-known
values, and which attributes a span of each kind carries.

It is written in the same format as the OpenTelemetry semantic conventions (the
[Weaver](https://github.com/open-telemetry/weaver) `definition/2` schema) and validated with the same tool and shared
policies as [`semantic-conventions-genai`](https://github.com/open-telemetry/semantic-conventions-genai). That makes the
conventions versioned, checkable, and consumable by other registries.

The prose specification is in [`spec/`](../spec/README.md). The model here is the source of truth for attribute keys,
types, and well-known values. The spec's attribute tables and the constants in the Python, Go, TypeScript, and Java
semantic convention packages are generated from it.

## Layout

```
semantic-conventions/
├── model/                  # The registry. Hand-written YAML, one directory per domain.
│   ├── manifest.yaml       #   Registry name, schema URL (version), and dependencies
│   ├── openinference/      #   openinference.span.kind, shared attribute groups, CHAIN and GUARDRAIL spans
│   ├── llm/                #   registry.yaml defines attributes; spans.yaml defines the LLM span
│   └── ...                 #   input-output, context, message, prompt, tool, multimodal, document,
│                           #   retrieval, reranker, embedding, decision, agent, feedback
├── policies/
│   ├── check/openinference/  # OpenInference-specific Rego policies
│   └── exceptions.json       # Accepted findings from the shared OpenTelemetry policies, each with a reason
├── templates/registry/     # Weaver templates: markdown/ for docs and spec tables; python/, go/,
│                           #   typescript/, java/ for the language packages
├── docs/registry/          # GENERATED reference (do not edit by hand)
├── vendor/opentelemetry/   # The OpenTelemetry attributes the model reuses, vendored for offline generation
├── tests/live-check/       # Sample wire-format keys that must / must not resolve against the registry
├── tests/policies/         # A registry that breaks each OpenInference policy, to prove that the rules fire
├── tests/consumer/         # A registry that depends on this one, to prove that others can
├── scripts/                # Checks run by the Makefile
├── versions.env            # Pinned Weaver and policy versions
└── Makefile
```

Each `registry.yaml` defines attributes. Each `spans.yaml` defines the span for an `openinference.span.kind` by
referencing those attributes with a requirement level. Browse the result in [`docs/registry/`](docs/registry/README.md).

## Working on the model

You need either [Weaver](https://github.com/open-telemetry/weaver/releases) at the version pinned in
[`versions.env`](versions.env), or Docker/Podman (the Makefile then runs Weaver from its container image). The checks
that run Python scripts with dependencies also need [uv](https://github.com/astral-sh/uv).

```bash
make -C semantic-conventions check-all       # everything CI runs
make -C semantic-conventions check           # validate the model and run the policies
make -C semantic-conventions test            # check sample keys against the registry
make -C semantic-conventions generate        # regenerate docs, spec tables, and language packages after changing the model
make -C semantic-conventions check-parity    # compare the model with spec/ and the Python package
make -C semantic-conventions help            # all targets
```

To change the conventions:

1. Edit the YAML under `model/`. Add new attributes to the `registry.yaml` of their domain, and reference them from the
   `spans.yaml` of every span kind that carries them.
2. Run `make generate` and commit the result: the reference in `docs/registry/`, the tables in
   [`spec/attributes/semantic-conventions.md`](../spec/attributes/semantic-conventions.md), and the constants in the
   four language packages (see [Language packages](#language-packages)).
3. Update the prose in [`spec/`](../spec/README.md) where the change needs explaining. `make check-parity` fails if
   a new attribute is not picked up by any table in the spec (see [Spec tables](#spec-tables)).
4. Use a conventional commit scoped to this directory so the change lands in the registry changelog, e.g.
   `feat(semantic-conventions): add llm.reasoning_effort`.

### Modeling rules

- **Stability.** Everything is `development` for now. Marking an attribute `stable` is a commitment that the shared
  policies then enforce, and it cannot be undone: a stable attribute can be deprecated, but it cannot change type or
  go back to `development`. The long-established attributes are to be promoted once real instrumentation output has
  been checked against the model
  ([#4419](https://github.com/Arize-ai/openinference/issues/4419)). Until then, removals fail CI at any stability,
  but a changed type does not.
- **Never remove or rename an attribute.** Deprecate it with `deprecated:` and point to the replacement. On pull
  requests, CI compares the model with the base branch and fails on removals and incompatible changes.
- **Reuse OpenTelemetry attributes** when one exists for the same concept. `session.id` and `user.id` are referenced
  from the OpenTelemetry registry (a dependency in [`model/manifest.yaml`](model/manifest.yaml)), not redefined.
  Run `make vendor` after referencing another one.
- **Well-known values** are enum `members`. Enums are open: a custom value may be used when no member applies.
  Members are append-only: add a new member at the end and never reorder. The generated Java enums take their
  ordinals from this order, and the existing order is that of the published Java enums.
- **Resource attributes.** An attribute recorded on the OpenTelemetry resource, not on spans, is the identity or
  description of an entity in [`model/openinference/entities.yaml`](model/openinference/entities.yaml). The language
  packages then expose it as a resource attribute (`ResourceAttributes.PROJECT_NAME`, `SEMRESATTRS_PROJECT_NAME`)
  and leave it out of the span attribute classes.
- **Structured values.** OTLP span attributes cannot hold objects, so OpenInference flattens them or serializes them
  to JSON. The model records which, in an annotation that policies check and tools can read:

  ```yaml
  - key: llm.input_messages
    type: template[any]          # a key prefix: llm.input_messages.<index>.message.role
    annotations:
      openinference:
        encoding: flattened_list # or flattened_object, or json (on a string)
        item_namespaces: [message]
  ```

  `template[any]` is what lets Weaver match a flattened key such as `llm.input_messages.0.message.role` to the
  `llm.input_messages` attribute; `item_namespaces` says its items are `message.*` attributes. When only part of a
  namespace applies, list the attributes in `item_attributes` instead: `llm.tools` items are
  `[tool.name, tool.description, tool.json_schema]`, not the `tool.parameters` and `tool.id` of a `TOOL` span.

  Weaver only matches the prefix. Nothing checks yet that what follows it is one of the declared items, or has the
  right type ([#4419](https://github.com/Arize-ai/openinference/issues/4419));
  [`tests/live-check/unvalidated.json`](tests/live-check/unvalidated.json) pins that gap.
- **Types Weaver cannot express.** When instrumentations record more than one primitive type for a key, give the
  attribute its main type and list the others in `annotations.openinference.also_accepts`, e.g. `[int]` on
  `document.id`.
- **Constant names.** The language packages name a constant after its key (`llm.model_name` is `LLM_MODEL_NAME`).
  Where a published constant is named differently, record the key its name derives from in
  `annotations.openinference.constant_key`, e.g. `tool_call.function.arguments_json` on
  `tool_call.function.arguments` (`TOOL_CALL_FUNCTION_ARGUMENTS_JSON`).
- **Naming exceptions.** `metadata`, `annotations`, and `evaluations` have no namespace, which the shared naming policy
  rejects. They predate the registry and cannot be renamed, so they are listed in
  [`policies/exceptions.json`](policies/exceptions.json). Do not add to that list for new attributes.

### Hermetic generation

Generation is hermetic: its only inputs are files in this repository and Weaver at the pinned version.

- **No network.** The model reuses a few OpenTelemetry attributes and declares the OpenTelemetry registry as a
  dependency, which Weaver would otherwise clone. The definitions of exactly those attributes are vendored in
  [`vendor/opentelemetry`](vendor/opentelemetry), and the `generate-*` targets resolve the dependency from there. When
  Weaver runs from its container image, they run with `--network none`.
- **No language toolchain.** The templates emit code that already satisfies each language's formatter (ruff, gofmt,
  oxfmt, Spotless), so nothing is run after Weaver.
- **Deterministic.** The same model always produces the same files. CI regenerates everything with the network
  disabled and fails on any difference, so a generated file cannot be edited by hand or go stale. Each language's own
  CI then compiles and tests the result.

`model/manifest.yaml` still names the real OpenTelemetry registry, because that is what other registries resolve
when they depend on this one. `make check-vendor` (needs the network) fails if the vendored copy differs from
upstream. After changing the pinned OpenTelemetry version or referencing another OpenTelemetry attribute, run
`make vendor`.

### Spec tables

The tables in the prose spec are filled in by `make generate-spec`. A table is a pair of markers; Weaver replaces what
is between them:

```markdown
<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("llm.token_count."))]} -->
...generated table...
<!-- endweaver -->
```

The marker holds a [jq](https://jqlang.org/) query over the resolved registry. It must produce one of:

- `{attributes: [...]}` for an attribute table,
- `{members: [...]}` for a table of well-known values, or
- `{span: <type>, span_attributes: [...]}` for the requirement levels of a span's attributes, as in
  [`spec/spans/llm.md`](../spec/spans/llm.md). Its notes column shows the attribute's description on that span,
  which a span definition can override (`- ref: input.value` with its own `brief`).
- `{span_kinds: [...]}` for the table of span kinds and `{characteristic: [...]}` for the attributes that set one
  kind apart from a plain `CHAIN` span, both in [`spec/general/span-kinds.md`](../spec/general/span-kinds.md). The
  page a kind links to is the span's `annotations.openinference.spec`.

The rendering is in [`templates/registry/markdown/snippet.md.j2`](templates/registry/markdown/snippet.md.j2). Every table
shows each attribute's stability as a badge, as the OpenTelemetry semantic conventions do; a deprecated attribute
shows the deprecated badge. Most tables select a key
prefix, so a new attribute under that prefix appears on its own. Tables that list keys explicitly, to keep a curated
order, need the new key added to the marker. To add a table to any page under `spec/`, add a marker pair and run
`make generate-spec`.

### Language packages

`make generate-code` writes the constants of the four semantic convention packages from the model:

| Language | Template | Generated file |
| --- | --- | --- |
| Python | [`templates/registry/python`](templates/registry/python) | `trace/__init__.py` and `resource/__init__.py` in `python/openinference-semantic-conventions/src/openinference/semconv` |
| Go | [`templates/registry/go`](templates/registry/go) | `attributes.go`, `enums.go`, and `indexers.go` in `go/openinference-semantic-conventions` |
| TypeScript | [`templates/registry/typescript`](templates/registry/typescript) | `trace/SemanticConventions.ts` and `resource/SemanticResourceAttributes.ts` in `js/packages/openinference-semantic-conventions/src` |
| Java | [`templates/registry/java`](templates/registry/java) | `SemanticConventions.java` and `SemanticResourceAttributes.java` in `java/openinference-semantic-conventions/src/main/java/com/arize/semconv/trace` |

Constant names follow from the attribute key (`llm.model_name` becomes `LLM_MODEL_NAME`, or `LLMModelName` in Go).
Where a published name differs:

- A name that differs in every language is recorded once in the model, as
  `annotations.openinference.constant_key` (`tool_call.function.arguments` is named as if it were
  `tool_call.function.arguments_json`).
- A name that differs in one language is in that language's template (`legacy_names`), next to the other things
  that are specific to a package: which class an attribute belongs to, constants for key prefixes, enum members that
  are not in the registry, and the casing of Go identifiers.

Those tables are public API. Add to them; do not rename or remove entries. Enum members follow the model, so adding
a well-known value adds a member in every language.

A deprecated attribute keeps its constant, marked the way each language's tools recognize: `@Deprecated` and a
`@deprecated` Javadoc tag in Java, a `@deprecated` JSDoc tag in TypeScript, and a `Deprecated:` paragraph in Go.
Python has no marker for a class constant, so the docstring says it.

Helpers are derived too:

- **Go indexers.** Every flattened list that sits directly on a span gets `<List>Key(i, child)`, for example
  `RetrievalDocumentKey(0, DocumentContent)`. The fixed-field and nested helpers (`LLMInputMessageRoleKey`,
  `LLMOutputMessageToolCallKey`) are listed in the template by the model keys they join.
- **TypeScript and Java prefix objects.** `SemanticAttributePrefixes` is the set of root namespaces, and each
  `*Postfixes` object is what follows its namespace in the model's keys.

## Using the registry

Released versions are tagged `semantic-conventions-vX.Y.Z`. Each release has the publication manifest
(`manifest.yaml`) and the resolved registry (`registry.yaml`) attached as assets, and the docs site serves the same two
files at the registry's schema URL: <https://openinf.dev/schemas/> lists the versions. See [RELEASING.md](RELEASING.md).

To see them on a local build of the site, run `make -C semantic-conventions site-schemas` first.

Check telemetry against the conventions:

```bash
weaver registry live-check \
  -r 'https://github.com/Arize-ai/openinference.git@semantic-conventions-v0.0.1[semantic-conventions/model]'
```

Build another registry on top of OpenInference by declaring it as a dependency, then `ref` its attributes. This
also gives the dependent registry the OpenTelemetry attributes that OpenInference reuses, such as `session.id`.
[`tests/consumer`](tests/consumer) is a working example that `make test` checks.

```yaml
# manifest.yaml
file_format: definition_manifest/2.0
name: acme
schema_url: https://example.com/schemas/1.0.0
dependencies:
  - schema_url: https://openinf.dev/schemas/0.0.1
    registry_path: https://github.com/Arize-ai/openinference.git@semantic-conventions-v0.0.1[semantic-conventions/model]
```

```yaml
# spans.yaml
file_format: definition/2
spans:
  - type: acme.routed_llm
    kind: internal
    requirement_level: recommended
    stability: development
    brief: An LLM call made through the Acme router.
    name:
      note: Free form.
    attributes:
      - ref: openinference.span.kind
        requirement_level: required
      - ref: llm.model_name
        requirement_level: recommended
```

Depend on the tagged source, as above. The `manifest.yaml` and `registry.yaml` attached to each release are the
published form of the same registry, but Weaver cannot resolve them as a dependency yet: a published manifest names
its own dependencies by schema URL only, and OpenTelemetry does not serve a registry at
`https://opentelemetry.io/schemas/<version>`. The assets are still the right input for tools that read a resolved
registry directly.

## What is checked

| Check | What it catches |
| --- | --- |
| `make check` | A model that does not parse or resolve; violations of the shared OpenTelemetry policies (naming, collisions, stability, deprecation); violations of the OpenInference policies in [`policies/check/openinference`](policies/check/openinference); with a baseline, removed or incompatibly changed attributes and spans |
| `make check-vendor` | Vendored OpenTelemetry attributes that differ from the pinned upstream version |
| `make test` | A change to the model (or to Weaver) that stops flattened wire-format keys from resolving to their attribute, or that stops a registry depending on this one from resolving; an OpenInference policy rule that no longer fires |
| `make check-generated` | Docs, spec tables, or language constants not regenerated after a model change, or a generated file edited by hand |
| `make check-parity` | Model attributes that no table in `spec/attributes/semantic-conventions.md` shows, or that the Python template leaves out of every class |

Not checked yet: the leaves of a flattened list, and a changed type on a `development` attribute. See
[Modeling rules](#modeling-rules).

The remaining entries in `KNOWN_DRIFT` in [`scripts/check_parity.py`](scripts/check_parity.py) are the key-prefix
constants, which are constants in the packages but not attributes in the model.
