# OpenInference Semantic Conventions Registry

The machine-readable definition of the OpenInference semantic conventions: every attribute, its type and well-known
values, and which attributes a span of each kind carries.

It is written in the same format as the OpenTelemetry semantic conventions (the
[Weaver](https://github.com/open-telemetry/weaver) `definition/2` schema) and validated with the same tool and shared
policies as [`semantic-conventions-genai`](https://github.com/open-telemetry/semantic-conventions-genai). That makes the
conventions versioned, checkable, and consumable by other registries.

The prose specification is in [`spec/`](../spec/README.md). The model here is the source of truth for attribute keys,
types, and well-known values; the spec explains them.

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
├── vendor/opentelemetry/   # The OpenTelemetry attributes the model reuses, vendored for offline use
├── tests/live-check/       # Sample wire-format keys that must / must not resolve against the registry
├── tests/policies/         # A registry that breaks each OpenInference policy, to prove that the rules fire
├── tests/consumer/         # A registry that depends on this one, to prove that others can
├── scripts/                # Checks run by the Makefile
├── versions.env            # Pinned Weaver and policy versions
└── Makefile
```

Each `registry.yaml` defines attributes. Each `spans.yaml` defines the span for an `openinference.span.kind` by
referencing those attributes with a requirement level.

## Working on the model

You need either [Weaver](https://github.com/open-telemetry/weaver/releases) at the version pinned in
[`versions.env`](versions.env), or Docker/Podman (the Makefile then runs Weaver from its container image). The checks
that run Python scripts with dependencies also need [uv](https://github.com/astral-sh/uv).

```bash
make -C semantic-conventions check-all       # everything CI runs
make -C semantic-conventions check           # validate the model and run the policies
make -C semantic-conventions test            # check sample keys against the registry
make -C semantic-conventions help            # all targets
```

To change the conventions:

1. Edit the YAML under `model/`. Add new attributes to the `registry.yaml` of their domain, and reference them from the
   `spans.yaml` of every span kind that carries them.
2. Update the prose specification in [`spec/`](../spec/README.md) and the language packages to match.
3. Use a conventional commit scoped to this directory so the change lands in the registry changelog, e.g.
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
- **Naming exceptions.** `metadata`, `annotations`, and `evaluations` have no namespace, which the shared naming policy
  rejects. They predate the registry and cannot be renamed, so they are listed in
  [`policies/exceptions.json`](policies/exceptions.json). Do not add to that list for new attributes.

### Offline checks

The model reuses a few OpenTelemetry attributes and declares the OpenTelemetry registry as a dependency, which
Weaver would otherwise clone. The definitions of exactly those attributes are vendored in
[`vendor/opentelemetry`](vendor/opentelemetry), and `make test` resolves the dependency from there, with no network
when Weaver runs from its container image.

`model/manifest.yaml` still names the real OpenTelemetry registry, because that is what other registries resolve
when they depend on this one. `make check-vendor` (needs the network) fails if the vendored copy differs from
upstream. After changing the pinned OpenTelemetry version or referencing another OpenTelemetry attribute, run
`make vendor`.

## Using the registry

Released versions are tagged `semantic-conventions-vX.Y.Z`. Each release has the publication manifest
(`manifest.yaml`) and the resolved registry (`registry.yaml`) attached as assets. See [RELEASING.md](RELEASING.md).

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

Not checked yet: the leaves of a flattened list, and a changed type on a `development` attribute. See
[Modeling rules](#modeling-rules).
