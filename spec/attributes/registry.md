# Semantic Convention Registry

The conventions in this specification are also published as a machine-readable registry: every attribute key, its
type, its well-known values, and the attributes each span kind carries. The registry is an
[OpenTelemetry Weaver](https://github.com/open-telemetry/weaver) registry, and it is the source of truth. The
attribute tables in this specification and the constants in the Python, Go, TypeScript, and Java
`openinference-semantic-conventions` packages are generated from it.

Use the registry when a tool needs the conventions as data: to check that emitted telemetry follows them, to generate
code, or to define conventions that build on OpenInference.

## Versions

The registry is versioned on its own, separately from the language packages. Each version is identified by its
schema URL:

```
https://openinf.dev/schemas/X.Y.Z
```

<https://openinf.dev/schemas/> lists the versions. A released version does not change. Everything in the registry is
at `development` stability for now, so a new version may change an attribute's type; attributes are deprecated, not
removed or renamed.

## Getting the registry

A version is available in three forms.

| Form | Where | Use it for |
| --- | --- | --- |
| Resolved registry | `https://openinf.dev/schemas/X.Y.Z/registry.yaml` | Tools that read one self-contained file |
| Publication manifest | `https://openinf.dev/schemas/X.Y.Z/manifest.yaml` | The registry's name, version, and where the resolved registry is |
| Source model | The `semantic-conventions/model` directory at the Git tag `semantic-conventions-vX.Y.Z` | Weaver: checking telemetry and depending on the registry |

Download the resolved registry and its manifest from the schema URL:

```bash
curl -fsSLO https://openinf.dev/schemas/X.Y.Z/registry.yaml
curl -fsSLO https://openinf.dev/schemas/X.Y.Z/manifest.yaml
```

The same two files are attached to the GitHub release of each version:

```bash
gh release download semantic-conventions-vX.Y.Z --repo Arize-ai/openinference \
  --pattern registry.yaml --pattern manifest.yaml
```

## Checking telemetry

Weaver's `live-check` compares attributes and spans with the registry and reports the ones it does not define. Point
it at the source model of a version:

```bash
weaver registry live-check \
  -r 'https://github.com/Arize-ai/openinference.git@semantic-conventions-vX.Y.Z[semantic-conventions/model]'
```

Flattened lists are checked by their prefix only: `llm.input_messages.0.message.role` is accepted because
`llm.input_messages` is in the registry, and so is a key with a misspelled leaf.

## Building on the registry

A registry that defines its own conventions can depend on OpenInference and refer to its attributes. Declare the
dependency in the manifest:

```yaml
# manifest.yaml
file_format: definition_manifest/2.0
name: acme
schema_url: https://example.com/schemas/1.0.0
dependencies:
  - schema_url: https://openinf.dev/schemas/X.Y.Z
    registry_path: https://github.com/Arize-ai/openinference.git@semantic-conventions-vX.Y.Z[semantic-conventions/model]
```

Then refer to an attribute with `ref`:

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

Depend on the tagged source model, not on the resolved registry: Weaver cannot yet resolve a dependency from the
published files.

## Further reading

The [registry's README](https://github.com/Arize-ai/openinference/tree/main/semantic-conventions) describes the model
layout, the annotations OpenInference adds for flattened lists and JSON strings, and how to change the conventions.
