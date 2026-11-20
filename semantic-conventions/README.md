# OpenInference Semantic Conventions Registry

The machine-readable definition of the OpenInference semantic conventions: every attribute, its type and well-known
values, and which attributes a span of each kind carries.

It is written in the same format as the OpenTelemetry semantic conventions (the
[Weaver](https://github.com/open-telemetry/weaver) `definition/2` schema) and validated with the same tool and shared
policies as [`semantic-conventions-genai`](https://github.com/open-telemetry/semantic-conventions-genai). That makes the
conventions versioned, checkable, and consumable by other registries.

The prose specification is in [`spec/`](../spec/README.md). The model here is the source of truth for attribute keys,
types, and well-known values; the spec explains them, and its attribute tables are generated from the model.

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
├── templates/registry/     # Weaver templates for the generated reference
├── docs/registry/          # GENERATED reference (do not edit by hand)
├── tests/live-check/       # Sample wire-format keys that must / must not resolve against the registry
├── scripts/                # Checks run by the Makefile
├── versions.env            # Pinned Weaver and policy versions
└── Makefile
```

Each `registry.yaml` defines attributes. Each `spans.yaml` defines the span for an `openinference.span.kind` by
referencing those attributes with a requirement level. Browse the result in [`docs/registry/`](docs/registry/README.md).

## Working on the model

You need either [Weaver](https://github.com/open-telemetry/weaver/releases) at the version pinned in
[`versions.env`](versions.env), or Docker/Podman (the Makefile then runs Weaver from its container image). The parity
check also needs [uv](https://github.com/astral-sh/uv).

```bash
make -C semantic-conventions check-all       # everything CI runs
make -C semantic-conventions check           # validate the model and run the policies
make -C semantic-conventions test            # check sample keys against the registry
make -C semantic-conventions generate        # regenerate docs/registry/ and the spec tables after changing the model
make -C semantic-conventions check-parity    # compare the model with spec/ and the Python package
make -C semantic-conventions help            # all targets
```

To change the conventions:

1. Edit the YAML under `model/`. Add new attributes to the `registry.yaml` of their domain, and reference them from the
   `spans.yaml` of every span kind that carries them.
2. Run `make generate` and commit the result: the reference in `docs/registry/` and the tables in
   [`spec/attributes/semantic-conventions.md`](../spec/attributes/semantic-conventions.md).
3. Update the prose in [`spec/`](../spec/README.md) where the change needs explaining, and the language packages.
   `make check-parity` fails until the Python package agrees with the model, and if a new attribute is not picked up
   by any table in the spec (see [Spec tables](#spec-tables)).
4. Use a conventional commit scoped to this directory so the change lands in the registry changelog, e.g.
   `feat(semantic-conventions): add llm.reasoning_effort`.

### Modeling rules

- **Stability.** Everything is `development` for now. Marking an attribute `stable` is a commitment that the shared
  policies then enforce.
- **Never remove or rename an attribute.** Deprecate it with `deprecated:` and point to the replacement. On pull
  requests, CI compares the model with the base branch and fails on removals and incompatible changes.
- **Reuse OpenTelemetry attributes** when one exists for the same concept. `session.id` and `user.id` are referenced
  from the OpenTelemetry registry (a dependency in [`model/manifest.yaml`](model/manifest.yaml)), not redefined.
- **Well-known values** are enum `members`. Enums are open: a custom value may be used when no member applies.
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
  `llm.input_messages` attribute; `item_namespaces` says its items are `message.*` attributes.
- **Types Weaver cannot express.** When instrumentations record more than one primitive type for a key, give the
  attribute its main type and list the others in `annotations.openinference.also_accepts`, e.g. `[int]` on
  `document.id`.
- **Naming exceptions.** `metadata`, `annotations`, and `evaluations` have no namespace, which the shared naming policy
  rejects. They predate the registry and cannot be renamed, so they are listed in
  [`policies/exceptions.json`](policies/exceptions.json). Do not add to that list for new attributes.

### Spec tables

The tables in the prose spec are filled in by `make generate-spec`. A table is a pair of markers; Weaver replaces what
is between them:

```markdown
<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("llm.token_count."))]} -->
...generated table...
<!-- endweaver -->
```

The marker holds a [jq](https://jqlang.org/) query over the resolved registry. It must produce either
`{attributes: [...]}` for an attribute table or `{members: [...]}` for a table of well-known values; the rendering is
in [`templates/registry/markdown/snippet.md.j2`](templates/registry/markdown/snippet.md.j2). Most tables select a key
prefix, so a new attribute under that prefix appears on its own. Tables that list keys explicitly, to keep a curated
order, need the new key added to the marker. To add a table to any page under `spec/`, add a marker pair and run
`make generate-spec`.

## Using the registry

Released versions are tagged `semantic-conventions-vX.Y.Z`. Each release has the publication manifest
(`manifest.yaml`) and the resolved registry (`registry.yaml`) attached as assets. See [RELEASING.md](RELEASING.md).

Check telemetry against the conventions:

```bash
weaver registry live-check \
  -r 'https://github.com/Arize-ai/openinference.git@semantic-conventions-v0.1.0[semantic-conventions/model]'
```

Build another registry on top of OpenInference by declaring it as a dependency, then `ref` its attributes:

```yaml
# manifest.yaml
file_format: definition_manifest/2.0
name: acme
schema_url: https://example.com/schemas/1.0.0
dependencies:
  - schema_url: https://openinf.dev/schemas/0.1.0
    registry_path: https://github.com/Arize-ai/openinference.git@semantic-conventions-v0.1.0[semantic-conventions/model]
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

## What is checked

| Check | What it catches |
| --- | --- |
| `make check` | A model that does not parse or resolve; violations of the shared OpenTelemetry policies (naming, collisions, stability, deprecation); violations of the OpenInference policies in [`policies/check/openinference`](policies/check/openinference); with a baseline, removed or incompatibly changed attributes and spans |
| `make test` | A change to the model (or to Weaver) that stops flattened wire-format keys from resolving to their attribute |
| `make check-docs` | `docs/registry/` or the tables in `spec/` not regenerated after a model change, or a generated table edited by hand |
| `make check-parity` | Attribute keys or well-known values that differ between the model and the Python package; model attributes that no table in `spec/attributes/semantic-conventions.md` shows |

The differences that exist today are listed in `KNOWN_DRIFT` in [`scripts/check_parity.py`](scripts/check_parity.py).
The JavaScript and Java packages are not compared yet: they compose keys from prefixes, so they need to be generated
from the model rather than parsed. The Go package is pinned to the Python one by `scripts/check_go_semconv.py`.
