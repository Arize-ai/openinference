# Semantic Convention Registry

The conventions in this specification are also published as a machine-readable registry: every attribute key, its
type, its well-known values, and the attributes each span kind carries. The registry is an
[OpenTelemetry Weaver](https://github.com/open-telemetry/weaver) registry, the format the OpenTelemetry semantic
conventions use, and it is the source of truth. The attribute tables in this specification and the constants in the
Python, Go, TypeScript, and Java `openinference-semantic-conventions` packages are generated from it.

Use the registry when a tool needs the conventions as data: to check that emitted telemetry follows them, to generate
code, or to define conventions that build on OpenInference.

The registry describes what instrumentation already emits. It adds no attribute and changes no key or value.

## Versions

The registry is versioned on its own, separately from the language packages. Each version is identified by its
schema URL, which is also where the version is served:

```
https://openinf.dev/schemas/0.0.1
```

`0.0.1` is the first version, and the examples on this page use it. <https://openinf.dev/schemas/> lists every
version. A released version does not change.

Everything in the registry is at `development` stability for now, so a new version may change an attribute's type.
Attributes are deprecated, not removed or renamed.

## Getting the registry

A version is available in three forms.

| Form | Where | Use it for |
| --- | --- | --- |
| Resolved registry | `https://openinf.dev/schemas/0.0.1/registry.yaml` | Tools that read one self-contained file |
| Publication manifest | `https://openinf.dev/schemas/0.0.1/manifest.yaml` | The registry's name, version, and where the resolved registry is |
| Source model | The `semantic-conventions/model` directory at the Git tag `semantic-conventions-v0.0.1` | Weaver: checking telemetry and depending on the registry |

Download the resolved registry and its manifest from the schema URL:

```bash
curl -fsSLO https://openinf.dev/schemas/0.0.1/registry.yaml
curl -fsSLO https://openinf.dev/schemas/0.0.1/manifest.yaml
```

The same two files are attached to the GitHub release of each version:

```bash
gh release download semantic-conventions-v0.0.1 --repo Arize-ai/openinference \
  --pattern registry.yaml --pattern manifest.yaml
```

## What is in it

The resolved registry is one YAML file with every reference and dependency inlined. Two parts of it matter to most
readers:

- `attribute_catalog` lists every attribute once: its key, type, brief, examples, and stability.
- `registry.spans` lists one span per span kind (`openinference.llm`, `openinference.tool`, and so on). Each of its
  attributes points into the catalog by position (`base`) and adds a requirement level.

An attribute with well-known values lists them as the `members` of its type:

```yaml
- key: annotation.annotator_kind
  type:
    members:
      - id: human
        value: HUMAN
        brief: A human reviewer.
        stability: development
      - id: llm
        value: LLM
        brief: An LLM judge.
        stability: development
  brief: "Judge kind: `HUMAN`, `LLM`, `CODE`, or custom."
  stability: development
```

The well-known values are not exhaustive: a custom value may be recorded when no member applies.

### OpenInference annotations

OTLP span attributes cannot hold objects, so OpenInference flattens lists and objects into indexed keys or serializes
them to JSON (see [Attribute Conventions](./conventions.md)). Weaver has no type for either, so the registry records
them under `annotations.openinference`:

```yaml
- key: llm.input_messages
  type: template[any]
  brief: List of messages sent to the LLM in a chat API request, e.g. `llm.input_messages.0.message.role`.
  stability: development
  annotations:
    openinference:
      encoding: flattened_list
      item_namespaces:
        - message
```

| Annotation | Meaning |
| --- | --- |
| `encoding` | `flattened_list` or `flattened_object`: the key is a prefix, followed by an index or a field name. `json`: the string value holds JSON. |
| `item_namespaces` | The namespaces of the attributes that each item of a flattened value carries, e.g. `message` for `message.*`. |
| `item_attributes` | The attributes of each item, when only part of a namespace applies. |
| `also_accepts` | Other primitive types instrumentation records for the key, e.g. `int` on `document.id`. |
| `constant_key` | The key that the name of the language packages' constant derives from, where it differs from the attribute's key. |

### Reading it in code

The file is plain YAML, so no Weaver is needed to read it. This lists the attributes of an LLM span with their
requirement levels:

```python
import urllib.request

import yaml

url = "https://openinf.dev/schemas/0.0.1/registry.yaml"
registry = yaml.safe_load(urllib.request.urlopen(url))
catalog = registry["attribute_catalog"]

llm = next(span for span in registry["registry"]["spans"] if span["type"] == "openinference.llm")
for attribute in llm["attributes"]:
    print(catalog[attribute["base"]]["key"], attribute["requirement_level"])
```

## Checking telemetry

Weaver's `live-check` compares attributes and spans with the registry and reports the ones it does not define. Point
it at the source model of a version and give it the telemetry to check. With a file of attributes:

```json
[
  {"attribute": {"name": "openinference.span.kind", "value": "LLM"}},
  {"attribute": {"name": "llm.input_messages.0.message.role", "value": "user"}},
  {"attribute": {"name": "llm.token_count.prompts", "value": 12}}
]
```

```bash
weaver registry live-check \
  -r 'https://github.com/Arize-ai/openinference.git@semantic-conventions-v0.0.1[semantic-conventions/model]' \
  --input-source attributes.json --input-format json
```

The report has an entry for each attribute. The misspelled `llm.token_count.prompts` is a violation, and the command
exits with an error:

```
llm.input_messages.0.message.role = user
    - [information] Attribute 'llm.input_messages.0.message.role' is a template
    - [improvement] Attribute 'llm.input_messages.0.message.role' is not stable; stability = development.

llm.token_count.prompts = 12
    - [violation] Attribute 'llm.token_count.prompts' does not exist in the registry.
```

Every attribute is reported as not stable for now, because the whole registry is at `development` stability.

Without `--input-source`, `live-check` listens for OTLP over gRPC, so an application can export its traces to it
directly. Set the port with `--otlp-grpc-port`, and how long it waits for telemetry with `--inactivity-timeout`.

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
  - schema_url: https://openinf.dev/schemas/0.0.1
    registry_path: https://github.com/Arize-ai/openinference.git@semantic-conventions-v0.0.1[semantic-conventions/model]
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

## Changing the conventions

A change to an attribute starts in the model, not in this specification or in a language package:

1. Edit the YAML under [`semantic-conventions/model`](../../semantic-conventions/model).
2. Run `make -C semantic-conventions generate`. It rewrites the attribute tables in this specification and the
   constants in the four language packages.
3. Update the prose in this specification where the change needs explaining.

CI fails a pull request whose generated files do not match the model, and one that removes an attribute. The
[registry's README](../../semantic-conventions/README.md) has the model layout and the modeling rules.

## Feedback

The registry is new. If an attribute's type does not match what your instrumentation records, a well-known value is
missing, or the schema cannot express something you need, open an
[issue](https://github.com/Arize-ai/openinference/issues). While everything is at `development` stability, the
schema can still change.

## Further reading

- The [registry's README](../../semantic-conventions/README.md): the model layout, the modeling rules, and the checks
  CI runs.
- The [generated reference](../../semantic-conventions/docs/registry/README.md): every span and attribute namespace,
  rendered from the model.
- [Releasing the registry](../../semantic-conventions/RELEASING.md): how a version is cut and published.
