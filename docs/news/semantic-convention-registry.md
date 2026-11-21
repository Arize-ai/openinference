---
title: The spec now comes from a schema registry
description: The OpenInference conventions are now a machine-readable, versioned registry. The spec's tables and the language packages are generated from it, and we'd like your feedback on the schema.
pubDate: 2026-11-20
authors: [mikeldking]
tags: [spec, announcement]
releases: ["semantic-conventions-v0.0.1"]
---

Until now the OpenInference conventions lived in prose, and each language
package kept its own hand-written copy of the attribute names. They drifted.

The conventions are now a
[schema registry](../../spec/attributes/registry.md): one model that lists every
attribute, its type, its well-known values, and the attributes each span kind
carries. It's an [OpenTelemetry Weaver](https://github.com/open-telemetry/weaver)
registry, the same format the OpenTelemetry semantic conventions use.

Nothing changes on the wire. The attribute keys and values are the ones your
instrumentation already emits.

## What changed

- **The spec.** The attribute tables in the
  [specification](../../spec/README.md) are generated from the registry. The
  prose around them is still written by hand.
- **The packages.** The constants in the Python, Go, TypeScript and Java
  `openinference-semantic-conventions` packages are generated from the same
  model, so a new attribute lands in all four at once.
- **A schema you can pull.** Each version is served at its schema URL. The
  first one is [0.0.1](https://openinf.dev/schemas/0.0.1/), and
  [openinf.dev/schemas](https://openinf.dev/schemas/) lists them all.

## Pull it

A version is one YAML file with everything inlined:

```bash
curl -fsSLO https://openinf.dev/schemas/0.0.1/registry.yaml
```

Each attribute is an entry in it. Where OpenInference flattens a list into
indexed keys, the entry says so:

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

## Check your telemetry

With Weaver you can check attributes against the conventions. Put a few in a
file:

```json
[
  {"attribute": {"name": "openinference.span.kind", "value": "LLM"}},
  {"attribute": {"name": "llm.token_count.prompts", "value": 12}}
]
```

Then run:

```bash
weaver registry live-check \
  -r 'https://github.com/Arize-ai/openinference.git@semantic-conventions-v0.0.1[semantic-conventions/model]' \
  --input-source attributes.json --input-format json
```

It catches the typo:

```
llm.token_count.prompts = 12
    - [violation] Attribute 'llm.token_count.prompts' does not exist in the registry.
```

Weaver can also listen for OTLP, so you can point an application's exporter at
it, and you can define conventions of your own that build on OpenInference. The
[registry page](../../spec/attributes/registry.md) in the spec shows how.

## If you contribute

A new attribute now starts in the model under
[`semantic-conventions/model`](../../semantic-conventions/model), not in a
language package. Edit the YAML, run `make -C semantic-conventions generate`,
and the spec tables and all four packages follow. CI fails a pull request whose
generated files don't match the model.

## Tell us what you think

The registry is new and everything in it is marked `development`, which means
we can still change it. This is the time to say so if something looks wrong:

- an attribute whose type doesn't match what your instrumentation records
- a well-known value that's missing
- a shape the schema can't express, or that your tooling can't read

Open an [issue](https://github.com/Arize-ai/openinference/issues) and tell us.
