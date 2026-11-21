---
title: OpenInference Semantic Conventions as a Weaver Registry
description: The OpenInference semantic conventions are now defined in an OpenTelemetry Weaver registry, validated with the OpenTelemetry policies and published at a schema URL. The spec's tables and the language packages are generated from it.
pubDate: 2026-11-21
authors: [mikeldking]
tags: [spec, opentelemetry, announcement]
releases: ["semantic-conventions-v0.0.1"]
---

The OpenInference semantic conventions are now defined in a
[schema registry](../../spec/attributes/registry.md) built with
[OpenTelemetry Weaver](https://github.com/open-telemetry/weaver), the toolchain
the OpenTelemetry semantic conventions are authored and validated with. The
registry is the source of truth: the attribute tables in the
[specification](../../spec/README.md) and the constants in the Python,
TypeScript, Java and Go `openinference-semantic-conventions` packages are
generated from it.

## Weaver

OpenInference exists to make AI applications observable now and to bring what
works to OpenTelemetry. Until now that second half meant translating by hand:
the conventions were markdown and four hand-maintained sets of constants, in no
format that OpenTelemetry tooling could read.

Defining them in the same format as OpenTelemetry removes the translation:

- **Same schema.** The model is YAML in Weaver's
`definition/2` [syntax](https://github.com/open-telemetry/weaver/blob/main/schemas/semconv-syntax.v2.md): attributes with a
type, stability, examples and enum members, and a span definition for each
`openinference.span.kind` except `UNKNOWN` that references those attributes
with a requirement level.
- **Same policies.** CI runs
[`weaver registry check`](https://github.com/open-telemetry/weaver/blob/main/docs/usage.md) with the shared
[OpenTelemetry policy package](https://github.com/open-telemetry/opentelemetry-weaver-packages)
that
[`semantic-conventions-genai`](https://github.com/open-telemetry/semantic-conventions-genai)
uses, pinned to the same revision: naming conventions, stability and
backwards compatibility. Three legacy keys (`metadata`, `annotations`,
`evaluations`) have no namespace and are listed as explicit exceptions, each
with its reason.
- **Shared attributes.** The registry declares the OpenTelemetry semantic
conventions (1.45.0) as a [dependency](https://github.com/open-telemetry/weaver/blob/main/docs/dependencies.md) and references `session.id` and
`user.id` from it instead of redefining them.
- **Backwards compatibility.** On every pull request the shared policy compares
the model with the one on the base branch. Removing an attribute or a span
fails the build; attributes are deprecated, not removed or renamed.
- **Published at a schema URL.** Each version is a resolved registry and a
publication manifest, the two files Weaver packages, served at the version's
schema URL. The first is
[`https://openinf.dev/schemas/0.0.1`](https://openinf.dev/schemas/0.0.1/).

With both sets of conventions in one format, an OpenInference attribute and a
`gen_ai.*` attribute can be compared field by field, and a convention proposed
upstream starts from a definition OpenTelemetry's tooling already accepts.

## Complex attributes

OTLP span attributes cannot hold objects, so OpenInference flattens lists into
indexed keys (`llm.input_messages.0.message.role`) or serializes them to JSON.
OTel doesn't have primitives for this. Because of this, the registry models a flattened value as
`template[any]`, which is what lets Weaver resolve an indexed key to its
attribute, and records the encoding in an annotation:

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

OpenInference-specific Rego policies check those annotations. The limit today:
Weaver matches the prefix only, so a misspelled leaf after `llm.input_messages.0.`
still resolves. Validating the leaves is tracked in
[#4419](https://github.com/Arize-ai/openinference/issues/4419).

## Telemetry compliance

[`weaver registry live-check`](https://github.com/open-telemetry/weaver/tree/main/crates/weaver_live_check)
resolves the registry and reports every attribute it does not define. Given a file of attributes:

```json
[
  {"attribute": {"name": "openinference.span.kind", "value": "LLM"}},
  {"attribute": {"name": "llm.token_count.prompts", "value": 12}}
]
```

```bash
weaver registry live-check \
  -r 'https://github.com/Arize-ai/openinference.git@semantic-conventions-v0.0.1[semantic-conventions/model]' \
  --input-source attributes.json --input-format json
```

the misspelled key is a violation and the command exits non-zero:

```
llm.token_count.prompts = 12
    - [violation] Attribute 'llm.token_count.prompts' does not exist in the registry.
```

Without `--input-source`, `live-check` listens for OTLP over gRPC, so an
application's exporter can point at it directly. The repository's own CI runs
the same check on sample keys that must and must not resolve.

## Consuming the registry

The resolved registry is one YAML file with every reference and dependency
inlined:

```bash
curl -fsSLO https://openinf.dev/schemas/0.0.1/registry.yaml
```

To define conventions on top of OpenInference, declare it as a dependency in a
Weaver manifest and `ref` its attributes; the dependent registry also inherits
the OpenTelemetry attributes that OpenInference references. The
[registry page](../../spec/attributes/registry.md) in the spec covers the file
layout, the annotations, and the manifest.

## Changing the conventions

A new attribute now starts in the model under
[`semantic-conventions/model`](../../semantic-conventions/model), not in a
language package. `make -C semantic-conventions generate` renders the spec
tables and the four packages from
[Weaver templates](https://github.com/open-telemetry/weaver/blob/main/docs/codegen.md), and CI fails a pull request
whose generated files do not match the model.

## Feedback

Everything in the registry is at `development` stability, so the schema can
still change. We want to hear about:

- a missing well-known value
- a structure the annotations cannot express, or that your tooling cannot read
- anything that would make the conventions easier to align with OpenTelemetry

Open an [issue](https://github.com/Arize-ai/openinference/issues).
