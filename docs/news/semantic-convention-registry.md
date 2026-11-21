---
title: The spec now comes from a schema registry
description: The OpenInference conventions are now a machine-readable registry. The spec's tables and the language packages are generated from it, and we'd like your feedback on the schema.
pubDate: 2026-11-20
authors: [mikeldking]
tags: [spec, announcement]
---

Until now the OpenInference conventions lived in prose, and each language
package kept its own hand-written copy of the attribute names. They drifted.

The conventions are now a
[schema registry](../../spec/attributes/registry.md): one model that lists every
attribute, its type, its well-known values, and the attributes each span kind
carries. It's an [OpenTelemetry Weaver](https://github.com/open-telemetry/weaver)
registry, the same format the OpenTelemetry semantic conventions use.

## What changed

- **The spec.** The attribute tables in the
  [specification](../../spec/README.md) are generated from the registry. The
  prose around them is still written by hand.
- **The packages.** The constants in the Python, Go, TypeScript and Java
  `openinference-semantic-conventions` packages are generated from the same
  model, so a new attribute lands in all four at once.
- **A schema you can pull.** Each version is served at its schema URL, listed
  at [openinf.dev/schemas](https://openinf.dev/schemas/).

Nothing changes on the wire. The attribute keys and values are the ones your
instrumentation already emits.

## Try it

Download a version, replacing `X.Y.Z` with one from the list:

```bash
curl -fsSLO https://openinf.dev/schemas/X.Y.Z/registry.yaml
```

With Weaver you can also check your own telemetry against the conventions, or
define conventions of your own that build on OpenInference. The
[registry page](../../spec/attributes/registry.md) in the spec shows how.

## Tell us what you think

The registry is new and everything in it is marked `development`, which means
we can still change it. This is the time to say so if something looks wrong:

- an attribute whose type doesn't match what your instrumentation records
- a well-known value that's missing
- a shape the schema can't express, or that your tooling can't read

Open an [issue](https://github.com/Arize-ai/openinference/issues) and tell us.
