#!/usr/bin/env python3
"""Vendor the OpenTelemetry attribute definitions that the model references.

The model reuses a few OpenTelemetry attributes (`session.id`, `user.id`) and
declares the OpenTelemetry registry as a dependency in `model/manifest.yaml`.
Resolving that dependency means cloning the OpenTelemetry repository, so
anything that resolves the model would need the network.

This script copies the definitions of exactly the referenced attributes, at
the pinned version, into `vendor/opentelemetry/`. The hermetic targets
(`make test`) resolve the dependency from that copy and run with no network
access. The manifest keeps pointing at the real registry, which is what
`make check` and other registries that depend on this one resolve against.

It asks Weaver for the definitions rather than parsing upstream files, so it
does not depend on how the OpenTelemetry repository lays out its model:

1. Find the keys the model references but does not define.
2. Resolve a throwaway registry that has the same dependency as the model and
   references those keys without overriding anything.
3. Write the resolved definitions as a small `definition/2` registry.

Usage (needs the network; `make vendor` supplies the weaver command):

    vendor_upstream.py --weaver "<weaver command>" [--check]

`--check` writes nothing and fails if the vendored copy differs from upstream.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model"
VENDOR = ROOT / "vendor" / "opentelemetry"
PROBE = ROOT / ".build" / "vendor-probe"

# Fields of an attribute definition, in the order they are written.
FIELDS = ("key", "type", "brief", "note", "examples", "stability", "deprecated")

HEADER = """\
# Vendored from the OpenTelemetry semantic conventions. Do not edit by hand.
#
# Source:  {registry_path}
# Schema:  {schema_url}
#
# Only the attributes that the OpenInference model references are copied, so
# that docs and code can be generated from the model without network access.
# Refresh with `make -C semantic-conventions vendor` after changing the pinned
# version in model/manifest.yaml or referencing another OpenTelemetry attribute.
"""


def refs(node: Any) -> Iterator[str]:
    """Yield every attribute key referenced with `ref`, wherever it appears.

    Spans and attribute groups list references under `attributes`, entities
    under `identity` and `description`; walking the document covers them all.
    """
    if isinstance(node, dict):
        if isinstance(node.get("ref"), str):
            yield node["ref"]
        for value in node.values():
            yield from refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from refs(value)


def load_model() -> tuple[dict[str, Any], list[str]]:
    """Return (the model's upstream dependency, keys it references but does not define)."""
    manifest = yaml.safe_load((MODEL / "manifest.yaml").read_text())
    dependencies = manifest.get("dependencies") or []
    if len(dependencies) != 1:
        sys.exit("vendor_upstream: expected exactly one dependency in model/manifest.yaml")
    defined: set[str] = set()
    referenced: set[str] = set()
    for path in sorted(MODEL.rglob("*.yaml")):
        if path.name == "manifest.yaml":
            continue
        document = yaml.safe_load(path.read_text()) or {}
        defined.update(a["key"] for a in document.get("attributes", []))
        referenced.update(refs(document))
    return dependencies[0], sorted(referenced - defined)


def resolve_upstream(weaver: list[str], dependency: dict[str, Any], keys: list[str]) -> list[dict[str, Any]]:
    """Ask Weaver for the upstream definitions of `keys`."""
    if PROBE.exists():
        shutil.rmtree(PROBE)
    PROBE.mkdir(parents=True)
    (PROBE / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "file_format": "definition_manifest/2.0",
                "name": "vendor-probe",
                "description": "Throwaway registry used to resolve upstream attribute definitions.",
                "schema_url": "https://openinf.dev/schemas/vendor-probe/0.0.0",
                "stability": "development",
                "dependencies": [dependency],
            },
            sort_keys=False,
        )
    )
    (PROBE / "probe.yaml").write_text(
        yaml.safe_dump(
            {
                "file_format": "definition/2",
                "spans": [
                    {
                        "type": "vendor.probe",
                        "kind": "internal",
                        "requirement_level": "recommended",
                        "stability": "development",
                        "brief": "References the upstream attributes to vendor.",
                        "name": {"note": "Not a real span."},
                        "attributes": [{"ref": key, "requirement_level": "recommended"} for key in keys],
                    }
                ],
            },
            sort_keys=False,
        )
    )
    probe = PROBE.relative_to(ROOT)
    result = subprocess.run(
        [*weaver, "registry", "resolve", "-r", str(probe), "--v2", "--quiet", "-f", "yaml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"vendor_upstream: weaver could not resolve the upstream registry:\n{result.stderr}")
    resolved = yaml.safe_load(result.stdout)
    probe_span = next(span for span in resolved["registry"]["spans"] if span["type"] == "vendor.probe")
    catalog = {a["key"]: a for a in probe_span["attributes"]}
    missing = [key for key in keys if key not in catalog]
    if missing:
        sys.exit(f"vendor_upstream: upstream does not define: {', '.join(missing)}")
    return [{field: catalog[key][field] for field in FIELDS if field in catalog[key]} for key in keys]


def render(dependency: dict[str, Any], attributes: list[dict[str, Any]]) -> dict[str, str]:
    header = HEADER.format(**dependency)
    manifest = {
        "file_format": "definition_manifest/2.0",
        "name": "opentelemetry-vendored",
        "description": "The OpenTelemetry attributes that the OpenInference model references.",
        "schema_url": dependency["schema_url"],
        "stability": "development",
    }
    registry = {"file_format": "definition/2", "attributes": attributes}
    return {
        "manifest.yaml": header + yaml.safe_dump(manifest, sort_keys=False, width=100),
        "registry.yaml": header + yaml.safe_dump(registry, sort_keys=False, width=100),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weaver", required=True, help="command that runs weaver from this directory")
    parser.add_argument("--check", action="store_true", help="fail if the vendored copy is out of date")
    args = parser.parse_args()

    dependency, keys = load_model()
    attributes = resolve_upstream(shlex.split(args.weaver), dependency, keys)
    files = render(dependency, attributes)

    if args.check:
        stale = [
            name for name, content in files.items() if not (VENDOR / name).exists() or (VENDOR / name).read_text() != content
        ]
        if stale:
            print(f"FAIL: {VENDOR.relative_to(ROOT)} differs from upstream in: {', '.join(stale)}")
            print("Run 'make -C semantic-conventions vendor' and commit.")
            return 1
        print(f"OK: {VENDOR.relative_to(ROOT)} matches upstream for {', '.join(keys)}.")
        return 0

    VENDOR.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (VENDOR / name).write_text(content)
    print(f"Vendored {', '.join(keys)} from {dependency['registry_path']} into {VENDOR.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
