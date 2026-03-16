"""
CLI entry point: harbor-to-otel

Usage:
    harbor-to-otel trajectory.json -o traces.json
    harbor-to-otel trajectories_dir/ -o output_dir/
    harbor-to-otel trajectory.json --phoenix http://localhost:6006
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="harbor-to-otel",
        description="Convert Harbor ATIF trajectory JSON to OTLP JSON traces.",
    )
    parser.add_argument("input", help="Trajectory .json file or directory of .json files.")
    parser.add_argument("-o", "--output", help="Output path for OTLP JSON.")
    parser.add_argument("--phoenix", metavar="URL", help="Push traces to Phoenix at URL.")
    parser.add_argument(
        "--resource-attr",
        action="append",
        default=[],
        help="Resource attribute as key=value. Can be repeated.",
    )
    args = parser.parse_args(argv)
    input_path = Path(args.input)

    resource_attributes: dict[str, str] = {}
    for attr in args.resource_attr:
        if "=" not in attr:
            print(f"Invalid resource attribute (expected key=value): {attr}", file=sys.stderr)
            sys.exit(1)
        key, value = attr.split("=", 1)
        resource_attributes[key] = value

    from openinference.instrumentation.harbor._converter import (
        convert_trajectory_dir,
        convert_trajectory_file,
    )
    from openinference.instrumentation.harbor._file_exporter import export_spans_to_file

    res_attrs = resource_attributes or None
    if input_path.is_dir():
        spans = convert_trajectory_dir(input_path, resource_attributes=res_attrs)
    elif input_path.is_file():
        spans = convert_trajectory_file(input_path, resource_attributes=res_attrs)
    else:
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not spans:
        print("No spans produced from input.", file=sys.stderr)
        sys.exit(1)

    print(f"Converted {len(spans)} spans from {input_path}")

    if not args.output and not args.phoenix:
        print("Error: either --output or --phoenix is required.", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        export_spans_to_file(spans, output_path)
        print(f"Wrote {len(spans)} spans to {output_path}")

    if args.phoenix:
        from openinference.instrumentation.harbor._phoenix import phoenix_import_spans

        phoenix_import_spans(spans, endpoint=args.phoenix)
        print(f"Imported to Phoenix at {args.phoenix}")


if __name__ == "__main__":
    main()
