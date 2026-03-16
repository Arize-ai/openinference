"""
Post-hoc conversion of ATIF trajectory JSON to OTLP JSON file.

Usage:
    python post_hoc_convert.py path/to/trajectory.json
"""

from pathlib import Path

from openinference.instrumentation.harbor import convert_trajectory_file, export_spans_to_file


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python post_hoc_convert.py <trajectory.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = input_path.with_suffix(".otlp.json")

    # Convert ATIF trajectory to OTel spans
    spans = convert_trajectory_file(input_path)
    print(f"Converted {len(spans)} spans from {input_path}")

    # Export to OTLP JSON file
    export_spans_to_file(spans, output_path)
    print(f"Wrote OTLP JSON to {output_path}")


if __name__ == "__main__":
    main()
