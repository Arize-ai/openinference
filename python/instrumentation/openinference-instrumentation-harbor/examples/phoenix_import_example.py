"""
Import OTLP JSON trace files into Phoenix for visualization.

Usage:
    # First, start Phoenix:
    #   phoenix serve
    #
    # Then run:
    python phoenix_import_example.py ./traces/
"""

from pathlib import Path

from openinference.instrumentation.harbor import phoenix_import


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python phoenix_import_example.py <traces_dir_or_file>")
        sys.exit(1)

    source = Path(sys.argv[1])
    endpoint = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:6006"

    phoenix_import(source, endpoint=endpoint)
    print(f"Imported traces from {source} to Phoenix at {endpoint}")


if __name__ == "__main__":
    main()
