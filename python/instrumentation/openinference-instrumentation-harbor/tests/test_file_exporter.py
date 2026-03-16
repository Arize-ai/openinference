import json
from pathlib import Path

from openinference.instrumentation.harbor._converter import convert_trajectory
from openinference.instrumentation.harbor._file_exporter import (
    OTLPJsonFileExporter,
    export_spans_to_file,
)
from openinference.semconv.trace import SpanAttributes


class TestOTLPJsonFileExporter:
    def test_writes_valid_json(self, sample_trajectory: dict, tmp_path: Path) -> None:
        """Exporter writes a valid JSON file with OTLP structure."""
        spans = convert_trajectory(sample_trajectory)
        exporter = OTLPJsonFileExporter(output_dir=str(tmp_path))

        from opentelemetry.sdk.trace.export import SpanExportResult

        result = exporter.export(spans)
        assert result == SpanExportResult.SUCCESS

        # Should have written exactly one file
        files = list(tmp_path.glob("harbor_trace_*.json"))
        assert len(files) == 1

        # File should be valid JSON with OTLP structure
        with open(files[0]) as f:
            data = json.load(f)
        assert "resource_spans" in data

    def test_otlp_structure(self, sample_trajectory: dict, tmp_path: Path) -> None:
        """OTLP JSON has correct nested structure: resource_spans → scope_spans → spans."""
        spans = convert_trajectory(sample_trajectory)
        exporter = OTLPJsonFileExporter(output_dir=str(tmp_path))
        exporter.export(spans)

        files = list(tmp_path.glob("*.json"))
        with open(files[0]) as f:
            data = json.load(f)

        resource_spans = data["resource_spans"]
        assert len(resource_spans) > 0

        # Navigate into the structure
        for rs in resource_spans:
            assert "scope_spans" in rs
            for ss in rs["scope_spans"]:
                assert "spans" in ss
                for span_data in ss["spans"]:
                    assert "name" in span_data
                    assert "trace_id" in span_data
                    assert "span_id" in span_data

    def test_attributes_survive_roundtrip(self, sample_trajectory: dict, tmp_path: Path) -> None:
        """Span attributes are preserved through serialization."""
        spans = convert_trajectory(sample_trajectory)
        exporter = OTLPJsonFileExporter(output_dir=str(tmp_path))
        exporter.export(spans)

        files = list(tmp_path.glob("*.json"))
        with open(files[0]) as f:
            data = json.load(f)

        # Find the root span and check its attributes
        all_spans = []
        for rs in data["resource_spans"]:
            for ss in rs["scope_spans"]:
                all_spans.extend(ss["spans"])

        root_span = next(s for s in all_spans if "trajectory" in s["name"])
        attr_map: dict[str, str] = {}
        for attr in root_span.get("attributes", []):
            key = attr["key"]
            val = attr["value"]
            if "string_value" in val:
                attr_map[key] = val["string_value"]
            elif "int_value" in val:
                attr_map[key] = val["int_value"]

        assert attr_map[SpanAttributes.AGENT_NAME] == "research_agent"
        assert attr_map[SpanAttributes.SESSION_ID] == "test-session-abc123"

    def test_custom_prefix(self, sample_trajectory: dict, tmp_path: Path) -> None:
        """Custom file prefix is used in the output filename."""
        spans = convert_trajectory(sample_trajectory)
        exporter = OTLPJsonFileExporter(output_dir=str(tmp_path), file_prefix="custom")
        exporter.export(spans)

        files = list(tmp_path.glob("custom_*.json"))
        assert len(files) == 1

    def test_empty_spans(self, tmp_path: Path) -> None:
        """Exporting empty spans returns SUCCESS without writing a file."""
        exporter = OTLPJsonFileExporter(output_dir=str(tmp_path))

        from opentelemetry.sdk.trace.export import SpanExportResult

        result = exporter.export([])
        assert result == SpanExportResult.SUCCESS
        assert len(list(tmp_path.glob("*.json"))) == 0


class TestExportSpansToFile:
    def test_convenience_function(self, sample_trajectory: dict, tmp_path: Path) -> None:
        """export_spans_to_file writes to the exact path specified."""
        spans = convert_trajectory(sample_trajectory)
        output = tmp_path / "output.json"
        result_path = export_spans_to_file(spans, output)

        assert result_path == output
        assert output.exists()

        with open(output) as f:
            data = json.load(f)
        assert "resource_spans" in data
