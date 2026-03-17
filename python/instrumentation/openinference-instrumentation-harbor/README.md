# OpenInference Harbor Instrumentation

OpenInference instrumentation for [Harbor](https://github.com/laude-institute/harbor), an agent evaluation framework that runs agents in sandboxed Docker containers.

Harbor produces [ATIF (Agent Trajectory Interchange Format)](https://harborframework.com/docs/agents/trajectory-format) JSON files containing every agent step, tool call, and token usage metric. This package converts those trajectories into OpenInference/OpenTelemetry spans so you can visualize and analyze agent evaluations in [Phoenix](https://github.com/Arize-ai/phoenix).

## Why this exists

Harbor sandboxes typically lack network access, so you can't export traces via OTLP HTTP during evaluation. Instead, this package converts ATIF trajectory files *after* the evaluation completes — either to OTLP JSON files on disk or directly into Phoenix.

## Installation

```bash
pip install openinference-instrumentation-harbor
```

## Quick start

After a Harbor evaluation run, trajectory files are saved at `jobs/<job-id>/<trial>/agent/trajectory.json`. Convert them and push to Phoenix:

```bash
harbor-to-otel jobs/*/agent/trajectory.json --phoenix http://localhost:6006
```

Or convert to an OTLP JSON file for later import:

```bash
harbor-to-otel jobs/*/agent/trajectory.json -o traces.json
```

## Usage

### CLI

The `harbor-to-otel` command converts ATIF trajectory files and optionally imports them into Phoenix.

```bash
# Convert a single trajectory to OTLP JSON
harbor-to-otel trajectory.json -o traces.json

# Convert all trajectories in a directory
harbor-to-otel jobs/ -o output_dir/

# Convert and push directly to Phoenix
harbor-to-otel trajectory.json --phoenix http://localhost:6006

# Both at once
harbor-to-otel trajectory.json -o traces.json --phoenix http://localhost:6006

# Add resource attributes
harbor-to-otel trajectory.json -o traces.json --resource-attr service.name=harbor-eval
```

### Python API

```python
from openinference.instrumentation.harbor import (
    convert_trajectory_file,
    export_spans_to_file,
    phoenix_import_spans,
)

# Convert ATIF trajectory to OTel spans
spans = convert_trajectory_file("trajectory.json")

# Export to OTLP JSON file
export_spans_to_file(spans, "traces.json")

# Or push directly to Phoenix
phoenix_import_spans(spans, endpoint="http://localhost:6006")
```

### Batch processing

```python
from pathlib import Path
from openinference.instrumentation.harbor import convert_trajectory_file, phoenix_import_spans

for trajectory in Path("jobs").rglob("trajectory.json"):
    spans = convert_trajectory_file(trajectory)
    phoenix_import_spans(spans, endpoint="http://localhost:6006")
```

### Import previously exported files

If you exported OTLP JSON files (e.g., from inside a sandbox), import them later:

```python
from openinference.instrumentation.harbor import phoenix_import

# Single file or directory of files
phoenix_import("traces.json", endpoint="http://localhost:6006")
phoenix_import("traces_dir/", endpoint="http://localhost:6006")
```

## What the traces look like

Each ATIF trajectory becomes a trace with this span hierarchy:

```
claude-code trajectory (AGENT)     ← root span with session ID, total tokens
├── step 1 (LLM)                   ← first LLM call with input/output, per-call tokens
├── step 2 (LLM)                   ← second LLM call (context grows)
│   └── ToolSearch (TOOL)           ← tool invocation with parameters and result
├── step 3 (LLM)
│   └── Write (TOOL)
└── step 4 (LLM)                   ← final response
```

- Empty ATIF steps (no message, no tool calls) are skipped
- Each LLM span has its own `llm.token_count.prompt` / `llm.token_count.completion` (per-call, not summed)
- Conversation context accumulates — step 3's input includes step 1 and 2's outputs
- Tool results feed back into subsequent LLM call inputs
- Timestamps use real ATIF step timestamps; each step's duration spans until the next step starts
- `<synthetic>` model names (used by Harbor internally) are filtered out
- Span IDs are deterministic (SHA-256 of session_id + context) so re-converting produces identical traces

## More information

- [Harbor documentation](https://harborframework.com)
- [ATIF specification](https://harborframework.com/docs/agents/trajectory-format)
- [Phoenix documentation](https://docs.arize.com/phoenix)
- [OpenInference specification](https://github.com/Arize-ai/openinference/tree/main/spec)
