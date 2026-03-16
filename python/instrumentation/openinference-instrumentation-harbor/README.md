# OpenInference Harbor Instrumentation

OpenInference instrumentation for [Harbor](https://github.com/harbor-ai/harbor), an agent evaluation framework.

Converts Harbor's ATIF (Agent Trajectory Interchange Format) trajectory JSON files to OpenInference/OpenTelemetry spans, enabling observability for agent evaluations that run in sandboxed (offline) environments.

## Installation

```bash
pip install openinference-instrumentation-harbor
```

## Usage

### Post-hoc Conversion

Convert ATIF trajectory files to OTLP JSON after evaluation completes:

```python
from openinference.instrumentation.harbor import convert_trajectory_file, export_spans_to_file

spans = convert_trajectory_file("trajectory.json")
export_spans_to_file(spans, "traces.json")
```

### CLI

```bash
harbor-to-otel trajectory.json -o traces.json
harbor-to-otel trajectories_dir/ -o output_dir/
harbor-to-otel trajectory.json --phoenix http://localhost:6006
```

### Phoenix Import

```python
from openinference.instrumentation.harbor import phoenix_import

phoenix_import("traces.json", endpoint="http://localhost:6006")
```

### Real-time Instrumentation

When Harbor SDK is installed, the instrumentor can hook into trial execution:

```python
from openinference.instrumentation.harbor import HarborInstrumentor

HarborInstrumentor().instrument()
```
