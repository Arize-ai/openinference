# OpenInference AutoGen

> [!WARNING]
> This package is deprecated. Install `openinference-instrumentation-ag2` and import
> `AG2Instrumentor` from `openinference.instrumentation.ag2` for new applications.

`AutogenInstrumentor` remains available as a compatibility alias and delegates to the AG2
instrumentor. Existing applications can migrate without changing tracing behavior:

```python
from openinference.instrumentation.ag2 import AG2Instrumentor

AG2Instrumentor().instrument()
```
