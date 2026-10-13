# OpenInference AutoGen

> [!WARNING]
> This package is deprecated. Install `openinference-instrumentation-ag2` and import
> `AG2Instrumentor` from `openinference.instrumentation.ag2` for new applications.

`AutogenInstrumentor` remains available for applications using the legacy `autogen`
distribution and delegates tracing to the AG2 instrumentor singleton. New applications can
migrate by changing the package and import:

```python
from openinference.instrumentation.ag2 import AG2Instrumentor

AG2Instrumentor().instrument()
```
