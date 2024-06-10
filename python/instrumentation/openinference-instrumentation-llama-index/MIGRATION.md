# Migrations

## v1.x to v2.0

- Requires `llama-index-core>=0.10.43`
- v2.0 uses the new LlamaIndex [instrumentation](https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation/) system as the default option.
- The old callback system is deprecated but still works.
- However, calling the legacy `set_global_handler` method will not activate the new instrumentation system.

##### Old (v1.x)

Calling `set_global_handler` is now considered a [legacy](https://docs.llamaindex.ai/en/stable/module_guides/observability/) functionality.

```python
import llama_index.core

llama_index.core.set_global_handler("arize_phoenix")
```

##### New (v2.x)

Calling `LlamaIndexInstrumentor().instrument` directly will use the new [instrumentation](https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation/) system by default.

```shell
python -m pip install --upgrade \
    openinference-instrumentation-llama-index \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    "opentelemetry-proto>=1.12.0"
```

```python
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)

# The legacy callback system can still be used by setting 
# `use_legacy_callback_handler=True` as shown below.
# 
# LlamaIndexInstrumentor().instrument(
#     tracer_provider=tracer_provider,
#     use_legacy_callback_handler=True,
# )
```
