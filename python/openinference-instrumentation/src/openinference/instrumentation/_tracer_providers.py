from typing import Any, Optional

from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider

from ._tracers import OITracer
from .config import TraceConfig


class TracerProvider(OTelTracerProvider):
    def __init__(
        self,
        *args: Any,
        config: Optional[TraceConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._oi_trace_config = config or TraceConfig()

    def get_tracer(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> OITracer:
        tracer = super().get_tracer(*args, **kwargs)
        return OITracer(tracer, config=self._oi_trace_config)
