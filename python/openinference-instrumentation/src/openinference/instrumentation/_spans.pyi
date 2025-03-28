from typing import Any, Dict, Optional, Union

from opentelemetry.sdk.trace import _Span
from opentelemetry.trace import Span

from ._types import OpenInferenceMimeType
from .config import TraceConfig

class OpenInferenceSpan(_Span):
    def __init__(self, wrapped: Span, config: TraceConfig) -> None: ...
    def set_input(
        self,
        value: Any,
        *,
        mime_type: Optional[OpenInferenceMimeType] = None,
    ) -> None: ...
    def set_output(
        self,
        value: Any,
        *,
        mime_type: Optional[OpenInferenceMimeType] = None,
    ) -> None: ...
    def set_tool(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        parameters: Union[str, Dict[str, Any]],
    ) -> None: ...
