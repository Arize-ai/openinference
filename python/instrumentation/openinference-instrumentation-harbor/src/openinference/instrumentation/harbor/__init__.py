import logging
from typing import Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore

from openinference.instrumentation.harbor._converter import (
    convert_trajectory,
    convert_trajectory_dir,
    convert_trajectory_file,
)
from openinference.instrumentation.harbor._file_exporter import (
    OTLPJsonFileExporter,
    export_spans_to_file,
)
from openinference.instrumentation.harbor._phoenix import phoenix_import, phoenix_import_spans

_instruments = ("harbor-ai",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class HarborInstrumentor(BaseInstrumentor):  # type: ignore
    """
    OpenInference instrumentor for Harbor agent evaluation framework.

    Primary usage is post-hoc: convert ATIF trajectory JSON files to OTel spans
    using convert_trajectory() / convert_trajectory_file().
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        pass

    def _uninstrument(self, **kwargs: Any) -> None:
        pass


__all__ = [
    "HarborInstrumentor",
    "OTLPJsonFileExporter",
    "convert_trajectory",
    "convert_trajectory_dir",
    "convert_trajectory_file",
    "export_spans_to_file",
    "phoenix_import",
    "phoenix_import_spans",
]
