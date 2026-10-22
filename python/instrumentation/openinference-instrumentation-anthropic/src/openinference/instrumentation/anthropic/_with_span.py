from __future__ import annotations

import logging
from itertools import chain
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from opentelemetry import trace as trace_api
from opentelemetry.util.types import Attributes, AttributeValue

from openinference.instrumentation import TraceConfig

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithSpan:
    __slots__ = (
        "_span",
        "_is_finished",
        "_params",
        "_context_metadata",
        "_config",
    )

    def __init__(
        self,
        span: trace_api.Span,
        params: Iterable[Tuple[str, AttributeValue]] = (),
        context_metadata: Optional[AttributeValue] = None,
        config: Optional[TraceConfig] = None,
    ) -> None:
        self._span = span
        self._params = params
        # the metadata from the context the span was started in, to merge the response's into
        self._context_metadata = context_metadata
        # the trace config, for the attributes that the tracer's masking does not cover
        self._config = config
        try:
            self._is_finished = not self._span.is_recording()
        except Exception:
            logger.exception("Failed to check if span is recording")
            self._is_finished = True

    @property
    def is_finished(self) -> bool:
        return self._is_finished

    @property
    def context_metadata(self) -> Optional[AttributeValue]:
        return self._context_metadata

    @property
    def config(self) -> Optional[TraceConfig]:
        return self._config

    @property
    def request_messages(self) -> Any:
        """The request's messages, if the params are a request's, else None."""
        return getattr(self._params, "request_messages", None)

    def set_attributes(self, attributes: Dict[str, AttributeValue]) -> None:
        self._span.set_attributes(attributes)

    def record_exception(self, exception: BaseException) -> None:
        if self._is_finished:
            return
        try:
            self._span.record_exception(exception)
        except Exception:
            logger.exception("Failed to record exception on span")

    def set_status(self, status: Union[trace_api.Status, trace_api.StatusCode]) -> None:
        if self._is_finished:
            return
        try:
            self._span.set_status(status=status)
        except Exception:
            logger.exception("Failed to set status on span")

    def add_event(self, name: str) -> None:
        if self._is_finished:
            return
        try:
            self._span.add_event(name)
        except Exception:
            logger.exception("Failed to add event to span")

    def finish_tracing(
        self,
        status: Optional[trace_api.Status] = None,
        attributes: Attributes = None,
        extra_attributes: Attributes = None,
    ) -> None:
        if self._is_finished:
            return
        for key, value in chain(
            self._params,
            (attributes or {}).items(),
            (extra_attributes or {}).items(),
        ):
            if value is None:
                continue
            try:
                self._span.set_attribute(key, value)
            except Exception:
                logger.exception("Failed to set attribute on span")
        if status is not None:
            try:
                self._span.set_status(status=status)
            except Exception:
                logger.exception("Failed to set status code on span")
        try:
            self._span.end()
        except Exception:
            logger.exception("Failed to end span")
        self._is_finished = True
