"""Span wrapper with a finish-once contract.

``_WithSpan`` holds the attributes to apply when the span ends. Extra
attributes are set at span start (see ``_start_as_current_span``) and the
most important ones at finish, so that OTel's default 128-attribute limit
drops the least important attributes first.
"""

import logging
from typing import Optional

from opentelemetry import trace as trace_api
from opentelemetry.util.types import Attributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithSpan:
    __slots__ = (
        "_span",
        "_extra_attributes",
        "_is_finished",
    )

    def __init__(
        self,
        span: trace_api.Span,
        extra_attributes: Attributes = None,
    ) -> None:
        self._span = span
        self._extra_attributes = extra_attributes
        try:
            self._is_finished = not self._span.is_recording()
        except Exception:
            logger.exception("Failed to check if span is recording")
            self._is_finished = True

    def record_exception(self, exception: BaseException) -> None:
        if self._is_finished:
            return
        try:
            self._span.record_exception(exception)
        except Exception:
            logger.exception("Failed to record exception on span")

    def finish_tracing(
        self,
        status: Optional[trace_api.Status] = None,
        attributes: Attributes = None,
        extra_attributes: Attributes = None,
    ) -> None:
        if self._is_finished:
            return
        for mapping in (
            attributes,
            self._extra_attributes,
            extra_attributes,
        ):
            if not mapping:
                continue
            for key, value in mapping.items():
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
