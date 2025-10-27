"""OpenInference observer for Pipecat pipelines."""

import logging

from opentelemetry import trace as trace_api
from pipecat.observers.base_observer import BaseObserver

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.pipecat._attributes import _FrameAttributeExtractor
from openinference.instrumentation.pipecat._service_detector import _ServiceDetector
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

logger = logging.getLogger(__name__)


class OpenInferenceObserver(BaseObserver):
    """
    Observer that creates OpenInference spans for Pipecat frame processing.

    Observes frame flow through pipeline and creates spans for LLM, TTS, and STT services.
    """

    def __init__(self, tracer: OITracer, config: TraceConfig):
        """
        Initialize the observer.

        Args:
            tracer: OpenInference tracer
            config: Trace configuration
        """
        super().__init__()
        self._tracer = tracer
        self._config = config
        self._detector = _ServiceDetector()
        self._attribute_extractor = _FrameAttributeExtractor()

        # Track active spans per service instance
        # Key: id(service), Value: {"span": span, "frame_count": int}
        self._active_spans = {}

        # Track the last frame seen from each service to detect completion
        self._last_frames = {}

    async def on_push_frame(self, data):
        """
        Called when a frame is pushed between processors.

        Args:
            data: FramePushed event data with source, destination, frame, direction
        """
        try:
            # Detect if source is a service we care about
            service_type = self._detector.detect_service_type(data.source)

            if service_type:
                await self._handle_service_frame(data, service_type)

        except Exception as e:
            logger.debug(f"Error in observer: {e}")

    async def on_process_frame(self, data):
        """
        Called when a frame is being processed.

        Args:
            data: FrameProcessed event data
        """
        # For now, we only care about push events
        pass

    async def _handle_service_frame(self, data, service_type: str):
        """
        Handle frame from an LLM, TTS, or STT service.

        Args:
            data: FramePushed event data
            service_type: "llm", "tts", or "stt"
        """
        from pipecat.frames.frames import EndFrame, ErrorFrame

        service = data.source
        service_id = id(service)
        frame = data.frame

        # Check if we already have a span for this service
        if service_id not in self._active_spans:
            # Create new span and set as active
            span = self._create_service_span(service, service_type)
            self._active_spans[service_id] = {
                "span": span,
                "frame_count": 0,
                "service_type": service_type,
            }

        # Increment frame count for this service
        span_info = self._active_spans[service_id]
        span_info["frame_count"] += 1

        # Extract and add attributes from this frame to the span
        span = span_info["span"]
        frame_attrs = self._attribute_extractor.extract_from_frame(frame)
        for key, value in frame_attrs.items():
            span.set_attribute(key, value)

        # Store this as the last frame from this service
        self._last_frames[service_id] = frame

        # Finish span only on completion frames (EndFrame or ErrorFrame)
        if isinstance(frame, (EndFrame, ErrorFrame)):
            self._finish_span(service_id)

    def _create_service_span(self, service, service_type: str):
        """
        Create a span for a service.

        Args:
            service: The service instance
            service_type: "llm", "tts", or "stt"

        Returns:
            The created span
        """
        # Extract metadata
        metadata = self._detector.extract_service_metadata(service)

        # Create span name
        span_name = f"pipecat.{service_type}"

        # Build attributes
        attributes = {
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            "service.name": metadata.get("provider", "unknown"),
        }

        # Add model if available
        if "model" in metadata:
            attributes["model"] = metadata["model"]

        # Add voice if available (TTS)
        if "voice" in metadata:
            attributes["voice"] = metadata["voice"]

        if "voice_id" in metadata:
            attributes["voice_id"] = metadata["voice_id"]

        # Create span using start_as_current_span to ensure it's active
        span = self._tracer.start_span(
            name=span_name,
            attributes=attributes,
        )

        logger.debug(f"Created span {span_name} for {metadata.get('provider')} {service_type}")

        return span

    def _finish_span(self, service_id: int):
        """
        Finish a span for a service.

        Args:
            service_id: The id() of the service instance
        """
        if service_id not in self._active_spans:
            return

        span_info = self._active_spans.pop(service_id)
        span = span_info["span"]

        # End the span with OK status
        span.set_status(trace_api.Status(trace_api.StatusCode.OK))
        span.end()

        logger.debug(
            f"Finished span {span.name} after {span_info['frame_count']} frames"
        )

        # Clean up last frame tracking
        self._last_frames.pop(service_id, None)
