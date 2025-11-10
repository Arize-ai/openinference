# Pipecat Tracing Integration Plan

## Executive Summary

Reference Implementation: https://github.com/pipecat-ai/pipecat/tree/main/src/pipecat/utils/tracing

This document outlines the design and implementation plan for integrating Pipecat's native tracing capabilities into the OpenInference instrumentation for Pipecat. The goal is to align with Pipecat's official tracing implementation while maintaining OpenInference semantic conventions.

## Current State Analysis

### OpenInference Instrumentation (Current)

**Architecture:**
- Observer-based pattern using `OpenInferenceObserver` extending `BaseObserver`
- Frame-by-frame attribute extraction via specialized extractors
- Turn tracking with context attachment/detachment
- Service span creation on-demand as frames arrive

**Strengths:**
- ✅ Comprehensive frame attribute extraction
- ✅ OpenInference semantic conventions compliance
- ✅ Automatic span hierarchy (turn → service spans)
- ✅ Duplicate frame detection
- ✅ Rich metadata capture

**Weaknesses:**
- ❌ Non-standard attribute naming compared to Pipecat's conventions
- ❌ No TTFB (Time To First Byte) metrics capture
- ❌ Missing character count for TTS operations
- ❌ No VAD (Voice Activity Detection) status tracking
- ❌ Limited streaming output aggregation
- ❌ No GenAI semantic conventions alignment

### Pipecat Native Tracing

**Architecture:**
- Decorator-based instrumentation (`@traced_llm`, `@traced_tts`, `@traced_stt`)
- Context providers for conversation and turn management
- `TurnTraceObserver` for turn lifecycle management
- GenAI semantic conventions (gen_ai.*)

**Strengths:**
- ✅ GenAI semantic convention alignment
- ✅ TTFB metrics capture
- ✅ Character count tracking for TTS
- ✅ VAD status for STT
- ✅ Streaming output aggregation
- ✅ Tool call tracking with definitions
- ✅ Session-level attributes for real-time services

**Weaknesses:**
- ❌ Requires manual decorator application
- ❌ Less comprehensive frame-level instrumentation
- ❌ OpenInference conventions not followed

---

## Comparison: Attribute Naming

### Current OpenInference vs. Pipecat GenAI Conventions

| **Feature** | **OpenInference (Current)** | **Pipecat GenAI** | **Recommendation** |
|-------------|------------------------------|-------------------|-------------------|
| **LLM Model** | `llm.model_name` | `gen_ai.request.model` | Add both |
| **Provider** | `llm.provider` | `gen_ai.system` | Add both |
| **Operation** | `openinference.span.kind` | `gen_ai.operation.name` | Add both |
| **Input** | `input.value` | `input` (for prompts) | Keep both |
| **Output** | `output.value` | `output` (for responses) | Keep both |
| **Messages** | `llm.input_messages` | (in `input`) | Keep current |
| **Tokens** | `llm.token_count.*` | `gen_ai.usage.*` | Add GenAI |
| **TTFB** | ❌ Missing | `metrics.ttfb` | **Add** |
| **TTS Chars** | ❌ Missing | `metrics.character_count` | **Add** |
| **Tools** | `tool.name`, `tool.parameters` | `tools.count`, `tools.names`, `tools.definitions` | **Add** |
| **VAD** | ❌ Missing | `vad_enabled` | **Add** |
| **Voice** | `audio.voice_id` | `voice_id` | Keep current |
| **Transcript** | `audio.transcript` | `transcript`, `is_final` | Add `is_final` |

---

## Integration Strategy

### Phase 1: Enhance Attribute Extraction (High Priority)

**Goal:** Add missing metrics and GenAI semantic conventions while maintaining OpenInference compatibility.

#### 1.1 Add TTFB Metrics Extraction

**Location:** `_attributes.py`

**Implementation:**
```python
class MetricsFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from metrics frames."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}

        if not hasattr(frame, "data") or not frame.data:
            return results

        for metrics_data in frame.data:
            if isinstance(metrics_data, TTFBMetricsData):
                # Add both conventions
                ttfb_value = getattr(metrics_data, "value", None)
                if ttfb_value:
                    results["metrics.ttfb"] = ttfb_value  # Pipecat convention
                    results["service.ttfb_seconds"] = ttfb_value  # OpenInference
```

#### 1.2 Add Character Count for TTS

**Location:** `_attributes.py` - `TTSServiceAttributeExtractor`

**Implementation:**
```python
class TextFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from text frames."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results = super().extract_from_frame(frame)
        if hasattr(frame, "text") and frame.text:
            text = frame.text
            # Add character count for TTS frames
            if isinstance(frame, TTSTextFrame):
                results["metrics.character_count"] = len(text)
                results["tts.character_count"] = len(text)
```

#### 1.3 Add VAD Status for STT

**Location:** `_attributes.py` - `STTServiceAttributeExtractor`

**Implementation:**
```python
class STTServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract attributes from an STT service."""

    attributes: Dict[str, Any] = {
        # ... existing attributes ...
        "vad_enabled": lambda service: getattr(service, "vad_enabled", None),
        "vad.enabled": lambda service: getattr(service, "vad_enabled", None),
    }
```

#### 1.4 Add `is_final` for Transcriptions

**Location:** `_attributes.py` - `TextFrameExtractor`

**Implementation:**
```python
class TextFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from text frames."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results = super().extract_from_frame(frame)
        if hasattr(frame, "text"):
            text = frame.text
            if isinstance(frame, TranscriptionFrame):
                results[SpanAttributes.INPUT_VALUE] = text
                results[AudioAttributes.AUDIO_TRANSCRIPT] = text
                results["transcript"] = text  # GenAI convention
                results["is_final"] = True
                results["transcript.is_final"] = True
            elif isinstance(frame, InterimTranscriptionFrame):
                results[SpanAttributes.INPUT_VALUE] = text
                results[AudioAttributes.AUDIO_TRANSCRIPT] = text
                results["transcript"] = text
                results["is_final"] = False
                results["transcript.is_final"] = False
```

#### 1.5 Add GenAI Semantic Conventions

**Location:** `_attributes.py` - All service extractors

**Implementation:**
```python
class LLMServiceAttributeExtractor(ServiceAttributeExtractor):
    """Extract attributes from an LLM service."""

    attributes: Dict[str, Any] = {
        # OpenInference conventions
        SpanAttributes.OPENINFERENCE_SPAN_KIND: lambda service: (
            OpenInferenceSpanKindValues.LLM.value
        ),
        SpanAttributes.LLM_MODEL_NAME: lambda service: (
            getattr(service, "model_name", None) or getattr(service, "model", None)
        ),
        SpanAttributes.LLM_PROVIDER: lambda service: detect_provider_from_service(service),

        # GenAI semantic conventions (dual convention support)
        "gen_ai.request.model": lambda service: (
            getattr(service, "model_name", None) or getattr(service, "model", None)
        ),
        "gen_ai.system": lambda service: detect_provider_from_service(service),
        "gen_ai.operation.name": lambda service: "chat",  # or detect from service
        "gen_ai.output.type": lambda service: "text",
    }
```

**Similar updates for:**
- `TTSServiceAttributeExtractor` - add `gen_ai.operation.name = "text_to_speech"`
- `STTServiceAttributeExtractor` - add `gen_ai.operation.name = "speech_to_text"`

#### 1.6 Enhanced Tool Tracking

**Location:** `_attributes.py` - `LLMContextFrameExtractor`

**Implementation:**
```python
class LLMContextFrameExtractor(FrameAttributeExtractor):
    """Extract attributes from an LLM context frame."""

    def extract_from_frame(self, frame: Frame) -> Dict[str, Any]:
        results: Dict[str, Any] = super().extract_from_frame(frame)

        if hasattr(frame.context, "_tools") and frame.context._tools:
            tools = frame.context._tools
            results["llm.tools_count"] = len(tools)
            results["tools.count"] = len(tools)  # GenAI convention

            # Extract tool names
            tool_names = [tool.get("name", tool.get("function", {}).get("name", ""))
                          for tool in tools if isinstance(tool, dict)]
            if tool_names:
                results["tools.names"] = safe_json_dumps(tool_names)

            # Extract tool definitions (truncated for large payloads)
            tools_json = safe_json_dumps(tools)
            if tools_json and len(tools_json) < 10000:  # 10KB limit
                results["tools.definitions"] = tools_json

        return results
```

---

### Phase 2: Nested LLM Call Detection (High Priority)

**Goal:** Capture LLM calls that happen within TTS/STT services as separate child spans.

#### 2.1 Problem Statement

Many modern TTS and STT services use LLMs internally:
- **TTS Examples:**
  - OpenAI TTS can use GPT models for voice modulation
  - Cartesia uses LLMs for natural speech patterns
  - ElevenLabs may use LLMs for context-aware intonation
- **STT Examples:**
  - Post-processing transcriptions with LLMs for punctuation/formatting
  - Context-aware transcription refinement
  - Language detection using LLM classifiers

**Current Issue:** These nested LLM calls are either:
1. Not captured at all
2. Merged into the parent TTS/STT span without visibility
3. Missing prompt/response details

#### 2.2 Detection Strategy

**Location:** `_observer.py` - `_handle_service_frame()`

**Approach:** Track service nesting depth and parent-child relationships.

**Implementation:**
```python
class OpenInferenceObserver(BaseObserver):
    def __init__(self, ...):
        # ... existing init ...

        # Track service call stack for nesting detection
        self._service_call_stack: List[Tuple[int, str, Span]] = []  # [(service_id, type, span)]
        self._nested_llm_calls: Set[int] = set()  # Track which LLM calls are nested

    async def _handle_service_frame(self, data: FramePushed) -> None:
        """Handle frame from any service, detecting nested calls."""
        from pipecat.frames.frames import EndFrame, ErrorFrame

        service = data.source
        service_id = id(service)
        frame = data.frame
        service_type = detect_service_type(service)

        # Check if this is a new service call
        if service_id not in self._active_spans:
            # Detect if we're nested inside another service
            parent_service_span = None
            if self._service_call_stack:
                # We have an active parent service - this is a nested call
                parent_service_id, parent_type, parent_span = self._service_call_stack[-1]
                parent_service_span = parent_span

                # Mark as nested if this is an LLM within TTS/STT
                if service_type == "llm" and parent_type in ("tts", "stt", "vision"):
                    self._nested_llm_calls.add(service_id)
                    self._log_debug(
                        f"  Detected nested LLM call within {parent_type} service"
                    )

            # Create span with proper parent context
            span = self._create_service_span(
                service,
                service_type,
                parent_span=parent_service_span
            )

            self._active_spans[service_id] = {
                "span": span,
                "frame_count": 0,
                "input_texts": [],
                "output_texts": [],
                "nested": service_id in self._nested_llm_calls,
                "parent_type": self._service_call_stack[-1][1] if self._service_call_stack else None,
            }

            # Push this service onto the call stack
            self._service_call_stack.append((service_id, service_type, span))

        # ... existing frame attribute extraction ...

        # Finish span and pop from stack on completion
        if isinstance(frame, (EndFrame, ErrorFrame)):
            # Pop from call stack
            if self._service_call_stack and self._service_call_stack[-1][0] == service_id:
                self._service_call_stack.pop()

            # Clean up nested tracking
            if service_id in self._nested_llm_calls:
                self._nested_llm_calls.remove(service_id)

            self._finish_span(service_id)

    def _create_service_span(
        self,
        service: FrameProcessor,
        service_type: str,
        parent_span: Optional[Span] = None
    ) -> Span:
        """
        Create a span for a service with proper parent relationship.

        Args:
            service: The service instance
            service_type: Service type (llm, tts, stt, etc.)
            parent_span: Optional parent span for nested calls
        """
        # Determine span name based on nesting
        if parent_span:
            span_name = f"pipecat.{service_type}.nested"
        else:
            span_name = f"pipecat.{service_type}"

        self._log_debug(f">>> Creating {span_name} span")

        # Create span with parent context if provided
        if parent_span:
            # Create child span under the parent service span
            from opentelemetry import trace as trace_api
            parent_context = trace_api.set_span_in_context(parent_span)
            span = self._tracer.start_span(
                name=span_name,
                context=parent_context,
            )
        else:
            # Regular span under the turn context
            span = self._tracer.start_span(
                name=span_name,
            )

        # Set service attributes
        span.set_attribute("service.name", service.__class__.__name__)

        # Extract and apply service-specific attributes
        service_attrs = extract_service_attributes(service)
        for key, value in service_attrs.items():
            if value is not None:
                span.set_attribute(key, value)

        return span
```

#### 2.3 Enhanced Span Metadata for Nested Calls

**Location:** `_observer.py` - `_finish_span()`

Add metadata to identify nested calls:

```python
def _finish_span(self, service_id: int) -> None:
    """Finish a span for a service."""
    if service_id not in self._active_spans:
        return

    span_info = self._active_spans.pop(service_id)
    span = span_info["span"]

    # Mark as nested if applicable
    if span_info.get("nested"):
        span.set_attribute("service.nested", True)
        span.set_attribute("service.parent_type", span_info.get("parent_type"))
        span.set_attribute("service.purpose", f"internal_to_{span_info.get('parent_type')}")

    # ... existing input/output aggregation ...
```

#### 2.4 Example Trace Structure

With this implementation, a TTS call using an internal LLM would produce:

```
Turn Span (pipecat.conversation.turn)
└── TTS Span (pipecat.tts)
    ├── attributes:
    │   ├── gen_ai.system: "cartesia"
    │   ├── gen_ai.operation.name: "text_to_speech"
    │   ├── voice_id: "sonic"
    │   └── metrics.character_count: 145
    └── Nested LLM Span (pipecat.llm.nested)
        ├── attributes:
        │   ├── service.nested: true
        │   ├── service.parent_type: "tts"
        │   ├── service.purpose: "internal_to_tts"
        │   ├── gen_ai.system: "openai"
        │   ├── gen_ai.request.model: "gpt-4"
        │   ├── gen_ai.operation.name: "chat"
        │   ├── input.value: "Generate natural speech pattern for..."
        │   └── output.value: "[prosody instructions]"
```

---

### Phase 3: Streaming Output Aggregation (Medium Priority)

**Goal:** Capture complete streaming responses, not just final frames.

#### 2.1 Add Output Accumulation in Service Spans

**Location:** `_observer.py` - `_handle_service_frame()`

**Current behavior:** Service spans collect frames but don't aggregate streaming text properly.

**Enhancement:**
```python
async def _handle_service_frame(self, data: FramePushed) -> None:
    """Handle frame from an LLM, TTS, or STT service."""
    service = data.source
    service_id = id(service)
    frame = data.frame

    # ... existing span creation logic ...

    # Enhanced streaming aggregation
    span_info = self._active_spans[service_id]

    # Detect streaming LLM responses
    if isinstance(frame, (LLMFullResponseStartFrame, LLMFullResponseEndFrame)):
        # Track response phase
        span_info["response_phase"] = "start" if isinstance(frame, LLMFullResponseStartFrame) else "end"

    # Aggregate streaming text output
    from pipecat.frames.frames import TextFrame, LLMTextFrame
    if isinstance(frame, (TextFrame, LLMTextFrame)):
        if hasattr(frame, "text") and frame.text:
            service_type = detect_service_type(service)
            if service_type == "llm":
                # This is LLM output - aggregate it
                span_info["output_texts"].append(str(frame.text))
```

---

### Phase 3: Context Provider Integration (Low Priority)

**Goal:** Align with Pipecat's context provider pattern for better ecosystem compatibility.

**Note:** This is optional since our current implementation already manages context properly. This would primarily benefit users who want to use both native Pipecat tracing and OpenInference simultaneously.

#### 3.1 Add Turn Context Provider

**New File:** `_context_providers.py`

**Implementation:**
```python
"""Context providers for OpenInference Pipecat instrumentation."""

from typing import Optional
from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.trace import SpanContext


class TurnContextProvider:
    """Singleton provider for turn-level trace context."""

    _instance: Optional["TurnContextProvider"] = None
    _current_context: Optional[Context] = None

    @classmethod
    def get_instance(cls) -> "TurnContextProvider":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_current_turn_context(self, span_context: SpanContext) -> None:
        """Set the current turn's span context."""
        # Create non-recording span for context propagation
        span = trace_api.NonRecordingSpan(span_context)
        self._current_context = trace_api.set_span_in_context(span)

    def get_current_turn_context(self) -> Optional[Context]:
        """Get the current turn's context."""
        return self._current_context

    def clear(self) -> None:
        """Clear the current turn context."""
        self._current_context = None


# Convenience function
def get_current_turn_context() -> Optional[Context]:
    """Get the OpenTelemetry context for the current turn."""
    return TurnContextProvider.get_instance().get_current_turn_context()
```

**Integration in `_observer.py`:**
```python
from openinference.instrumentation.pipecat._context_providers import TurnContextProvider

async def _start_turn(self, data: FramePushed) -> Token[Context]:
    """Start a new conversation turn."""
    # ... existing turn creation logic ...

    # Update context provider for ecosystem compatibility
    if self._turn_span:
        span_context = self._turn_span.get_span_context()
        TurnContextProvider.get_instance().set_current_turn_context(span_context)

    return self._turn_context_token

async def _finish_turn(self, interrupted: bool = False) -> None:
    """Finish the current turn."""
    # ... existing finish logic ...

    # Clear context provider
    TurnContextProvider.get_instance().clear()
```

---

## Implementation Roadmap

### Immediate Actions (Week 1)

**Priority 1: Core Metrics & Conventions**

1. **Add TTFB metrics** - Enhance `MetricsFrameExtractor`
2. **Add character count** - Update `TextFrameExtractor` for TTS
3. **Add VAD status** - Update `STTServiceAttributeExtractor`
4. **Add `is_final` flag** - Update `TextFrameExtractor` for transcriptions

**Files to modify:**
- `src/openinference/instrumentation/pipecat/_attributes.py`

**Estimated effort:** 4-6 hours

**Priority 2: Nested LLM Call Detection**

5. **Add service call stack tracking** - Track parent-child service relationships
6. **Implement nested span creation** - Create child spans for nested LLM calls
7. **Add nested call metadata** - Mark spans with nesting information

**Files to modify:**
- `src/openinference/instrumentation/pipecat/_observer.py`

**Estimated effort:** 6-8 hours

**Total Week 1:** 10-14 hours

### Short-term (Week 2)

1. **Add GenAI semantic conventions** - Dual attribute support
2. **Enhanced tool tracking** - Tool names and definitions
3. **Testing for nested calls** - Validate service nesting detection
4. **Unit and integration tests**

**Files to modify:**
- `src/openinference/instrumentation/pipecat/_attributes.py`
- `src/openinference/instrumentation/pipecat/_observer.py`
- Tests

**Estimated effort:** 10-12 hours

### Medium-term (Week 3-4)

1. **Streaming output aggregation** - Better LLM response capture
2. **Documentation updates** - Include nested call examples
3. **Example updates** - Show TTS/STT with internal LLM usage
4. **Performance testing** - Ensure minimal overhead for nesting detection

**Files to modify:**
- `src/openinference/instrumentation/pipecat/_observer.py`
- `README.md`
- Examples
- Performance benchmarks

**Estimated effort:** 12-16 hours

### Long-term (Optional)

1. **Context provider integration** - Ecosystem compatibility
2. **Decorator support** - Optional manual instrumentation
3. **GenAI convention migration guide**

**New files:**
- `src/openinference/instrumentation/pipecat/_context_providers.py`
- `src/openinference/instrumentation/pipecat/_decorators.py` (optional)
- Migration guide documentation

**Estimated effort:** 16-20 hours

---

## Attribute Mapping Reference

### Complete Dual Convention Mapping

```python
ATTRIBUTE_MAPPING = {
    # Service identification
    "service.type": "service.type",  # Keep
    "service.provider": "gen_ai.system",  # Add GenAI

    # LLM attributes
    "llm.model_name": "gen_ai.request.model",  # Add GenAI
    "llm.provider": "gen_ai.system",  # Add GenAI
    "openinference.span.kind": "gen_ai.operation.name",  # Map to operation

    # Input/Output
    "input.value": "input",  # Both
    "output.value": "output",  # Both
    "llm.input_messages": None,  # OpenInference only

    # Metrics
    "service.ttfb_seconds": "metrics.ttfb",  # Add GenAI
    "tts.character_count": "metrics.character_count",  # Add GenAI

    # Audio
    "audio.transcript": "transcript",  # Both
    "audio.is_final": "is_final",  # Add flat version
    "audio.voice_id": "voice_id",  # Both
    "vad.enabled": "vad_enabled",  # Add flat version

    # Tools
    "llm.tools_count": "tools.count",  # Add GenAI
    None: "tools.names",  # Add (missing)
    None: "tools.definitions",  # Add (missing)
}
```

---

## Testing Strategy

### Unit Tests

1. **Test dual attribute generation**
   ```python
   def test_llm_service_dual_conventions():
       """Test that both OpenInference and GenAI attributes are set."""
       service = MockLLMService(model="gpt-4")
       attributes = extract_service_attributes(service)

       # OpenInference conventions
       assert attributes["llm.model_name"] == "gpt-4"
       assert attributes["llm.provider"] == "openai"

       # GenAI conventions
       assert attributes["gen_ai.request.model"] == "gpt-4"
       assert attributes["gen_ai.system"] == "openai"
   ```

2. **Test TTFB metrics extraction**
3. **Test character count for TTS**
4. **Test VAD status extraction**
5. **Test tool definition extraction**

6. **Test nested LLM call detection**
   ```python
   async def test_nested_llm_in_tts():
       """Test that nested LLM calls are properly detected and traced."""
       observer = OpenInferenceObserver(tracer=mock_tracer, config=TraceConfig())

       # Simulate TTS service
       tts_service = MockTTSService()
       tts_frame = StartFrame()

       # Start TTS span
       await observer._handle_service_frame(
           FramePushed(source=tts_service, frame=tts_frame, ...)
       )

       # Simulate nested LLM call within TTS
       llm_service = MockLLMService()
       llm_frame = LLMMessagesFrame(...)

       await observer._handle_service_frame(
           FramePushed(source=llm_service, frame=llm_frame, ...)
       )

       # Verify nesting
       assert len(observer._service_call_stack) == 2
       assert llm_service_id in observer._nested_llm_calls

       # Verify span attributes
       llm_span_info = observer._active_spans[id(llm_service)]
       assert llm_span_info["nested"] == True
       assert llm_span_info["parent_type"] == "tts"
   ```

7. **Test service call stack management**
   ```python
   async def test_service_call_stack_push_pop():
       """Test that service call stack is properly managed."""
       observer = OpenInferenceObserver(tracer=mock_tracer, config=TraceConfig())

       # Push services onto stack
       tts_service = MockTTSService()
       llm_service = MockLLMService()

       # Start TTS
       await observer._handle_service_frame(
           FramePushed(source=tts_service, frame=StartFrame(), ...)
       )
       assert len(observer._service_call_stack) == 1

       # Start nested LLM
       await observer._handle_service_frame(
           FramePushed(source=llm_service, frame=LLMMessagesFrame(), ...)
       )
       assert len(observer._service_call_stack) == 2

       # End LLM
       await observer._handle_service_frame(
           FramePushed(source=llm_service, frame=EndFrame(), ...)
       )
       assert len(observer._service_call_stack) == 1

       # End TTS
       await observer._handle_service_frame(
           FramePushed(source=tts_service, frame=EndFrame(), ...)
       )
       assert len(observer._service_call_stack) == 0
   ```

### Integration Tests

1. **End-to-end trace validation** - Verify complete traces with all attributes
2. **Streaming aggregation test** - Verify LLM streaming output collection
3. **Backward compatibility** - Ensure existing traces still work

### Performance Tests

1. **Overhead measurement** - Dual attributes shouldn't add significant overhead
2. **Memory usage** - Tool definitions might increase memory usage
3. **Attribute size limits** - Test with large tool definitions

---

## Migration Guide (for users)

### No Breaking Changes

All changes are **additive** - existing OpenInference attributes remain unchanged. New GenAI convention attributes are added alongside.

### New Attributes Available

After upgrading, traces will include:

**GenAI Semantic Conventions:**
- `gen_ai.request.model`
- `gen_ai.system`
- `gen_ai.operation.name`

**Enhanced Metrics:**
- `metrics.ttfb` - Time to first byte
- `metrics.character_count` - TTS character count
- `is_final` - Transcription finality status
- `vad_enabled` - Voice activity detection status

**Enhanced Tool Tracking:**
- `tools.count` - Number of tools available
- `tools.names` - Array of tool names
- `tools.definitions` - Full tool definitions (if < 10KB)

### Querying Traces

Both conventions can be queried:

```python
# OpenInference convention (existing)
traces.filter(lambda t: t.attributes.get("llm.model_name") == "gpt-4")

# GenAI convention (new)
traces.filter(lambda t: t.attributes.get("gen_ai.request.model") == "gpt-4")
```

---

## Benefits Summary

### For Users

1. **Better observability** - TTFB, character counts, VAD status
2. **Nested call visibility** - See LLM calls inside TTS/STT services with full prompts and responses
3. **Standard compliance** - GenAI semantic conventions alignment
4. **Enhanced tool tracking** - See all tool definitions
5. **Backward compatible** - No breaking changes
6. **Ecosystem compatibility** - Works with Pipecat's native tracing
7. **Cost tracking** - Track LLM usage even when embedded in other services
8. **Performance debugging** - Identify slow nested LLM calls affecting TTS/STT latency

### For the Project

1. **Alignment with Pipecat** - Follows official patterns
2. **Future-proof** - GenAI conventions are industry standard
3. **Richer telemetry** - More actionable data
4. **Better debugging** - TTFB and streaming metrics
5. **Complete visibility** - No hidden service calls
6. **Accurate span hierarchy** - Proper parent-child relationships

### Key Use Cases Enabled

#### 1. TTS with LLM-based Voice Modulation
```
User speaks → STT → LLM (main) → TTS (with nested LLM for prosody) → Audio output
```
**Before:** Only see TTS span, miss the LLM call for voice modulation
**After:** See complete chain including nested LLM with its prompt/response

#### 2. STT with LLM Post-Processing
```
Audio input → STT (with nested LLM for punctuation) → Formatted text
```
**Before:** Only see STT span with final output
**After:** See both raw STT output AND the LLM refinement step

#### 3. Cost Attribution
Track token usage from LLMs even when they're called internally by TTS/STT:
- See which services use nested LLMs
- Track token costs per service type
- Identify opportunities to cache or optimize nested calls

---

## Open Questions

1. **Should we deprecate old attribute names?**
   - Recommendation: No, maintain both for compatibility

2. **How to handle attribute size limits?**
   - Recommendation: 10KB limit for tool definitions, truncate with warning

3. **Should we support decorator-based instrumentation?**
   - Recommendation: Not initially, observer pattern is sufficient

4. **GenAI token usage attributes?**
   - Recommendation: Add `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` mapping

---

## References

- [Pipecat Tracing Source](https://github.com/pipecat-ai/pipecat/tree/main/src/pipecat/utils/tracing)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [OpenInference Semantic Conventions](https://github.com/Arize-ai/openinference)

---

**Document Version:** 1.0
**Last Updated:** 2025-01-10
**Author:** OpenInference Pipecat Instrumentation Team
