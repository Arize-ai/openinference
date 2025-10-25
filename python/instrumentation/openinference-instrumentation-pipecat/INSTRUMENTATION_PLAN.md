# OpenInference Instrumentation for Pipecat - Implementation Plan

## Executive Summary

This document outlines the plan to generalize the current manual tracing implementation for Pipecat into a proper OpenInference instrumentation package that follows established patterns from other OpenInference instrumentations (OpenAI, LangChain, LlamaIndex).

## Current State Analysis

### Existing Example Implementation

The current tracing example ([examples/trace/tracing_setup.py](examples/trace/tracing_setup.py)) uses a **manual monkey-patching approach** with the following characteristics:

1. **Manual Span Creation**: Directly patches `OpenAILLMService.process_frame`, `OpenAISTTService._transcribe`, and `OpenAITTSService.run_tts`
2. **Turn-Based Tracing**: Implements a `TurnTracker` class to manage conversation turns as separate traces
3. **Trace Structure**: Creates hierarchical traces:
   - Root: `Interaction` span (one per user turn)
   - Children: `STT` → `LLM` → `TTS` spans
   - Auto-instrumented OpenAI spans nested under appropriate parents
4. **OpenInference Conventions**: Uses `CHAIN` span kind for manual operations, relies on OpenAI auto-instrumentation for `LLM` spans

### Key Insights from Current Implementation

**Strengths:**
- Captures full conversation context (user input → bot output)
- Proper parent-child relationships between pipeline phases
- Handles streaming and async operations correctly
- Integrates well with existing OpenAI instrumentation

**Limitations:**
- Hardcoded for OpenAI services only
- Manual patching is fragile and library-specific
- No generalization to other LLM/TTS/STT providers
- Requires deep knowledge of Pipecat internals
- Not reusable across different Pipecat applications

## OpenInference Instrumentation Patterns

### Pattern Analysis from Existing Instrumentations

#### 1. OpenAI Instrumentation Pattern
**File**: [openinference-instrumentation-openai](../openinference-instrumentation-openai/src/openinference/instrumentation/openai/__init__.py)

**Key Characteristics:**
- **BaseInstrumentor**: Extends OpenTelemetry's `BaseInstrumentor`
- **Wrapping Strategy**: Uses `wrapt.wrap_function_wrapper` to intercept method calls
- **Target**: Single method interception - `OpenAI.request()` and `AsyncOpenAI.request()`
- **Span Management**:
  - Creates spans before method execution
  - Handles streaming responses by monkey-patching response objects
  - Extracts attributes from both request and response
- **Context Propagation**: Uses OpenTelemetry context API for proper parent-child relationships

**Code Pattern:**
```python
class OpenAIInstrumentor(BaseInstrumentor):
    def _instrument(self, **kwargs):
        tracer = OITracer(...)
        wrap_function_wrapper(
            module="openai",
            name="OpenAI.request",
            wrapper=_Request(tracer=tracer, openai=openai)
        )
```

#### 2. LangChain Instrumentation Pattern
**File**: [openinference-instrumentation-langchain](../openinference-instrumentation-langchain/src/openinference/instrumentation/langchain/__init__.py)

**Key Characteristics:**
- **Callback-Based**: Integrates with LangChain's existing callback system
- **Hook Point**: Wraps `BaseCallbackManager.__init__` to inject custom callback handler
- **Tracer Integration**: Adds `OpenInferenceTracer` to all callback managers
- **Run Tracking**: Maintains a map of run IDs to spans for context propagation
- **Non-Invasive**: Works through LangChain's designed extension points

**Code Pattern:**
```python
class LangChainInstrumentor(BaseInstrumentor):
    def _instrument(self, **kwargs):
        tracer = OpenInferenceTracer(...)
        wrap_function_wrapper(
            module="langchain_core.callbacks",
            name="BaseCallbackManager.__init__",
            wrapper=_BaseCallbackManagerInit(tracer)
        )
```

#### 3. LlamaIndex Instrumentation Pattern
**File**: [openinference-instrumentation-llama-index](../openinference-instrumentation-llama-index/src/openinference/instrumentation/llama_index/__init__.py)

**Key Characteristics:**
- **Event/Span Handlers**: Uses LlamaIndex's built-in instrumentation dispatcher
- **Handler Registration**: Registers custom `_SpanHandler` and `EventHandler` with dispatcher
- **Framework Integration**: Leverages library's native instrumentation hooks
- **No Monkey-Patching**: Uses official extension points instead

**Code Pattern:**
```python
class LlamaIndexInstrumentor(BaseInstrumentor):
    def _instrument(self, **kwargs):
        dispatcher = get_dispatcher()
        span_handler = _SpanHandler(tracer=tracer)
        event_handler = EventHandler(span_handler=span_handler)
        dispatcher.add_span_handler(span_handler)
        dispatcher.add_event_handler(event_handler)
```

### Common Patterns Across All Instrumentations

1. **BaseInstrumentor Inheritance**: All extend `opentelemetry.instrumentation.instrumentor.BaseInstrumentor`
2. **OITracer Usage**: Wrap OTEL tracer with `openinference.instrumentation.OITracer`
3. **TraceConfig Support**: Accept optional `TraceConfig` for customization
4. **Clean Uninstrumentation**: Implement `_uninstrument()` to restore original behavior
5. **Attribute Extraction**: Separate request/response attribute extraction logic
6. **Context Preservation**: Maintain OpenTelemetry context for proper span hierarchy

## Pipecat Architecture Analysis

### Core Architecture Overview

Pipecat is built on a **frame-based processing model** where:
- All data flows through the pipeline as `Frame` objects
- Processors are linked sequentially and process frames asynchronously
- Frames can flow both downstream (source → sink) and upstream (sink → source)
- System frames have priority over data frames

### Base Classes - Detailed Analysis

#### 1. FrameProcessor (`src/pipecat/processors/frame_processor.py`)

**Inheritance**: `FrameProcessor` extends `BaseObject`

**Key Methods for Instrumentation**:
- `__init__(*, name, enable_direct_mode, metrics, **kwargs)`: Initialization hook
- `process_frame(frame, direction)`: Main frame processing dispatcher
- `queue_frame(frame, direction, callback)`: Frame queueing with cancellation support
- `push_frame(frame, direction)`: Pushes frames to next/previous processor
- `setup(setup)` / `cleanup()`: Lifecycle management

**Event Handlers Available**:
- `on_before_process_frame`: Before frame processing
- `on_after_process_frame`: After frame processing
- `on_before_push_frame`: Before pushing to next processor
- `on_after_push_frame`: After pushing to next processor

**Instrumentation Strategy**: We can hook into the event handlers to create spans around frame processing.

#### 2. Pipeline (`src/pipecat/pipeline/pipeline.py`)

**Inheritance**: Compound `FrameProcessor`

**Key Components**:
- `__init__(processors, *, source, sink)`: Accepts list of processors and links them
- `process_frame(frame, direction)`: Routes frames through source/sink
- `processors_with_metrics`: Returns processors that support metrics
- `_link_processors()`: Connects processors sequentially

**Instrumentation Strategy**: Pipeline acts as a container; we'll primarily instrument individual processors rather than the pipeline itself.

#### 3. PipelineTask (`src/pipecat/pipeline/task.py`)

**Inheritance**: Extends `BasePipelineTask`

**Key Parameters**:
- `pipeline`: The frame processor pipeline
- `observers`: List of `BaseObserver` instances for monitoring
- `enable_turn_tracking`: Whether to enable turn tracking (default: True)
- `enable_tracing`: Whether to enable tracing (default: False)
- `conversation_id`: Optional conversation identifier

**Observer Management**:
- `add_observer(observer)`: Add observer at runtime
- `remove_observer(observer)`: Remove observer
- `turn_tracking_observer`: Access to turn tracking instance

**Event Handlers**:
- `on_pipeline_started`, `on_pipeline_finished`, `on_pipeline_error`
- `on_frame_reached_upstream`, `on_frame_reached_downstream`
- `on_idle_timeout`

**Instrumentation Strategy**: This is our **primary injection point**. We'll wrap `PipelineTask.__init__` to automatically inject our `OpenInferenceObserver`.

#### 4. BaseObserver (`src/pipecat/observers/base_observer.py`)

**Class Definition**:
```python
class BaseObserver(BaseObject):
    async def on_process_frame(self, data: FrameProcessed):
        """Handle frame being processed by a processor"""
        pass

    async def on_push_frame(self, data: FramePushed):
        """Handle frame being pushed between processors"""
        pass
```

**Event Data Classes**:
```python
@dataclass
class FramePushed:
    source: FrameProcessor
    destination: FrameProcessor
    frame: Frame
    direction: FrameDirection
    timestamp: int
```

**Instrumentation Strategy**: We'll create `OpenInferenceObserver(BaseObserver)` to capture all frame flows and create appropriate spans.

#### 5. Service Base Classes

##### LLMService (`src/pipecat/services/llm_service.py`)

**Inheritance**: `LLMService` extends `AIService`

**Key Methods**:
- `process_frame(frame, direction)`: Handles LLM-related frames
- `run_function_calls()`: Executes function calls from LLM
- `register_function()`, `unregister_function()`: Function call management
- `get_llm_adapter()`: Returns adapter for LLM communication

**Detection Pattern**:
```python
isinstance(processor, LLMService)
```

**Provider Detection**: Check `processor.__class__.__module__`:
- `pipecat.services.openai.llm` → provider: "openai"
- `pipecat.services.anthropic.llm` → provider: "anthropic"
- etc.

##### TTSService (`src/pipecat/services/tts_service.py`)

**Inheritance**: `TTSService` extends `AIService`

**Key Methods**:
- `_process_text_frame(frame)`: Handles incoming text
- `run_tts(text)`: **Abstract method** - subclasses implement text-to-audio conversion
- `_push_tts_frames()`: Applies filters and manages audio output

**Processing Pipeline**:
1. Receives `TextFrame` or `TTSSpeakFrame`
2. Optional text aggregation (sentence grouping)
3. Text filtering
4. `run_tts()` call → generates audio frames
5. Emits `TTSAudioRawFrame` downstream

**Detection Pattern**:
```python
isinstance(processor, TTSService)
```

##### STTService

**Pattern**: Similar to TTSService, processes audio → text

**Detection Pattern**:
```python
isinstance(processor, STTService)
```

### Service Provider Architecture

Pipecat supports **61+ service providers** organized as:
```
src/pipecat/services/
├── openai/          # OpenAI LLM, TTS, STT
├── anthropic/       # Claude LLM
├── elevenlabs/      # ElevenLabs TTS
├── deepgram/        # Deepgram STT
├── cartesia/        # Cartesia TTS
└── ... (58 more providers)
```

**Provider Detection Strategy**:
```python
def get_provider_from_service(service: FrameProcessor) -> str:
    module = service.__class__.__module__
    # e.g., "pipecat.services.openai.llm" → "openai"
    parts = module.split('.')
    if len(parts) >= 3 and parts[0] == 'pipecat' and parts[1] == 'services':
        return parts[2]
    return "unknown"
```

### Potential Instrumentation Strategies

#### Option A: Observer-Based Instrumentation (Recommended)
**Advantages:**
- Uses Pipecat's native extension point (`BaseObserver`)
- Non-invasive, works with any service implementation
- Can capture all frame types and pipeline events
- Aligns with LangChain/LlamaIndex patterns (using framework hooks)

**Implementation:**
- Create `OpenInferenceObserver` extending `BaseObserver`
- Register with `PipelineTask` observers
- Hook into frame events: `on_push_frame`
- Use turn tracking events for conversation-level spans

#### Option B: Service Wrapper Pattern
**Advantages:**
- More direct control over span lifecycle
- Can wrap specific service methods
- Similar to OpenAI instrumentation pattern

**Disadvantages:**
- Requires wrapping multiple service base classes
- More invasive, brittle to Pipecat changes
- Doesn't generalize well across providers

#### Option C: Hybrid Approach (Best of Both Worlds)
**Advantages:**
- Observer for pipeline-level and conversation spans
- Selective wrapping for critical service methods
- Captures both high-level flow and detailed service metrics

**Implementation:**
- Observer for conversation/turn/pipeline spans
- Wrap `FrameProcessor.process_frame()` for detailed tracing
- Special handling for LLM/TTS/STT service types

## Recommended Implementation Plan

## Integration Strategy: No Pipecat Changes Required

### Key Design Principle: External Observer Pattern

**All logic stays in the OpenInference package** - we do not need to modify Pipecat itself. This works because:

1. **BaseObserver is Public API**: Pipecat's `BaseObserver` is designed for external extensions
2. **PipelineTask Accepts Observers**: Tasks can be initialized with custom observers
3. **Dynamic Registration**: `task.add_observer(observer)` works at runtime

### Implementation Approaches

#### Approach 1: Automatic Injection (Recommended)

Wrap `PipelineTask.__init__` to automatically inject our observer:

```python
# All code in openinference-instrumentation-pipecat package
from pipecat.pipeline.task import PipelineTask
from pipecat.observers.base_observer import BaseObserver

class OpenInferenceObserver(BaseObserver):
    """Our observer - entirely in OpenInference package"""
    def __init__(self, tracer: OITracer, config: TraceConfig):
        super().__init__()
        self._tracer = tracer
        self._config = config
        self._span_handler = _SpanHandler(tracer)

    async def on_push_frame(self, data: FramePushed):
        # Create spans based on frame type and processors
        await self._span_handler.handle_frame_push(data)

class PipecatInstrumentor(BaseInstrumentor):
    def _instrument(self, **kwargs):
        tracer = OITracer(...)
        self._observer = OpenInferenceObserver(tracer=tracer, config=config)

        # Store original __init__
        self._original_task_init = PipelineTask.__init__

        # Wrap PipelineTask.__init__ to inject our observer
        wrap_function_wrapper(
            module="pipecat.pipeline.task",
            name="PipelineTask.__init__",
            wrapper=_TaskInitWrapper(self._observer)
        )

    def _uninstrument(self, **kwargs):
        # Restore original
        PipelineTask.__init__ = self._original_task_init
        self._observer = None

class _TaskInitWrapper:
    def __init__(self, observer: OpenInferenceObserver):
        self._observer = observer

    def __call__(self, wrapped, instance, args, kwargs):
        # Call original __init__
        wrapped(*args, **kwargs)

        # Inject our observer after initialization
        instance.add_observer(self._observer)
```

**Advantages:**
- **Completely automatic** - users just call `PipecatInstrumentor().instrument()`
- **No application code changes** - works with existing Pipecat code
- **Clean migration** from manual tracing example
- **Consistent with other instrumentations** (OpenAI, LangChain patterns)

**Disadvantages:**
- Wraps framework initialization (slightly invasive, but still using public API)
- One shared observer instance across all tasks (may need thread safety)

#### Approach 2: Manual Observer Registration

Users explicitly add the observer to their tasks:

```python
# User's application code
from openinference.instrumentation.pipecat import PipecatInstrumentor, OpenInferenceObserver

# Instrument (sets up tracer, config)
instrumentor = PipecatInstrumentor()
instrumentor.instrument(tracer_provider=tracer_provider)

# User creates observer and adds it manually
observer = instrumentor.create_observer()  # Factory method
task = PipelineTask(pipeline, observers=[observer])
```

**Advantages:**
- **Simpler implementation** - no monkey-patching needed
- **Explicit control** - users see exactly what's being added
- **Multiple observers** - easy to combine with custom observers
- **Thread-safe** - each task gets its own observer instance

**Disadvantages:**
- **Requires code changes** - users must modify their applications
- **Less automatic** - not as seamless as other instrumentations
- **Migration friction** - harder to adopt

#### Recommended: Hybrid Approach

**Default to automatic injection, but expose observer for manual use:**

```python
# Automatic (default) - most users
from openinference.instrumentation.pipecat import PipecatInstrumentor

PipecatInstrumentor().instrument(tracer_provider=provider)
task = PipelineTask(pipeline)  # Observer auto-injected ✅

# Manual (advanced users) - explicit control
from openinference.instrumentation.pipecat import PipecatInstrumentor, OpenInferenceObserver

instrumentor = PipecatInstrumentor()
instrumentor.instrument(tracer_provider=provider)

# Create observer manually for custom configuration or multiple observers
observer = OpenInferenceObserver.create_from_instrumentor(instrumentor)
custom_observer = MyCustomObserver()
task = PipelineTask(pipeline, observers=[observer, custom_observer])

# Or disable automatic injection
instrumentor.instrument(tracer_provider=provider, auto_inject=False)
observer = instrumentor.create_observer()
task = PipelineTask(pipeline, observers=[observer])
```

**Benefits of Hybrid Approach:**
- **Automatic by default** - seamless instrumentation for most users
- **Manual override** - advanced users can disable auto-injection
- **Multi-observer support** - combine with custom observers
- **Configuration flexibility** - per-task observer configuration when needed

### Thread Safety Considerations

**Challenge**: If we auto-inject a single observer instance, it will be shared across all `PipelineTask` instances.

**Solutions**:

1. **Observer Factory Pattern** (Recommended):
```python
class _TaskInitWrapper:
    def __init__(self, tracer: OITracer, config: TraceConfig):
        self._tracer = tracer
        self._config = config

    def __call__(self, wrapped, instance, args, kwargs):
        wrapped(*args, **kwargs)

        # Create NEW observer instance for each task
        observer = OpenInferenceObserver(
            tracer=self._tracer,
            config=self._config
        )
        instance.add_observer(observer)
```

2. **Thread-Safe Shared Observer**:
```python
class OpenInferenceObserver(BaseObserver):
    def __init__(self, tracer, config):
        self._tracer = tracer
        self._config = config
        self._task_contexts = {}  # task_id -> context
        self._lock = asyncio.Lock()

    async def on_push_frame(self, data):
        task_id = id(data.source._parent_task)  # Get task identifier
        async with self._lock:
            # Handle per-task state safely
            pass
```

**Recommendation**: Use **Observer Factory Pattern** to create one observer per task. This is cleaner, safer, and aligns with the principle that each task represents an independent conversation/session.

### Implementation Summary

**What gets added to Pipecat**: Nothing ✅
**What stays in OpenInference package**: Everything ✅

```
openinference-instrumentation-pipecat/
└── src/openinference/instrumentation/pipecat/
    ├── __init__.py              # PipecatInstrumentor (wraps PipelineTask.__init__)
    ├── _observer.py             # OpenInferenceObserver(BaseObserver)
    ├── _span_handler.py         # Span lifecycle management
    └── _wrapper.py              # _TaskInitWrapper (injection logic)
```

### Phase 1: Core Infrastructure

#### 1.1 Package Structure
```
openinference-instrumentation-pipecat/
├── src/
│   └── openinference/
│       └── instrumentation/
│           └── pipecat/
│               ├── __init__.py              # Main instrumentor
│               ├── _observer.py             # OpenInferenceObserver implementation
│               ├── _span_handler.py         # Span lifecycle management
│               ├── _attributes.py           # Attribute extraction logic
│               ├── _utils.py                # Helper utilities
│               ├── package.py               # Package metadata
│               └── version.py               # Version info
├── tests/
│   └── ...
├── examples/
│   ├── basic_usage.py
│   ├── multi_provider.py
│   └── advanced_tracing.py
└── pyproject.toml
```

#### 1.2 Core Instrumentor Class
```python
class PipecatInstrumentor(BaseInstrumentor):
    """
    An instrumentor for Pipecat voice/text pipelines
    """

    def _instrument(self, **kwargs):
        # Get tracer and config
        tracer = OITracer(...)

        # Strategy: Wrap PipelineTask to inject observer
        wrap_function_wrapper(
            module="pipecat.pipeline.task",
            name="PipelineTask.__init__",
            wrapper=_PipelineTaskInit(tracer=tracer, config=config)
        )

    def _uninstrument(self, **kwargs):
        # Restore original behavior
        pass
```

#### 1.3 OpenInferenceObserver Implementation
```python
class OpenInferenceObserver(BaseObserver):
    """
    Observer that creates OpenInference-compliant spans for Pipecat operations
    """

    def __init__(self, tracer: OITracer, config: TraceConfig):
        super().__init__()
        self._tracer = tracer
        self._config = config
        self._span_handler = _SpanHandler(tracer)

    async def on_push_frame(self, data: FramePushed):
        # Determine frame type and create appropriate span
        # Delegate to _span_handler for lifecycle management
        pass
```

### Phase 2: Span Hierarchy Design

#### 2.1 Span Structure

**Level 1: Session Span** (Optional, based on config)
```
span_name: "pipecat.session"
span_kind: CHAIN
attributes:
  - session.id
  - pipeline.type (voice_agent, text_agent, etc.)
```

**Level 2: Conversation Turn Span**
```
span_name: "pipecat.conversation.turn"
span_kind: CHAIN
attributes:
  - conversation.turn_number
  - conversation.speaker (user, bot)
  - conversation.input (user message)
  - conversation.output (bot message)
  - session.id
```

**Level 3: Pipeline Phase Spans**
```
span_name: "pipecat.stt" / "pipecat.llm" / "pipecat.tts"
span_kind: CHAIN
attributes:
  - service.name (openai, elevenlabs, cartesia, etc.)
  - service.provider
  - model.name
  - input.value
  - output.value
```

**Level 4: Service-Specific Spans**
```
Auto-instrumented spans from provider libraries:
  - OpenAI ChatCompletion (via openinference-instrumentation-openai)
  - Other LLM/TTS/STT spans (if instrumented)
```

#### 2.2 Span Lifecycle Management

**Turn Detection Integration:**
```python
class _SpanHandler:
    def __init__(self, tracer: OITracer):
        self._tracer = tracer
        self._current_turn_span = None
        self._phase_spans = {}  # stt, llm, tts

    def on_turn_started(self, turn_number: int):
        # Create turn span
        self._current_turn_span = self._tracer.start_span(
            name="pipecat.conversation.turn",
            attributes={...}
        )

    def on_turn_ended(self, turn_number: int, duration: float):
        # Finalize turn span
        self._current_turn_span.end()
        self._phase_spans.clear()
```

### Phase 3: Service Detection and Attribution

#### 3.1 Service Type Detection
```python
class _ServiceDetector:
    """Detect service types and extract metadata"""

    def detect_service_type(self, processor: FrameProcessor) -> Optional[str]:
        # Check inheritance hierarchy
        if isinstance(processor, STTService):
            return "stt"
        elif isinstance(processor, LLMService):
            return "llm"
        elif isinstance(processor, TTSService):
            return "tts"
        return None

    def extract_service_metadata(self, service: FrameProcessor) -> Dict[str, Any]:
        # Extract provider, model, etc.
        metadata = {}

        # Common patterns across services
        if hasattr(service, '_model'):
            metadata['model'] = service._model
        if hasattr(service, '__class__'):
            # OpenAILLMService -> provider: openai
            class_name = service.__class__.__name__
            metadata['provider'] = self._extract_provider_from_class(class_name)

        return metadata
```

#### 3.2 Attribute Extraction Strategy

**Frame-Based Attributes:**
```python
class _FrameAttributeExtractor:
    """Extract OpenInference attributes from Pipecat frames"""

    def extract_from_frame(self, frame: Frame) -> Iterator[Tuple[str, Any]]:
        # TranscriptionFrame -> STT output
        if isinstance(frame, TranscriptionFrame):
            yield SpanAttributes.OUTPUT_VALUE, frame.text

        # TextFrame -> LLM/TTS input
        elif isinstance(frame, TextFrame):
            yield SpanAttributes.INPUT_VALUE, frame.text

        # AudioRawFrame -> audio metadata
        elif isinstance(frame, AudioRawFrame):
            yield "audio.sample_rate", frame.sample_rate
            yield "audio.num_channels", frame.num_channels
```

### Phase 4: Context Propagation

#### 4.1 OpenTelemetry Context Integration
```python
class _ContextManager:
    """Manage OpenTelemetry context across async operations"""

    def __init__(self):
        self._turn_contexts = {}

    def attach_turn_context(self, turn_number: int, span: Span):
        # Set span in context for all child operations
        ctx = trace_api.set_span_in_context(span)
        token = context_api.attach(ctx)
        self._turn_contexts[turn_number] = token

    def detach_turn_context(self, turn_number: int):
        if token := self._turn_contexts.pop(turn_number, None):
            context_api.detach(token)
```

#### 4.2 Integration with Existing Instrumentations

**Key Insight**: The OpenAI instrumentation (and others) will automatically:
- Detect the active span context
- Create child spans under the current context
- Use proper OpenInference span kinds (LLM for ChatCompletion)

**Implementation**:
```python
# When LLM service is called, ensure turn span is active
with trace_api.use_span(self._current_turn_span):
    # OpenAI service call happens here
    # OpenAI instrumentation creates LLM span as child
    result = await llm_service.process_frame(frame)
```

### Phase 5: Configuration and Customization

#### 5.1 TraceConfig Options
```python
@dataclass
class PipecatTraceConfig(TraceConfig):
    """Extended trace config for Pipecat-specific options"""

    # Session-level tracing
    enable_session_spans: bool = False

    # Turn-based tracing (default: True)
    enable_turn_spans: bool = True

    # Pipeline phase spans
    enable_stt_spans: bool = True
    enable_llm_spans: bool = True
    enable_tts_spans: bool = True

    # Frame-level tracing (verbose, default: False)
    enable_frame_spans: bool = False

    # Attribute collection
    capture_audio_metadata: bool = True
    capture_frame_timing: bool = True

    # Input/output truncation
    max_input_length: int = 1000
    max_output_length: int = 1000
```

#### 5.2 Usage Example
```python
from openinference.instrumentation.pipecat import PipecatInstrumentor
from openinference.instrumentation import TraceConfig

config = TraceConfig(
    enable_turn_spans=True,
    enable_frame_spans=False,
)

instrumentor = PipecatInstrumentor()
instrumentor.instrument(
    tracer_provider=tracer_provider,
    config=config,
)
```

### Phase 6: Testing Strategy

#### 6.1 Unit Tests
- Test span creation for each frame type
- Verify attribute extraction logic
- Test context propagation
- Validate span hierarchy

#### 6.2 Integration Tests
- Test with OpenAI services
- Test with alternative providers (ElevenLabs, Cartesia)
- Test turn detection integration
- Test with multiple simultaneous sessions

#### 6.3 Example Applications
- Basic voice agent (OpenAI only)
- Multi-provider agent (mixed services)
- Text-based pipeline
- Custom processor pipeline

## Implementation Roadmap

### Milestone 1: Foundation (Week 1-2)
- [ ] Package structure setup
- [ ] Core `PipecatInstrumentor` class
- [ ] Basic observer implementation
- [ ] Unit test framework

### Milestone 2: Observer Integration (Week 3-4)
- [ ] `OpenInferenceObserver` implementation
- [ ] Turn tracking integration
- [ ] Frame event handling
- [ ] Integration tests with example

### Milestone 3: Service Detection (Week 5-6)
- [ ] Service type detection logic
- [ ] Metadata extraction
- [ ] Attribute extractors for common frames
- [ ] Multi-provider testing

### Milestone 4: Context Management (Week 7-8)
- [ ] Context propagation implementation
- [ ] Integration with existing instrumentations (OpenAI, etc.)
- [ ] Async operation handling
- [ ] Streaming response support

### Milestone 5: Configuration & Docs (Week 9-10)
- [ ] TraceConfig implementation
- [ ] Configuration validation
- [ ] Usage documentation
- [ ] Example applications
- [ ] Migration guide from manual tracing

### Milestone 6: Production Readiness (Week 11-12)
- [ ] Performance optimization
- [ ] Error handling and recovery
- [ ] Production example with Arize
- [ ] Release preparation

## Key Design Decisions

### 1. Observer-Based vs Method Wrapping

**Decision**: Use observer pattern as primary mechanism
**Rationale**:
- Aligns with Pipecat's design philosophy
- More maintainable and less fragile
- Works across all service providers
- Similar to LangChain/LlamaIndex approach

### 2. Turn-Based Tracing as Default

**Decision**: Enable turn-based tracing by default
**Rationale**:
- Most intuitive for conversation applications
- Matches current example implementation
- Can be disabled for streaming/pipeline-only use cases

### 3. Integration with Existing Instrumentations

**Decision**: Rely on existing instrumentations (OpenAI, etc.) for service-level spans
**Rationale**:
- Avoid duplicate spans
- Leverage existing attribute extraction logic
- Ensure consistent OpenInference conventions
- Reduce maintenance burden

### 4. Frame-Level Tracing as Opt-In

**Decision**: Disable frame-level tracing by default
**Rationale**:
- Can be very verbose (hundreds of frames per turn)
- Most users want conversation-level visibility
- Can be enabled for debugging

## Migration Path

### From Manual Tracing to Instrumentation

**Current Manual Approach:**
```python
# examples/trace/001-trace.py
import tracing_setup
tracing_setup.setup_arize_tracing()
tracing_setup.set_session_id(session_id)
```

**New Instrumentation Approach:**
```python
# New approach
from openinference.instrumentation.pipecat import PipecatInstrumentor
from arize.otel import register

tracer_provider = register(space_id=..., api_key=...)

instrumentor = PipecatInstrumentor()
instrumentor.instrument(tracer_provider=tracer_provider)

# That's it! Automatic tracing for all pipelines
```

**Benefits:**
- No manual patching required
- Works with any service provider
- Automatic session/turn management
- Configurable span granularity

## Open Questions for Discussion

1. **Session Span Creation**: Should session spans be created automatically or require explicit API calls?
   - Option A: Automatic based on pipeline lifecycle
   - Option B: Explicit `instrumentor.start_session(session_id)`

2. **Frame Processor Wrapping**: Should we also wrap `FrameProcessor.process_frame()` for fine-grained tracing?
   - Pros: More detailed visibility
   - Cons: Performance overhead, span explosion

3. **Service Provider Detection**: How to handle custom services not following naming conventions?
   - Option A: Configuration-based service mapping
   - Option B: Service registration API

4. **Backward Compatibility**: Should we maintain the manual tracing API for advanced use cases?
   - Option A: Deprecate and migrate
   - Option B: Keep as alternative approach

## Next Steps

1. **Review this plan** with the team
2. **Analyze Pipecat base classes** in detail (next task)
3. **Create minimal proof-of-concept** with observer pattern
4. **Validate span hierarchy** with real application
5. **Iterate on design** based on feedback

## References

- [OpenInference Semantic Conventions](https://github.com/Arize-ai/openinference/tree/main/spec)
- [OpenTelemetry Instrumentation Guide](https://opentelemetry.io/docs/instrumentation/python/)
- [Pipecat Documentation](https://docs.pipecat.ai/)
- Current Example: [examples/trace/tracing_setup.py](examples/trace/tracing_setup.py)
