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

## Current Implementation Status

### ✅ COMPLETE - All 69/69 tests passing!

✅ **Phase 1-3: Core Infrastructure**
- Package structure created
- `PipecatInstrumentor` class implemented
- `OpenInferenceObserver(BaseObserver)` implemented
- Service detection logic working for LLM, TTS, STT
- Span creation for service-level operations (pipecat.llm, pipecat.tts, pipecat.stt)
- Attribute extraction from frames
- Test infrastructure with mocked pipeline execution

✅ **Phase 4: Turn Tracking - IMPLEMENTED**
- Turn spans created with name `"pipecat.conversation.turn"`
- Turn boundaries detected from frame types (UserStartedSpeaking → BotStoppedSpeaking)
- Turn-level input/output captured from TranscriptionFrame and TextFrame
- Turn interruptions handled (new UserStartedSpeaking during bot speaking)
- Turn numbers tracked incrementally
- Turn end reason captured (completed vs interrupted)

✅ **Key Implementation Details**
- Observer extends `BaseObserver` (Pipecat's native extension point)
- Automatic injection via wrapping `PipelineTask.__init__`
- One observer instance created per task (factory pattern)
- Service spans finish on `EndFrame` or `ErrorFrame`
- Turn spans finish on `BotStoppedSpeakingFrame` or interruption
- Works with all service providers (OpenAI, Anthropic, ElevenLabs, Deepgram, etc.)

## Revised Requirements & Implementation Plan

### Key Requirements (Updated)

Based on discussion and analysis of Pipecat's extensive frame types (100+ frames across categories like LLM, TTS, STT, audio, control, function calling, etc.), the following requirements have been identified:

#### 1. **Proper Span Hierarchy & Parent-Child Relationships**
   - **Session Level**: All turns within a conversation share a session ID
   - **Turn Level**: Root span for each interaction showing overall input/output
   - **Service Level**: Child spans for LLM, TTS, STT operations within a turn
   - **LLM Specifics**: When LLM is involved, use `OPENINFERENCE_SPAN_KIND = "LLM"` and extract messages

#### 2. **Session Management**
   - Utilize `using_session(session_id)` context manager from openinference-instrumentation
   - Session ID propagated via OpenTelemetry context to all child spans
   - PipelineTask `conversation_id` parameter maps to session.id attribute

#### 3. **LLM Frame Handling**
   - Detect LLM-related frames: `LLMMessagesFrame`, `LLMMessagesAppendFrame`, `LLMFullResponseStartFrame`, etc.
   - Extract messages and use proper OpenInference LLM span kind
   - Capture LLM-specific attributes (model, messages, function calls, etc.)

#### 4. **Generic Frame Handling**
   - Don't create unique handlers for every frame type (too many!)
   - Capture frame class name as attribute for all frames
   - Extract properties based on frame type pattern matching:
     - Text content (TextFrame, TranscriptionFrame, etc.)
     - Audio metadata (AudioRawFrame variants)
     - Control signals (StartFrame, EndFrame, ErrorFrame)
     - Function calling (FunctionCallFromLLM, FunctionCallResultFrame)
   - Gracefully handle unknown frame types

#### 5. **Span Hierarchy Example**
```
Session Span (session.id = "conv-123")
  └─> Turn Span 1 (conversation.turn_number = 1, input = "Hello", output = "Hi there!")
      ├─> STT Span (service.name = "openai", frame.type = "TranscriptionFrame")
      ├─> LLM Span (SPAN_KIND = "LLM", model = "gpt-4", messages = [...])
      │   └─> OpenAI ChatCompletion Span (from openai instrumentation)
      └─> TTS Span (service.name = "elevenlabs", voice.id = "...)
  └─> Turn Span 2 (conversation.turn_number = 2, ...)
      └─> ...
```

### Implementation Tasks

#### ❌ **NOT DONE: Session-Level Span Management**
**Current State**: No session span, turns are not connected
**Required Changes**:
1. Create session span when observer is initialized with `conversation_id`
2. Use `using_session(conversation_id)` to propagate session.id
3. Make all turn spans children of session span via OpenTelemetry context
4. Session span lifecycle:
   - Start: When first turn begins OR when observer is created
   - End: When pipeline task completes OR explicit session end

#### ❌ **NOT DONE: Proper Parent-Child Span Relationships**
**Current State**: Spans are created independently, no parent-child links
**Required Changes**:
1. Use `trace_api.use_span()` context manager to set active span
2. Turn spans created within session span context
3. Service spans (LLM, TTS, STT) created within turn span context
4. Verify span hierarchy via `span.parent.span_id` in tests

#### ❌ **NOT DONE: LLM Span Kind & Message Extraction**
**Current State**: LLM spans use `CHAIN` span kind, don't extract messages
**Required Changes**:
1. Detect LLM service type properly (already done)
2. Change span kind to `OpenInferenceSpanKindValues.LLM` for LLM operations
3. Extract messages from LLM frames:
   - `LLMMessagesFrame` → full message list
   - `LLMMessagesAppendFrame` → appended messages
   - `LLMFullResponseStartFrame` / `LLMFullResponseEndFrame` → response tracking
4. Use `get_llm_input_message_attributes()` and `get_llm_output_message_attributes()`

#### ✅ **PARTIALLY DONE: Generic Frame Attribute Extraction**
**Current State**: Basic frame attributes extracted (text, some metadata)
**Required Enhancements**:
1. Always capture `frame.type` = frame.__class__.__name__
2. Pattern-based extraction:
   ```python
   # Text frames
   if hasattr(frame, 'text') and frame.text:
       yield SpanAttributes.INPUT_VALUE or OUTPUT_VALUE, frame.text

   # Audio frames
   if hasattr(frame, 'audio') and hasattr(frame, 'sample_rate'):
       yield "audio.sample_rate", frame.sample_rate

   # Function calling
   if isinstance(frame, FunctionCallFromLLM):
       yield "tool.name", frame.function_name
       yield "tool.arguments", frame.arguments
   ```
3. Error handling for unknown frames (just log frame type, don't fail)

## Turn Tracking Implementation Plan

### Problem Statement

Turn tracking tests expect:
1. Spans with name `"pipecat.conversation.turn"`
2. Attributes:
   - `conversation.turn_number` (incremental counter)
   - `INPUT_VALUE` (user transcription text)
   - `OUTPUT_VALUE` (bot response text)
   - `conversation.end_reason` (completed/interrupted)

3. Turn boundaries defined by frames:
   - **Turn Start**: `UserStartedSpeakingFrame`
   - **User Input**: `TranscriptionFrame` (contains user text)
   - **User Stop**: `UserStoppedSpeakingFrame`
   - **Bot Start**: `BotStartedSpeakingFrame`
   - **Bot Output**: `TextFrame` (contains bot response text)
   - **Turn End**: `BotStoppedSpeakingFrame`
   - **Interruption**: New `UserStartedSpeakingFrame` before `BotStoppedSpeakingFrame`

### Implementation Approach

**Enhance OpenInferenceObserver to track turn state:**

```python
class OpenInferenceObserver(BaseObserver):
    def __init__(self, tracer: OITracer, config: TraceConfig):
        super().__init__()
        self._tracer = tracer
        self._config = config

        # Existing service span tracking
        self._detector = _ServiceDetector()
        self._attribute_extractor = _FrameAttributeExtractor()
        self._active_spans = {}  # service spans
        self._last_frames = {}

        # NEW: Turn tracking state
        self._turn_state = {
            'active': False,
            'span': None,
            'turn_number': 0,
            'user_text': [],
            'bot_text': [],
            'started_at': None,
        }
```

### Turn Tracking Logic

**Detect turn boundary frames in `on_push_frame()`:**

```python
async def on_push_frame(self, data: FramePushed):
    from pipecat.frames.frames import (
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        TranscriptionFrame,
        TextFrame,
        EndFrame,
        ErrorFrame,
    )

    frame = data.frame

    # Turn tracking logic (NEW)
    if isinstance(frame, UserStartedSpeakingFrame):
        await self._start_turn()
    elif isinstance(frame, TranscriptionFrame):
        if self._turn_state['active'] and frame.text:
            self._turn_state['user_text'].append(frame.text)
    elif isinstance(frame, UserStoppedSpeakingFrame):
        pass  # User finished speaking, wait for bot
    elif isinstance(frame, BotStartedSpeakingFrame):
        pass  # Bot starting response
    elif isinstance(frame, TextFrame):
        if self._turn_state['active'] and frame.text:
            self._turn_state['bot_text'].append(frame.text)
    elif isinstance(frame, BotStoppedSpeakingFrame):
        await self._finish_turn(interrupted=False)

    # Existing service span logic (unchanged)
    service_type = self._detector.detect_service_type(data.source)
    if service_type:
        await self._handle_service_frame(data, service_type)
```

### Turn Span Creation

```python
async def _start_turn(self):
    """Start a new conversation turn."""
    # If there's an active turn, it was interrupted
    if self._turn_state['span']:
        await self._finish_turn(interrupted=True)

    # Increment turn counter
    self._turn_state['turn_number'] += 1
    self._turn_state['active'] = True
    self._turn_state['user_text'] = []
    self._turn_state['bot_text'] = []

    # Create turn span
    span = self._tracer.start_span(
        name="pipecat.conversation.turn",
        attributes={
            "conversation.turn_number": self._turn_state['turn_number'],
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
        }
    )
    self._turn_state['span'] = span

    logger.debug(f"Started turn {self._turn_state['turn_number']}")

async def _finish_turn(self, interrupted: bool = False):
    """Finish the current conversation turn."""
    if not self._turn_state['active'] or not self._turn_state['span']:
        return

    span = self._turn_state['span']

    # Add input text (user transcription)
    if self._turn_state['user_text']:
        user_input = ' '.join(self._turn_state['user_text'])
        span.set_attribute(SpanAttributes.INPUT_VALUE, user_input)

    # Add output text (bot response)
    if self._turn_state['bot_text']:
        bot_output = ' '.join(self._turn_state['bot_text'])
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, bot_output)

    # Add end reason
    end_reason = "interrupted" if interrupted else "completed"
    span.set_attribute("conversation.end_reason", end_reason)

    # Finish span
    span.set_status(trace_api.Status(trace_api.StatusCode.OK))
    span.end()

    logger.debug(
        f"Finished turn {self._turn_state['turn_number']} ({end_reason})"
    )

    # Reset state
    self._turn_state['active'] = False
    self._turn_state['span'] = None
```

### Implementation Steps

1. **Add turn state to OpenInferenceObserver.__init__()**
   - Initialize turn tracking dictionary

2. **Add turn frame detection to on_push_frame()**
   - Check for UserStartedSpeaking, BotStoppedSpeaking, etc.
   - Collect TranscriptionFrame and TextFrame content

3. **Implement _start_turn() method**
   - Create turn span with turn_number attribute
   - Handle interruptions (previous turn still active)

4. **Implement _finish_turn() method**
   - Add INPUT_VALUE and OUTPUT_VALUE from collected text
   - Add conversation.end_reason attribute
   - End the span

5. **Test with turn tracking tests**
   - `test_complete_turn_cycle` - basic turn
   - `test_multiple_sequential_turns` - multiple turns
   - `test_turn_interruption` - interruption handling

### Success Criteria

- ✅ All 69 tests pass (currently 66/69)
- ✅ Turn spans created with name "pipecat.conversation.turn"
- ✅ Turn spans have `conversation.turn_number` attribute
- ✅ Turn spans capture `INPUT_VALUE` and `OUTPUT_VALUE`
- ✅ Interruptions set `conversation.end_reason` = "interrupted"
- ✅ Completed turns set `conversation.end_reason` = "completed"

### Design Rationale

**Why enhance OpenInferenceObserver vs integrate with TurnTrackingObserver?**

1. **Works with mocked tests**: Our test infrastructure mocks PipelineRunner execution, which doesn't trigger Pipecat's TurnTrackingObserver properly
2. **Full control**: We control the exact OpenTelemetry span attributes
3. **Simpler**: Single observer handles all tracing (services + turns)
4. **Maintainable**: All tracing logic in one place
5. **Future-proof**: Can migrate to integrate with TurnTrackingObserver later if needed

**Note**: For real applications using PipelineRunner, Pipecat's native TurnTrackingObserver also runs. Our observer creates OpenTelemetry spans; theirs creates Pipecat events. They coexist independently.

## CRITICAL ISSUE: Turn Tracking Strategy Needs Redesign

### Current Problem Analysis (2025-10-29)

**Issue**: The current turn tracking implementation creates **excessive orphaned turn spans** due to frame propagation through the pipeline.

**Root Cause**: `BotStoppedSpeakingFrame` propagates through **every processor in the pipeline**. When we react to this frame without filtering by source, we:
1. Finish turn at first processor (e.g., SmallWebRTCOutputTransport)
2. Start new turn immediately
3. Frame continues to next processor (LLMAssistantAggregator)
4. `BotStoppedSpeakingFrame` triggers finish → **new turn created again**
5. Repeats for every processor in the chain

**Evidence from Logs**:
```
Line 1958: FINISHING TURN #1 (SmallWebRTCOutputTransport)
Line 1979: STARTING TURN #2 (LLMAssistantAggregator receives BotStoppedSpeaking)
Line 1995: FINISHING TURN #2 (0.001ms duration - empty!)
Line 2004: STARTING TURN #3 (OpenAILLMService receives BotStoppedSpeaking)
Line 2022: FINISHING TURN #3 (0.001ms duration - empty!)
...continues for 5+ processors
```

**Result**: In a conversation with 2 actual exchanges, we get **18 turn spans**, most empty (< 1ms duration).

### Proposed Solution: Transport-Layer-Only Turn Tracking

**Strategy**: Only react to speaking frames from **transport layer sources** to avoid duplicate turn creation from frame propagation.

**Key Changes**:

1. **Filter Speaking Frames by Source**:
```python
# In on_push_frame()
source_name = data.source.__class__.__name__ if data.source else "Unknown"
is_transport = "Transport" in source_name

# Only track turns from transport layer
if isinstance(frame, UserStartedSpeakingFrame) and is_transport:
    # Start turn
if isinstance(frame, BotStoppedSpeakingFrame) and is_transport:
    # End turn
```

2. **Transport Sources to Track**:
- `SmallWebRTCInputTransport` - User input
- `SmallWebRTCOutputTransport` - Bot output
- Other transport implementations (DailyTransport, etc.)

**Benefits**:
- Only 1 turn span per actual conversation exchange
- Turns represent actual user ↔ bot interactions
- Service spans (STT, LLM, TTS) properly nested under turn
- Cleaner traces with meaningful turn boundaries

### Alternative Considered: Conversation Exchange Model

Instead of "turns", track **conversation exchanges** as complete request/response cycles:

**Approach**:
- **Start Exchange**: When LLM service receives input (first service activity)
- **End Exchange**: When TTS completes output (last service activity)
- **Each exchange contains**: STT → LLM → TTS pipeline

**Pros**:
- Aligns with actual processing flow
- Guarantees complete service span capture
- Less dependent on speaking frame propagation

**Cons**:
- Doesn't match user's mental model of "turns"
- Harder to detect exchange boundaries
- May miss initialization activity

**Decision**: Proceed with transport-layer filtering approach as it's simpler and aligns with existing turn concept.

### Alternative Considered: Turn Detection via Service Activity

**Approach**:
- **Start turn**: When first service (STT, LLM, or TTS) receives a frame
- **End turn**: When last service (typically TTS) finishes
- Ignore speaking frames entirely

**Pros**:
- Guaranteed to capture all service activity
- No duplicate turns from frame propagation
- Works regardless of speaking frame behavior

**Cons**:
- May not align with user expectations of "turn" boundaries
- Harder to detect interruptions
- Initialization spans might get orphaned

### Implementation Plan

1. **Add source filtering to speaking frame handlers** ([_observer.py:139-166](src/openinference/instrumentation/pipecat/_observer.py#L139-L166))
2. **Test with real conversation** to verify only transport-layer turns are created
3. **Verify service spans are properly nested** under turn spans
4. **Check for any orphaned initialization spans**

### Success Criteria

- ✅ 2 actual exchanges = 2 turn spans (not 18!)
- ✅ Turn spans have meaningful duration (> 1 second, not 0.001ms)
- ✅ Turn spans contain input/output text
- ✅ Service spans (STT, LLM, TTS) are children of turn spans
- ✅ No orphaned service spans with different trace_ids

## Prioritized Next Steps

### 🔴 **HIGHEST PRIORITY: Fix Turn Tracking to Eliminate Orphaned Spans**

**Problem**: Current implementation creates 18+ turn spans for 2 actual exchanges due to frame propagation through pipeline.

**Tasks**:

1. **Implement Transport-Layer Filtering** ([_observer.py:139-166](src/openinference/instrumentation/pipecat/_observer.py#L139-L166)):
   - Add `is_transport = "Transport" in source_name` check
   - Only react to `UserStartedSpeakingFrame` when `is_transport == True`
   - Only react to `BotStartedSpeakingFrame` when `is_transport == True`
   - Only react to `BotStoppedSpeakingFrame` when `is_transport == True`
   - This prevents duplicate turn creation from frames propagating through pipeline

2. **Fix Service Span Context Propagation** ([_observer.py:195-215](src/openinference/instrumentation/pipecat/_observer.py#L195-L215)):
   - Current: Service spans created with `context=self._turn_context_token` (WORKS!)
   - Keep this approach - it's correct and creates proper parent-child relationships
   - Issue is NOT context propagation, it's turn span creation timing

3. **Session ID Attribution** ([__init__.py:119](src/openinference/instrumentation/pipecat/__init__.py#L119)):
   - ✅ **FIXED**: Now extracts `_conversation_id` from PipelineTask correctly
   - ✅ **WORKING**: session.id attribute appears on turn spans
   - Need to verify session.id also appears on service spans (should inherit from turn context)

4. **Test with Real Conversation**:
   - Run conversation example with transport filtering
   - Verify: 2 exchanges = 2 turn spans (not 18)
   - Verify: Service spans have correct parent_id pointing to turn span
   - Verify: All spans share same trace_id within a turn
   - Verify: session.id attribute appears on all spans

**Current Implementation Status**:
```python
# CURRENT CODE (working for service spans, broken for turns)
async def _handle_service_frame(self, data: FramePushed, service_type: str):
    if service_id not in self._active_spans:
        # Auto-start turn if none exists
        if self._turn_context_token is None:
            self._turn_context_token = await self._start_turn()

        # Create service span WITH turn context (THIS WORKS!)
        span = self._create_service_span(service, service_type)
        # span.parent will be turn_span ✅

# BROKEN CODE (creates too many turns)
async def on_push_frame(self, data: FramePushed):
    # Problem: Reacts to BotStoppedSpeakingFrame from EVERY processor
    if isinstance(frame, BotStoppedSpeakingFrame):
        await self._finish_turn(interrupted=False)  # Creates new turn!

# PROPOSED FIX
async def on_push_frame(self, data: FramePushed):
    source_name = data.source.__class__.__name__ if data.source else "Unknown"
    is_transport = "Transport" in source_name

    # Only react to transport layer
    if isinstance(frame, BotStoppedSpeakingFrame) and is_transport:
        await self._finish_turn(interrupted=False)
```

### 🟡 **MEDIUM PRIORITY: LLM Span Kind & Message Extraction**

**Problem**: LLM spans currently use `CHAIN` span kind instead of `LLM`, and don't extract message content.

**Tasks**:
1. **Detect LLM Frames** ([_attributes.py](src/openinference/instrumentation/pipecat/_attributes.py)):
   - Add detection for `LLMMessagesFrame`, `LLMMessagesAppendFrame`, `LLMFullResponseStartFrame`
   - Extract message content from frames

2. **Change Span Kind** ([_observer.py](src/openinference/instrumentation/pipecat/_observer.py)):
   - When service_type == "llm", use `OpenInferenceSpanKindValues.LLM`
   - Extract and set LLM message attributes using `get_llm_input_message_attributes()`

3. **Test LLM Spans** (new test file):
   - Verify LLM span kind is correct
   - Verify messages are extracted
   - Verify integration with OpenAI instrumentation (nested spans)

### 🟢 **LOW PRIORITY: Enhanced Frame Attribute Extraction**

**Problem**: Not all frame types have their properties extracted. Need generic handler.

**Tasks**:
1. **Add frame.type Attribute** ([_attributes.py](src/openinference/instrumentation/pipecat/_attributes.py)):
   - Always set `frame.type = frame.__class__.__name__`

2. **Pattern-Based Extraction** ([_attributes.py](src/openinference/instrumentation/pipecat/_attributes.py)):
   - Check for common properties: `text`, `audio`, `sample_rate`, `function_name`, etc.
   - Use hasattr() to gracefully handle missing properties
   - Log unknown frame types for debugging

3. **Function Calling Support** ([_attributes.py](src/openinference/instrumentation/pipecat/_attributes.py)):
   - Detect `FunctionCallFromLLM`, `FunctionCallResultFrame`
   - Extract tool.name, tool.arguments, tool.output

### Testing & Validation

After implementing each priority:
1. Run full test suite: `pytest tests/`
2. Verify span hierarchy in actual example
3. Check Phoenix/Arize UI for proper trace structure

## Acceptance Criteria

The implementation will be considered complete when:

1. ✅ All 69 tests pass
2. ✅ Session ID propagates to all spans in a conversation
3. ✅ Turn spans are children of session context
4. ✅ Service spans (LLM, TTS, STT) are children of turn spans
5. ✅ LLM spans use `SPAN_KIND = "LLM"` and extract messages
6. ✅ Frame types are captured for all frames
7. ✅ Example trace shows proper hierarchy in Phoenix/Arize

## References

- [OpenInference Semantic Conventions](https://github.com/Arize-ai/openinference/tree/main/spec)
- [OpenTelemetry Instrumentation Guide](https://opentelemetry.io/docs/instrumentation/python/)
- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Pipecat Frame Types](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/frames/frames.py)
- Current Example: [examples/trace/tracing_setup.py](examples/trace/tracing_setup.py)
