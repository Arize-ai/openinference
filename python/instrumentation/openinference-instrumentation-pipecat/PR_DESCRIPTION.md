# Add OpenInference Instrumentation for Pipecat

This PR implements comprehensive OpenTelemetry tracing for Pipecat voice agents using OpenInference semantic conventions, enabling production-ready observability for voice AI applications.

## Overview

Adds automatic instrumentation for Pipecat pipelines that captures:
- **Turn-level spans**: Complete conversation exchanges with user input/output
- **Service-level spans**: Individual LLM, TTS, and STT operations
- **Proper span hierarchy**: Service spans nested under turn spans with correct parent-child relationships
- **Rich attributes**: Model names, providers, token counts, latency metrics, and full conversation text

## Key Features

### 1. Automatic Instrumentation via Wrapper Pattern

The instrumentor wraps `PipelineTask.__init__` to automatically inject an observer into every task:

```python
from openinference.instrumentation.pipecat import PipecatInstrumentor
from arize.otel import register

tracer_provider = register(
    space_id=os.getenv("ARIZE_SPACE_ID"),
    api_key=os.getenv("ARIZE_API_KEY"),
    project_name=os.getenv("ARIZE_PROJECT_NAME"),
)

PipecatInstrumentor().instrument(
    tracer_provider=tracer_provider,
    debug_log_filename="debug.log"  # Optional
)
```

No code changes needed in your pipeline - just instrument once and all `PipelineTask` instances get automatic tracing.

### 2. Turn Tracking with Service-Based Boundaries

Implements intelligent turn tracking using service-specific frames to avoid duplication:
- **Start turn**: When STT produces `TranscriptionFrame` (user input arrives)
- **End turn**: When LLM produces `LLMFullResponseEndFrame` (semantic completion)
- **Fallback**: `BotStoppedSpeakingFrame` for TTS-only responses

This approach ensures one turn span per actual conversation exchange, avoiding the 18+ orphaned spans that would occur from naive frame propagation handling.

### 3. Comprehensive Text Capture

Captures both user input and bot responses by collecting text from:
- **`TranscriptionFrame`**: User speech-to-text output
- **`LLMTextFrame`**: LLM streaming responses (token-by-token)
- **`TextFrame`**: TTS input text

All text chunks are aggregated throughout the turn and added to span attributes on completion.

### 4. Multi-Provider Service Detection

Automatically detects and attributes service types and providers:
- **LLM Services**: OpenAI, Anthropic (sets `llm.provider`, `llm.model_name`)
- **TTS Services**: OpenAI, ElevenLabs, Cartesia (sets `audio.voice`, `audio.voice_id`)
- **STT Services**: OpenAI, Deepgram, Cartesia
- **Generic detection**: Works with any service inheriting from Pipecat base classes

Sets `service.name` to the actual service class name for unique identification.

### 5. Session Tracking

Automatically extracts `conversation_id` from `PipelineTask` and sets as `session.id` attribute on all spans, enabling conversation-level filtering in observability platforms.

## Implementation Details

### Core Components

**`PipecatInstrumentor`** ([__init__.py](src/openinference/instrumentation/pipecat/__init__.py))
- Wraps `PipelineTask.__init__` using `wrapt`
- Injects `OpenInferenceObserver` into each task
- Supports optional `debug_log_filename` parameter for detailed frame logging
- Thread-safe: creates separate observer instance per task

**`OpenInferenceObserver`** ([_observer.py](src/openinference/instrumentation/pipecat/_observer.py))
- Implements Pipecat's `BaseObserver` interface
- Listens to `on_push_frame` events
- Creates turn spans and service spans with proper OpenTelemetry context propagation
- Tracks turn state: active turn, user text, bot text, speaking status
- Auto-starts turns when first service activity detected

**`_ServiceDetector`** ([_service_detector.py](src/openinference/instrumentation/pipecat/_service_detector.py))
- Pattern-based detection using `isinstance()` checks on Pipecat base classes
- Extracts metadata: model names, voice IDs, provider names
- Supports `LLMService`, `TTSService`, `STTService` and their subclasses

**`_FrameAttributeExtractor`** ([_attributes.py](src/openinference/instrumentation/pipecat/_attributes.py))
- Extracts OpenInference-compliant attributes from Pipecat frames
- Handles 100+ frame types via duck-typing patterns
- Captures: LLM messages, audio metadata, timestamps, errors, tool calls

### Span Hierarchy

```
pipecat.conversation.turn (trace_id: abc123)
├── pipecat.stt (parent_id: turn_span_id, trace_id: abc123)
├── pipecat.llm (parent_id: turn_span_id, trace_id: abc123)
└── pipecat.tts (parent_id: turn_span_id, trace_id: abc123)
```

All spans within a turn share the same `trace_id` and have `session.id` attribute set.

### Context Propagation

Service spans are created with the turn span's context:
```python
span = self._tracer.start_span(
    name=f"pipecat.{service_type}",
    context=self._turn_context_token,  # Links to turn span
)
```

This ensures proper parent-child relationships and enables distributed tracing.

## Testing

### Test Coverage

**69 tests** covering:

1. **Instrumentor Basics** (`test_instrumentor.py`):
   - Initialization, instrumentation, uninstrumentation
   - Observer injection into tasks
   - Singleton behavior
   - Configuration handling

2. **Turn Tracking** (`test_turn_tracking.py`):
   - Turn creation on user/bot speech
   - Multiple sequential turns
   - Turn interruption handling
   - Input/output text capture
   - Session ID attribution
   - Turn span hierarchy

3. **Service Detection** (`test_service_detection.py`):
   - LLM/TTS/STT service type detection
   - Multi-provider detection (OpenAI, Anthropic, ElevenLabs, Deepgram)
   - Metadata extraction (models, voices, providers)
   - Custom service inheritance

4. **Provider Spans** (`test_provider_spans.py`):
   - Span creation for different providers
   - Correct span attributes per service type
   - Input/output capture for each service
   - Mixed provider pipelines
   - Provider-specific attributes (model names, voice IDs)

### Mock Infrastructure

Comprehensive mocks in `conftest.py`:
- Mock LLM/TTS/STT services with configurable metadata
- Helper functions for running pipeline tasks
- Span extraction and assertion utilities
- Support for multiple provider combinations

All tests use in-memory span exporters for fast, isolated testing.

## Example Usage

### Complete Tracing Example

See [examples/trace/001-trace.py](examples/trace/001-trace.py) for a full working example:

```python
from openinference.instrumentation.pipecat import PipecatInstrumentor
from arize.otel import register

# Generate unique conversation ID
conversation_id = f"conversation-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
debug_log_filename = f"pipecat_frames_{conversation_id}.log"

# Set up tracing
tracer_provider = register(
    space_id=os.getenv("ARIZE_SPACE_ID"),
    api_key=os.getenv("ARIZE_API_KEY"),
    project_name=os.getenv("ARIZE_PROJECT_NAME"),
)

PipecatInstrumentor().instrument(
    tracer_provider=tracer_provider,
    debug_log_filename=debug_log_filename,
)

# Create your pipeline (STT -> LLM -> TTS)
pipeline = Pipeline([stt, llm, tts, transport.output()])

# Create task with conversation ID
task = PipelineTask(
    pipeline,
    conversation_id=conversation_id,
    params=PipelineParams(enable_metrics=True)
)

# Run - tracing happens automatically!
await runner.run(task)
```

### What Gets Traced

For a single user query → bot response:

**Turn Span** (`pipecat.conversation.turn`):
- `session.id`: "conversation-20251031_155612"
- `input.value`: "What is quantum computing?"
- `output.value`: "Quantum computing is a type of computing that uses quantum mechanics..."
- Duration: 3.5 seconds

**STT Span** (`pipecat.stt`):
- `service.name`: "OpenAISTTService"
- `output.value`: "What is quantum computing?"
- Duration: 0.78 seconds

**LLM Span** (`pipecat.llm`):
- `service.name`: "OpenAILLMService"
- `llm.provider`: "openai"
- `llm.model_name`: "gpt-4"
- `input.value`: [full message history]
- `llm.token_count.total`: 520
- Duration: 2.77 seconds

**TTS Span** (`pipecat.tts`):
- `service.name`: "OpenAITTSService"
- `audio.voice`: "alloy"
- `input.value`: "Quantum computing is..."
- Duration: 1.57 seconds

## Configuration Options

### Instrumentor Parameters

```python
PipecatInstrumentor().instrument(
    tracer_provider=tracer_provider,      # Required: OTel tracer provider
    config=TraceConfig(),                  # Optional: OpenInference config
    debug_log_filename="debug.log"        # Optional: Debug logging
)
```

### Per-Task Configuration

```python
task = PipelineTask(
    pipeline,
    conversation_id="user-session-123",   # Sets session.id attribute
)

# Optional: Override debug log for specific task
task._debug_log_filename = "task_specific_debug.log"
```

## OpenInference Semantic Conventions

Follows [OpenInference Semantic Conventions](https://github.com/Arize-ai/openinference/tree/main/spec):

- `openinference.span.kind`: `LLM` for language models, `CHAIN` for other services
- `session.id`: Conversation/session identifier
- `input.value` / `output.value`: Input/output text
- `llm.model_name`, `llm.provider`: LLM metadata
- `llm.token_count.*`: Token usage metrics
- Custom attributes: `audio.voice`, `service.name`, `frame.type`

## Benefits

1. **Production Observability**: Monitor voice agent performance, latency, and errors in production
2. **Debugging**: Detailed frame logs help diagnose pipeline issues
3. **Analytics**: Track conversation metrics, token usage, service latency
4. **Cost Monitoring**: Capture token counts for cost analysis
5. **Zero Code Changes**: Just add instrumentor - existing pipelines work unchanged
6. **Framework Agnostic**: Works with any OpenTelemetry-compatible backend (Arize, Jaeger, Phoenix, etc.)

## Compatibility

- **Pipecat**: 0.0.91+ (tested)
- **Python**: 3.8+
- **OpenTelemetry**: 1.20+
- **OpenInference**: Latest

## Related Issues

Implements instrumentation for Pipecat voice agent observability.
