# ElevenLabs Python SDK Auto-Instrumentation Plan

## Overview

This plan outlines the implementation of OpenInference auto-instrumentation for the ElevenLabs Python SDK. The instrumentation will capture telemetry from text-to-speech (TTS), speech-to-text (STT), speech-to-speech (STS), and conversational AI operations.

## Goals

1. Instrument ElevenLabs SDK to produce OpenTelemetry-compatible spans
2. Follow established patterns from existing OpenInference instrumentors (Anthropic, OpenAI, Groq)
3. Capture meaningful attributes for AI observability (input text, model, voice, audio metadata)
4. Support both sync and async client APIs
5. Handle streaming audio responses
6. Propagate context attributes (session_id, user_id, metadata, tags)
7. Support data masking via TraceConfig

---

## Scope

### Phase 1 - Core Implementation
- **Text-to-Speech**: `convert()`, `stream()`, `convert_with_timestamps()`, `stream_with_timestamps()`
- **Conversational AI**: Basic session-level tracing with `Conversation` class
- Both sync (`ElevenLabs`) and async (`AsyncElevenLabs`) clients
- ElevenLabs SDK version **1.x+** only

### Phase 2 - Extended (Future)
- Speech-to-Text: `convert()` transcription
- Speech-to-Speech: voice conversion

---

## Architecture

### Span Kind Mapping

| ElevenLabs Operation | OpenInference Span Kind | Rationale |
|---------------------|------------------------|-----------|
| `text_to_speech.convert()` | `LLM` | Generative AI converting text to audio |
| `text_to_speech.stream()` | `LLM` | Streaming generative operation |
| `Conversation` session | `CHAIN` | Parent span for conversation lifecycle |
| Conversation callbacks | Child of CHAIN | Individual events within conversation |

### File Structure

```
src/openinference/instrumentation/elevenlabs/
├── __init__.py              # ElevenLabsInstrumentor class (modify)
├── _wrappers.py             # NEW: Wrapper classes for TTS methods
├── _stream.py               # NEW: Streaming audio response wrappers
├── _with_span.py            # NEW: Span lifecycle management helper
├── _attributes.py           # NEW: Attribute extraction utilities
├── _conversation.py         # NEW: Conversation class instrumentation
├── package.py               # Package metadata (modify - add version)
└── version.py               # Version (existing)

tests/
├── conftest.py              # Test fixtures (modify)
├── test_instrumentor.py     # Instrumentor tests (existing)
├── test_text_to_speech.py   # NEW: TTS-specific tests
├── test_streaming.py        # NEW: Streaming tests
└── test_conversation.py     # NEW: Conversation tests
```

---

## Implementation Details

### 1. Main Instrumentor (`__init__.py`)

**Methods to wrap using `wrapt.wrap_function_wrapper()`:**

```python
# Text-to-Speech Sync Client
"elevenlabs.text_to_speech.client.TextToSpeechClient.convert"
"elevenlabs.text_to_speech.client.TextToSpeechClient.convert_with_timestamps"
"elevenlabs.text_to_speech.client.TextToSpeechClient.stream"
"elevenlabs.text_to_speech.client.TextToSpeechClient.stream_with_timestamps"

# Text-to-Speech Async Client
"elevenlabs.text_to_speech.client.AsyncTextToSpeechClient.convert"
"elevenlabs.text_to_speech.client.AsyncTextToSpeechClient.convert_with_timestamps"
"elevenlabs.text_to_speech.client.AsyncTextToSpeechClient.stream"
"elevenlabs.text_to_speech.client.AsyncTextToSpeechClient.stream_with_timestamps"

# Conversational AI
"elevenlabs.conversational_ai.conversation.Conversation.__init__"
"elevenlabs.conversational_ai.conversation.Conversation.start_session"
"elevenlabs.conversational_ai.conversation.Conversation.end_session"
"elevenlabs.conversational_ai.conversation.Conversation.wait_for_session_end"
```

**Instrumentor slots:**
```python
__slots__ = (
    "_tracer",
    "_config",
    "_original_tts_convert",
    "_original_tts_convert_with_timestamps",
    "_original_tts_stream",
    "_original_tts_stream_with_timestamps",
    "_original_async_tts_convert",
    "_original_async_tts_convert_with_timestamps",
    "_original_async_tts_stream",
    "_original_async_tts_stream_with_timestamps",
    "_original_conversation_init",
    "_original_conversation_start_session",
    "_original_conversation_end_session",
)
```

---

### 2. TTS Method Signatures (from SDK)

**convert() / stream():**
```python
def convert(
    self,
    voice_id: str,
    *,
    text: str,
    enable_logging: Optional[bool] = None,
    optimize_streaming_latency: Optional[int] = None,
    output_format: Optional[str] = None,
    model_id: Optional[str] = OMIT,
    language_code: Optional[str] = OMIT,
    voice_settings: Optional[VoiceSettings] = OMIT,
    pronunciation_dictionary_locators: Optional[Sequence[...]] = OMIT,
    seed: Optional[int] = OMIT,
    previous_text: Optional[str] = OMIT,
    next_text: Optional[str] = OMIT,
    previous_request_ids: Optional[Sequence[str]] = OMIT,
    next_request_ids: Optional[Sequence[str]] = OMIT,
    apply_text_normalization: Optional[str] = OMIT,
    request_options: Optional[RequestOptions] = None,
) -> Iterator[bytes]  # or AsyncIterator[bytes] for async
```

**convert_with_timestamps():**
```python
def convert_with_timestamps(...) -> AudioWithTimestampsResponse
# Returns object with audio_base64 and alignment data
```

---

### 3. Wrapper Classes (`_wrappers.py`)

**Base class:**
```python
class _WithTracer(ABC):
    __slots__ = ("_tracer",)

    def __init__(self, tracer: OITracer) -> None:
        self._tracer = tracer

    @contextmanager
    def _start_as_current_span(
        self,
        span_name: str,
        attributes: Mapping[str, AttributeValue],
        context_attributes: Mapping[str, AttributeValue],
    ) -> Iterator[_WithSpan]:
        span = self._tracer.start_span(name=span_name, attributes=dict(attributes))
        try:
            with trace_api.use_span(span, end_on_exit=False):
                yield _WithSpan(span, context_attributes)
        except Exception:
            raise
```

**Wrapper classes:**

| Class | Purpose |
|-------|---------|
| `_TextToSpeechConvertWrapper` | Sync `convert()` and `convert_with_timestamps()` |
| `_AsyncTextToSpeechConvertWrapper` | Async variants |
| `_TextToSpeechStreamWrapper` | Sync `stream()` - returns wrapped iterator |
| `_AsyncTextToSpeechStreamWrapper` | Async `stream()` - returns wrapped async iterator |

---

### 4. Streaming Support (`_stream.py`)

**Sync stream wrapper:**
```python
class _AudioStream(ObjectProxy):
    """Wraps sync audio byte iterator."""

    __slots__ = ("_self_with_span", "_self_byte_count", "_self_chunk_count")

    def __init__(self, stream: Iterator[bytes], with_span: _WithSpan) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_byte_count = 0
        self._self_chunk_count = 0

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        try:
            chunk = self.__wrapped__.__next__()
            self._self_byte_count += len(chunk)
            self._self_chunk_count += 1
            return chunk
        except StopIteration:
            self._finish_tracing(status=OK_STATUS)
            raise
        except Exception as e:
            self._finish_tracing(status=ERROR_STATUS, exception=e)
            raise

    def _finish_tracing(self, status, exception=None):
        if exception:
            self._self_with_span.record_exception(exception)
        self._self_with_span.set_attributes({
            "elevenlabs.audio_bytes": self._self_byte_count,
            "elevenlabs.audio_chunks": self._self_chunk_count,
        })
        self._self_with_span.finish_tracing(status=status)
```

**Async stream wrapper:**
```python
class _AsyncAudioStream(ObjectProxy):
    """Wraps async audio byte iterator."""
    # Similar pattern with __aiter__ and __anext__
```

---

### 5. Conversation Instrumentation (`_conversation.py`)

**Approach:** Wrap `Conversation.__init__` to inject tracing callbacks and create session span.

```python
class _ConversationInitWrapper(_WithTracer):
    """Wraps Conversation.__init__ to add tracing."""

    def __call__(self, wrapped, instance, args, kwargs):
        # Start CHAIN span for conversation session
        span = self._tracer.start_span(
            name="ElevenLabs.Conversation",
            attributes={
                OPENINFERENCE_SPAN_KIND: "CHAIN",
                "elevenlabs.agent_id": kwargs.get("agent_id"),
                "user.id": kwargs.get("user_id"),
            }
        )

        # Store span on instance for later access
        instance._otel_span = span

        # Wrap callbacks to create child spans
        original_callback = kwargs.get("callback_agent_response")
        kwargs["callback_agent_response"] = self._wrap_callback(
            original_callback, span, "agent_response"
        )

        # Similar wrapping for other callbacks...

        return wrapped(*args, **kwargs)
```

**Session lifecycle:**
- `start_session()`: No-op (span already created in __init__)
- `end_session()` / `wait_for_session_end()`: End the CHAIN span

---

### 6. Attribute Schema

**Request Attributes (TTS):**

| Attribute Key | Type | Source | Example |
|--------------|------|--------|---------|
| `openinference.span.kind` | string | constant | `"LLM"` |
| `input.value` | string | `text` param | `"Hello world"` |
| `input.mime_type` | string | constant | `"text/plain"` |
| `llm.model_name` | string | `model_id` param | `"eleven_multilingual_v2"` |
| `llm.provider` | string | constant | `"elevenlabs"` |
| `llm.system` | string | constant | `"elevenlabs"` |
| `llm.invocation_parameters` | string (JSON) | voice_settings, etc. | `{"stability": 0.5}` |

**ElevenLabs-specific Attributes:**

| Attribute Key | Type | Source |
|--------------|------|--------|
| `elevenlabs.voice_id` | string | `voice_id` param |
| `elevenlabs.output_format` | string | `output_format` param |
| `elevenlabs.optimize_streaming_latency` | int | param value |
| `elevenlabs.language_code` | string | `language_code` param |

**Response Attributes:**

| Attribute Key | Type | Source |
|--------------|------|--------|
| `output.value` | string | Metadata string | `"audio/mpeg, 12345 bytes"` |
| `output.mime_type` | string | Based on format | `"audio/mpeg"` |
| `elevenlabs.character_count` | int | `len(text)` |
| `elevenlabs.audio_bytes` | int | Total bytes streamed |
| `elevenlabs.audio_chunks` | int | Number of chunks |

**Conversation Attributes:**

| Attribute Key | Type | Source |
|--------------|------|--------|
| `openinference.span.kind` | string | `"CHAIN"` |
| `elevenlabs.agent_id` | string | `agent_id` param |
| `user.id` | string | `user_id` param |
| `elevenlabs.conversation_id` | string | From `wait_for_session_end()` return |

---

### 7. Output MIME Type Mapping

```python
OUTPUT_FORMAT_MIME_TYPES = {
    "mp3_44100_128": "audio/mpeg",
    "mp3_44100_64": "audio/mpeg",
    "mp3_44100_32": "audio/mpeg",
    "mp3_22050_32": "audio/mpeg",
    "pcm_16000": "audio/pcm",
    "pcm_22050": "audio/pcm",
    "pcm_24000": "audio/pcm",
    "pcm_44100": "audio/pcm",
    "ulaw_8000": "audio/basic",
    # Default
    None: "audio/mpeg",
}
```

---

### 8. Dependencies Update (`pyproject.toml`)

```toml
dependencies = [
  "opentelemetry-api",
  "opentelemetry-instrumentation",
  "opentelemetry-semantic-conventions",
  "openinference-instrumentation>=0.1.17",
  "openinference-semantic-conventions>=0.1.12",
  "wrapt",
]

[project.optional-dependencies]
instruments = [
  "elevenlabs>=1.0.0",
]
```

---

### 9. Package Metadata Update (`package.py`)

```python
_instruments = ("elevenlabs >= 1.0.0",)
```

---

## Testing Strategy

### Unit Tests

1. **test_instrumentor.py** (existing + expand):
   - Entry point registration
   - Instrument/uninstrument lifecycle
   - Config propagation
   - OITracer creation

2. **test_text_to_speech.py** (new):
   - Verify span created for `convert()`
   - Verify correct attributes extracted
   - Verify error handling and status
   - Test with custom TraceConfig (masking)
   - Test suppress_tracing context manager

3. **test_streaming.py** (new):
   - Verify stream wrapper works
   - Verify span ends on StopIteration
   - Verify byte/chunk counting
   - Verify error handling during stream

4. **test_conversation.py** (new):
   - Verify CHAIN span for session
   - Verify callback wrapping
   - Verify session end closes span

### Integration Tests (with cassettes)

Use `pytest-recording` for VCR-style tests:

```python
@pytest.mark.vcr()
def test_tts_convert_creates_span(setup_instrumentation, span_exporter):
    client = ElevenLabs()
    audio = client.text_to_speech.convert(
        text="Hello world",
        voice_id="test_voice",
        model_id="eleven_multilingual_v2",
    )
    # Consume iterator
    list(audio)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "ElevenLabs.TextToSpeech"
    assert spans[0].attributes["input.value"] == "Hello world"
```

---

## Verification Plan

1. **Install and run unit tests:**
   ```bash
   cd python/instrumentation/openinference-instrumentation-elevenlabs
   pip install -e ".[test]"
   pytest tests/ -v
   ```

2. **Run type checking:**
   ```bash
   mypy src/
   ```

3. **Run linting:**
   ```bash
   ruff check src/ tests/
   ruff format src/ tests/
   ```

4. **Manual verification with Phoenix:**
   ```python
   from openinference.instrumentation.elevenlabs import ElevenLabsInstrumentor
   from phoenix.otel import register

   tracer_provider = register()
   ElevenLabsInstrumentor().instrument(tracer_provider=tracer_provider)

   from elevenlabs import ElevenLabs
   client = ElevenLabs()

   # Test TTS
   audio = client.text_to_speech.convert(
       text="Hello world",
       voice_id="JBFqnCBsd6RMkjVDRZzb",
       model_id="eleven_multilingual_v2",
   )
   list(audio)  # Consume iterator

   # Check Phoenix UI for span with:
   # - name: "ElevenLabs.TextToSpeech"
   # - span.kind: "LLM"
   # - input.value: "Hello world"
   ```

5. **Verify tox integration:**
   ```bash
   tox run -e ruff-mypy-test-elevenlabs
   ```

---

## Critical Files Summary

| File | Action | Description |
|------|--------|-------------|
| `__init__.py` | Modify | Add wrapping logic in `_instrument()` |
| `_wrappers.py` | Create | TTS wrapper classes |
| `_stream.py` | Create | Audio stream wrappers |
| `_with_span.py` | Create | Span lifecycle helper |
| `_attributes.py` | Create | Attribute extraction utils |
| `_conversation.py` | Create | Conversation instrumentation |
| `pyproject.toml` | Modify | Add dependencies |
| `package.py` | Modify | Add version constraint |
| `test_text_to_speech.py` | Create | TTS tests |
| `test_streaming.py` | Create | Stream tests |
| `test_conversation.py` | Create | Conversation tests |

---

## Implementation Order

1. **Core infrastructure**: `_with_span.py`, `_attributes.py`
2. **TTS convert**: `_wrappers.py` (convert wrappers only)
3. **TTS streaming**: `_stream.py`, add stream wrappers
4. **Wire up instrumentor**: Update `__init__.py`
5. **Tests**: `test_text_to_speech.py`, `test_streaming.py`
6. **Conversation**: `_conversation.py`, `test_conversation.py`
7. **Final polish**: README, type hints, docs
