# Processor Tests

This directory contains integration tests for the agent-framework instrumentation processor using pytest-recording (VCR-based) to record and replay HTTP interactions.

## Setup

The processor tests use `pytest-recording` to record real API interactions with agent-framework and OpenAI, allowing tests to run without actual API calls after initial recording. pytest-recording has excellent httpx support, which is crucial since agent-framework uses the httpx-based OpenAI client.

### Prerequisites

1. Install test dependencies:
   ```bash
   pip install -r test-requirements.txt
   ```

2. Enable agent-framework instrumentation (done automatically in conftest.py):
   ```python
   from agent_framework.observability import enable_instrumentation
   enable_instrumentation(enable_sensitive_data=True)
   ```

   Note: API may vary by version. Pinned to 1.0.0b260130 for stability.

   Note: `enable_sensitive_data=True` is required to capture message content in spans.

## Recording New Cassettes

To record new cassettes with real API calls:

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your_actual_api_key
   ```

   Or update the default value in `test_processor.py` (line 56-58).

2. Delete existing cassettes you want to re-record:
   ```bash
   rm -rf tests/cassettes/test_processor/
   ```

3. Run the tests with `--record-mode=rewrite` to force re-recording:
   ```bash
   pytest tests/test_processor.py -v --record-mode=rewrite
   ```

   Or just run normally (default mode is `once` - only records if cassette doesn't exist):
   ```bash
   pytest tests/test_processor.py -v
   ```

4. The cassettes will be saved in `tests/cassettes/test_processor/` with:
   - Sanitized headers (no API keys stored via `filter_headers` in `vcr_config`)
   - HTTP request/response pairs in YAML format
   - Organized by test module and function name

## Running Tests with Existing Cassettes

After cassettes are recorded, tests run without an API key and are much faster:

```bash
pytest tests/test_processor.py -v
```

**Performance**: ~5s with API calls → ~1.3s with cassettes ⚡

## Recording Modes

pytest-recording supports several modes via `--record-mode`:

- `once` (default): Record if cassette doesn't exist, otherwise replay
- `rewrite`: Always record, overwriting existing cassettes
- `new_episodes`: Record new interactions, keep existing ones
- `none`: Never record, only replay (fails if cassette missing)
- `all`: Record every test run (not recommended)

## Why pytest-recording?

pytest-recording was chosen over pytest-vcr for several key advantages:

1. **Excellent httpx Support**: Works seamlessly with agent-framework's httpx-based OpenAI client
2. **Better Async Support**: Properly handles async/await patterns
3. **Simpler Configuration**: Less boilerplate, cleaner test decorators
4. **Active Maintenance**: Well-maintained with modern Python support
5. **Flexible Recording Modes**: Easy control over when to record vs replay

The switch from pytest-vcr to pytest-recording resolved playback issues that occurred with httpx responses.

## Test Configuration

### conftest.py

Key fixtures:

- `in_memory_span_exporter`: Session-scoped exporter to capture spans
- `tracer_provider`: Session-scoped tracer provider with OpenInference processor
- `vcr_config`: VCR configuration for sanitizing recordings
- `clear_spans`: Auto-use fixture to clean spans between tests

### pytest-recording Configuration

Tests use the simple `@pytest.mark.vcr` decorator:

```python
@pytest.mark.vcr
@pytest.mark.asyncio
async def test_openai_chat_with_agent(...):
    ...
```

Configuration is centralized in `conftest.py`:

```python
@pytest.fixture(scope="session")
def vcr_config():
    return {
        "filter_headers": ["authorization", "api-key", "x-api-key"],
        "decode_compressed_response": True,
        "record_mode": "once",
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
    }
```

This configuration:
- Filters sensitive headers (protects API keys)
- Decodes compressed responses for better readability
- Records once by default (idempotent tests)
- Matches requests by URL and method

## Test Coverage

### test_openai_chat_with_agent
Tests basic OpenAI chat agent with:
- Simple question/answer
- LLM span creation
- OpenInference attribute mapping
- Token usage tracking
- Message content capture

### test_agent_with_tool_calls
Tests agent with tool usage:
- Tool definition and registration
- Tool execution spans
- Multiple span types (LLM, TOOL)
- Tool invocation tracking

### test_conversation_with_history
Tests multi-turn conversations:
- Conversation context persistence
- Session ID tracking
- Multiple messages in same session
- History maintenance

## Debugging

To see detailed span attributes during test runs:

```python
# Add to test
print("\n=== Span Attributes ===")
for key in sorted(attrs.keys()):
    print(f"{key}: {attrs[key]}")
```

To check what spans were created:

```python
spans = in_memory_span_exporter.get_finished_spans()
for span in spans:
    print(f"Span: {span.name}")
    print(f"  Kind: {span.attributes.get('openinference.span.kind')}")
    print(f"  Attributes: {span.attributes.keys()}")
```

## CI/CD Integration

For CI environments:

1. **With recorded cassettes** (recommended):
   - Commit cassettes to repository
   - Tests run without API keys
   - Fast, deterministic tests
   - Note: May fail due to httpx playback issues

2. **With live API calls**:
   ```yaml
   env:
     OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
   ```
   - Set API key as secret
   - Tests make real API calls
   - Slower but validates against current API
