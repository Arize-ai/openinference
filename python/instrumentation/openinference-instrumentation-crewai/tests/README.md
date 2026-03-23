# CrewAI Instrumentation Tests

## Re-recording VCR Cassettes

When tests fail due to outdated VCR cassettes (e.g., API authentication errors or changed responses), follow these steps to re-record.

The wrapper-based integration tests in `test_instrumentor.py` and the event-listener integration tests in `test_event_listener.py` intentionally share the same cassettes under `tests/cassettes/test_instrumentor/`.

### Prerequisites
1. Ensure `OPENAI_API_KEY` is set in your environment with a valid API key
2. The `passenv = OPENAI_API_KEY` directive must be present in the root `tox.ini` file

### Steps to Re-record

1. Delete the existing cassette file:
```bash
rm tests/cassettes/test_instrumentor/test_crewai_instrumentation.yaml
```

2. Re-record a single targeted test without xdist:
```bash
OPENAI_API_KEY=$OPENAI_API_KEY uvx --with tox-uv tox -r -e py313-ci-crewai-latest -- \
  tests/test_instrumentor.py::test_crewai_instrumentation -xvs --vcr-record=rewrite
```

3. If you also want to verify the event-listener path against the same cassette, run the matching event-listener test after the wrapper test succeeds:
```bash
OPENAI_API_KEY=$OPENAI_API_KEY uvx --with tox-uv tox -r -e py313-ci-crewai-latest -- \
  tests/test_event_listener.py::test_event_listener_crewai_instrumentation -xvs
```

### Important Notes
- The test reads `OPENAI_API_KEY` from the environment, falling back to "sk-test" if not set
- VCR will cache responses including authentication errors (401), so always delete the cassette before re-recording
- Re-record one test at a time. Shared cassettes and `pytest -n auto` are a bad combination when a cassette file is missing
- Use `-r` flag with tox to ensure a clean environment when re-recording
- The tests use `MockScrapeWebsiteTool`, so the only external credential needed for re-recording is `OPENAI_API_KEY`
