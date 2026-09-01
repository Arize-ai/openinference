# Pydantic AI Instrumentation Tests

## Re-recording VCR Cassettes

When tests fail due to outdated VCR cassettes (e.g., API authentication errors or changed responses), follow these steps to re-record:

### Prerequisites
1. Ensure `OPENAI_API_KEY` is set in your environment with a valid API key
2. The `passenv = OPENAI_API_KEY` directive must be present in the root `tox.ini` file

### Steps to Re-record

1. Delete the existing cassette file:
```bash
rm tests/openinference/instrumentation/pydantic_ai/cassettes/test_openai_agent_and_llm_spans.yaml
```

2. Run the test with VCR in record mode using tox:
```bash
OPENAI_API_KEY=$OPENAI_API_KEY uvx --with tox-uv tox -r -e py313-ci-pydantic_ai -- tests/openinference/instrumentation/pydantic_ai/test_instrumentor.py::test_openai_agent_and_llm_spans -xvs --vcr-record=once
```

### Important Notes
- The test reads `OPENAI_API_KEY` from the environment, falling back to "sk-test" if not set
- VCR will cache responses including authentication errors (401), so always delete the cassette before re-recording
- The `--vcr-record=once` flag ensures the cassette is only recorded when it doesn't exist
- Use `-r` flag with tox to ensure a clean environment when re-recording
