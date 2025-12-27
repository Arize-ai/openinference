# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-12-12

### Added
- Initial release of OpenInference Strands instrumentation
- Span processor that transforms Strands' native OpenTelemetry spans to OpenInference format
- Leverages Strands' built-in OTEL instrumentation with GenAI semantic conventions
- Transforms spans for:
  - Agent invocations (`invoke_agent`)
  - Event loop cycles (`execute_event_loop_cycle`)
  - Tool executions (`execute_tool`)
  - LLM calls (`chat`)
- Attribute transformation from GenAI to OpenInference conventions:
  - Model information (`gen_ai.request.model` → `llm.model_name`)
  - Token usage (`gen_ai.usage.*` → `llm.token_count.*`)
  - Tool details (`gen_ai.tool.*` → `tool.*`)
  - Message extraction from events to `llm.input_messages` / `llm.output_messages`
- Support for custom trace attributes (session_id, user_id, metadata, tags)
- Debug mode for detailed transformation logging

