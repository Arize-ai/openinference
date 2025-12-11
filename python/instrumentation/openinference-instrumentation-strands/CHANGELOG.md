# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-11

### Added
- Initial release of OpenInference Strands instrumentation
- Automatic tracing for Agent invocations (invoke_async and stream_async)
- Automatic tracing for event loop cycles
- Automatic tracing for tool executions
- Support for OpenInference semantic conventions
- Comprehensive span attributes including:
  - Agent name and ID
  - Input/output values
  - Token usage metrics
  - Tool information
  - Stop reasons and execution metadata

