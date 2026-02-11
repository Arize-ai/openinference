# Changelog

## [0.1.0] - Unreleased

### Added
- Initial release of Claude Code SDK instrumentation
- Support for tracing query() function
- Support for tracing ClaudeSDKClient operations
- AGENT span creation for sessions
- LLM span creation for queries and turns
- TOOL span creation for tool usage
- Message parsing for text, tools, and thinking blocks
- Context attribute propagation (session_id, user_id)
- TraceConfig support for hiding sensitive data
- Suppress tracing support
- Comprehensive test suite
- Usage examples
