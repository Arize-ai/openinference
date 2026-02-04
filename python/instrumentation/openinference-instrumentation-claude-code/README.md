# OpenInference Claude Code Instrumentation

Instrumentation for [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) that provides observability into Claude Code operations.

## Installation

```bash
pip install openinference-instrumentation-claude-code
```

## Quickstart

```python
from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor
from claude_agent_sdk import query

ClaudeCodeInstrumentor().instrument()

async for message in query(prompt="What is 2+2?"):
    print(message)
```

## Features

- Traces SDK API calls (query, ClaudeSDKClient operations)
- Captures agent reasoning and tool usage
- Supports nested subagents
- Respects TraceConfig for hiding sensitive data

See `examples/` for more usage patterns.
