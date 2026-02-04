# OpenInference Claude Code Instrumentation

[![PyPI Version](https://img.shields.io/pypi/v/openinference-instrumentation-claude-code.svg)](https://pypi.python.org/pypi/openinference-instrumentation-claude-code)

Instrumentation for [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python) that provides comprehensive observability into Claude Code operations.

## Installation

```bash
pip install openinference-instrumentation-claude-code
pip install claude-agent-sdk
```

## Authentication

The Claude Agent SDK requires authentication. You have two options:

### Option 1: Use Claude Code CLI Authentication (Recommended)
If you've already authenticated with the Claude Code CLI:
```bash
claude  # Run this once to authenticate
```

The SDK will automatically use your existing authentication.

### Option 2: Provide API Key
Set your Anthropic API key as an environment variable:
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Get your API key from the [Anthropic Console](https://console.anthropic.com/).

## Quickstart

```python
from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor
from claude_agent_sdk import query

# Instrument the SDK
ClaudeCodeInstrumentor().instrument()

# Use Claude Code normally - traces are automatically captured
async for message in query(prompt="What is 2+2?"):
    print(message)
```

## What Gets Traced

The instrumentation captures:

### 🤖 Agent Sessions
- Root AGENT spans for each query or client session
- Session IDs for tracking conversations
- Model and configuration parameters

### 💬 LLM Operations
- Individual query calls and agent turns
- Input prompts and output responses
- Token usage and costs
- Claude's thinking blocks (internal reasoning)

### 🔧 Tool Usage
- Built-in tools (Read, Write, Bash, etc.)
- Custom MCP tools
- Tool inputs and outputs
- Tool execution timing

### 🔄 Nested Subagents
- Automatic detection of subagent spawning
- Hierarchical span structure
- Subagent metadata tracking

## Span Hierarchy

```
AGENT: "Claude Code Query Session"
└── LLM: "Claude Code Query"
    └── LLM: "Agent Turn 1"
        ├── TOOL: "Read file.py"
        ├── AGENT: "Subagent: code-reviewer" (nested!)
        │   └── LLM: "Subagent Turn 1"
        │       └── TOOL: "Read utils.py"
        └── TOOL: "Write output.txt"
```

## Configuration

### Basic Setup with Phoenix

```python
from phoenix.otel import register
from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor

# Configure Phoenix tracer (sends traces to http://localhost:6006)
tracer_provider = register(
    project_name="my-claude-code-app",
    endpoint="http://localhost:6006/v1/traces",
)

# Instrument Claude Code SDK
ClaudeCodeInstrumentor().instrument(tracer_provider=tracer_provider)
```

Start Phoenix to collect traces:
```bash
python -m phoenix.server.main serve
```

View traces at: http://localhost:6006

### Hiding Sensitive Data

Use `TraceConfig` to control what data is captured:

```python
from openinference.instrumentation import TraceConfig
from openinference.instrumentation.claude_code import ClaudeCodeInstrumentor

config = TraceConfig(
    hide_inputs=True,       # Hide prompt content
    hide_outputs=True,      # Hide response content
)

ClaudeCodeInstrumentor().instrument(config=config)
```

### Context Attributes

Add session and user tracking:

```python
from openinference.instrumentation import using_session, using_user
from claude_agent_sdk import query

with using_session("chat-session-123"):
    with using_user("user-456"):
        async for message in query(prompt="Hello"):
            pass  # Session and user IDs attached to all spans
```

### Suppressing Tracing

Temporarily disable tracing:

```python
from openinference.instrumentation import suppress_tracing
from claude_agent_sdk import query

with suppress_tracing():
    # No spans created inside this block
    async for message in query(prompt="Not traced"):
        pass
```

## Examples

See the `examples/` directory for complete examples:

- **simple_query.py** - Basic query() usage
- **client_with_tools.py** - ClaudeSDKClient with tools
- **trace_config.py** - Hiding sensitive data

## Requirements

- Python >=3.9
- claude-agent-sdk >=0.1.29
- openinference-instrumentation >=0.1.27

## Contributing

See the main [OpenInference repository](https://github.com/Arize-ai/openinference) for contribution guidelines.

## License

Apache License 2.0
