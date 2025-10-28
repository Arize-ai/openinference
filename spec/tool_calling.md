# Tool and Function Calling

This document describes how tool/function calling is represented in OpenInference spans.

## Tool Definitions

Tools available to the LLM are represented using the `llm.tools` prefix with flattened attributes:

### Attribute Pattern

`llm.tools.<index>.tool.json_schema`

The `json_schema` contains the complete tool definition as a JSON string, including:
- Tool type (usually "function")
- Function name
- Function description
- Parameter schema

### Example Tool Definition

```json
{
  "llm.tools.0.tool.json_schema": "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get current weather for a location\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City and state\"}}, \"required\": [\"location\"]}}}"
}
```

## Tool Calls in Messages

When an LLM generates tool calls, they are represented in the output messages:

### Attribute Pattern for Tool Calls

`llm.output_messages.<messageIndex>.message.tool_calls.<toolCallIndex>.tool_call.<attribute>`

Where:
- `<messageIndex>` is the zero-based index of the message
- `<toolCallIndex>` is the zero-based index of the tool call within the message
- `<attribute>` is the specific tool call attribute

### Tool Call Attributes

- `tool_call.id`: Unique identifier for the tool call
- `tool_call.function.name`: Name of the function being called
- `tool_call.function.arguments`: JSON string containing the function arguments

### Example Tool Call

```json
{
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_abc123",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{\"location\": \"San Francisco, CA\"}"
}
```

## Multiple Tool Calls

When an LLM makes multiple tool calls in a single response:

```json
{
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_001",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{\"location\": \"New York\"}",
  "llm.output_messages.0.message.tool_calls.1.tool_call.id": "call_002",
  "llm.output_messages.0.message.tool_calls.1.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments": "{\"location\": \"London\"}"
}
```

## Tool Results

Tool results are typically represented as input messages with role "tool":

```json
{
  "llm.input_messages.3.message.role": "tool",
  "llm.input_messages.3.message.content": "{\"temperature\": 72, \"condition\": \"sunny\"}",
  "llm.input_messages.3.message.tool_call_id": "call_abc123"
}
```

The `message.tool_call_id` links the result back to the original tool call.

## Complete Tool Call Flow Example

1. **User Request**:
```json
{
  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.0.message.content": "What's the weather in Boston?"
}
```

2. **Available Tools**:
```json
{
  "llm.tools.0.tool.json_schema": "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get current weather\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\"}}}}}"
}
```

3. **LLM Tool Call**:
```json
{
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_123",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
  "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": "{\"location\": \"Boston, MA\"}"
}
```

4. **Tool Result** (in next request):
```json
{
  "llm.input_messages.2.message.role": "tool",
  "llm.input_messages.2.message.content": "{\"temperature\": 65, \"condition\": \"cloudy\"}",
  "llm.input_messages.2.message.tool_call_id": "call_123"
}
```

5. **Final Response**:
```json
{
  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.content": "The current weather in Boston is 65°F and cloudy."
}
```

## Legacy Attributes

Some implementations may use legacy attributes for function calling:
- `message.function_call_name`: Function name (deprecated, use tool_calls)
- `message.function_call_arguments_json`: Function arguments (deprecated, use tool_calls)
- `llm.function_call`: Complete function call as JSON (deprecated)

New implementations should use the `tool_calls` structure described above.