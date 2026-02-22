# LLM Spans

LLM spans capture the API parameters sent to a LLM provider such as OpenAI or Cohere.

## Required Attributes

All LLM spans MUST include:
- `openinference.span.kind`: Set to `"LLM"`
- `llm.system`: The AI system/product (e.g., "openai", "anthropic")

## Common Attributes

LLM spans typically include:
- `llm.model_name`: The specific model used (e.g., "gpt-4-0613")
- `llm.invocation_parameters`: JSON string of parameters sent to the model
- `input.value`: The raw input as a JSON string
- `input.mime_type`: Usually "application/json"
- `output.value`: The raw output as a JSON string
- `output.mime_type`: Usually "application/json"
- `llm.input_messages`: Flattened list of input messages
- `llm.output_messages`: Flattened list of output messages
- `llm.token_count.*`: Token usage metrics

## Context Attributes

All LLM spans automatically inherit context attributes when they are set via the instrumentation context API. These attributes are propagated to every span in the trace without needing to be explicitly set on each span:

| Attribute | Description |
| --------- | ----------- |
| `session.id` | Unique identifier for the session |
| `user.id` | Unique identifier for the user |
| `metadata` | JSON string of key-value metadata associated with the trace |
| `tag.tags` | List of string tags for categorizing the span |
| `llm.prompt_template.template` | The prompt template used to generate the LLM input |
| `llm.prompt_template.variables` | JSON of key-value pairs applied to the prompt template |
| `llm.prompt_template.version` | Version identifier for the prompt template |

See [Configuration](./configuration.md) for details on how to set these context attributes.

## Attribute Flattening

Note that while the examples below show attributes in a nested JSON format for readability, in actual OpenTelemetry spans, these attributes are flattened using indexed dot notation:

- `llm.input_messages.0.message.role` instead of `llm.input_messages[0].message.role`
- `llm.output_messages.0.message.tool_calls.0.tool_call.function.name` for nested tool calls
- `llm.tools.0.tool.json_schema` for tool definitions

## Tool Role Messages

When a message with `message.role` set to `"tool"` represents the result of a function call, the `message.name` attribute MAY be set to identify which function produced the result. This complements `message.tool_call_id`, which links the result back to the original tool call request. For example:

```json
{
    "message.role": "tool",
    "message.content": "2001",
    "message.name": "multiply",
    "message.tool_call_id": "call_62136355"
}
```

See [Tool Calling](./tool_calling.md) for the complete tool calling flow.

## Examples

### Chat Completions

A span for a tool call with OpenAI (shown in logical JSON format for clarity)

```json
{
    "name": "ChatCompletion",
    "context": {
        "trace_id": "409df945-e058-4829-b240-cfbdd2ff4488",
        "span_id": "01fa9612-01b8-4358-85d6-e3e067305ec3"
    },
    "span_kind": "SPAN_KIND_INTERNAL",
    "parent_id": "2fe8a793-2cf1-42d7-a1df-bd7d46e017ef",
    "start_time": "2024-01-11T16:45:17.982858-07:00",
    "end_time": "2024-01-11T16:45:18.517639-07:00",
    "status_code": "OK",
    "status_message": "",
    "attributes": {
        "openinference.span.kind": "LLM",
        "llm.system": "openai",
        "llm.input_messages": [
            {
                "message.role": "system",
                "message.content": "You are a Shakespearean writing assistant who speaks in a Shakespearean style. You help people come up with creative ideas and content like stories, poems, and songs that use Shakespearean style of writing style, including words like \"thou\" and \"hath\u201d.\nHere are some example of Shakespeare's style:\n - Romeo, Romeo! Wherefore art thou Romeo?\n - Love looks not with the eyes, but with the mind; and therefore is winged Cupid painted blind.\n - Shall I compare thee to a summer's day? Thou art more lovely and more temperate.\n"
            },
            { "message.role": "user", "message.content": "what is 23 times 87" }
        ],
        "llm.model_name": "gpt-3.5-turbo-0613",
        "llm.invocation_parameters": "{\"model\": \"gpt-3.5-turbo-0613\", \"temperature\": 0.1, \"max_tokens\": null}",
        "output.value": "{\"tool_calls\": [{\"id\": \"call_Re47Qyh8AggDGEEzlhb4fu7h\", \"function\": {\"arguments\": \"{\\n  \\\"a\\\": 23,\\n  \\\"b\\\": 87\\n}\", \"name\": \"multiply\"}, \"type\": \"function\"}]}",
        "output.mime_type": "application/json",
        "llm.output_messages": [
            {
                "message.role": "assistant",
                "message.tool_calls": [
                    {
                        "tool_call.function.name": "multiply",
                        "tool_call.function.arguments": "{\n  \"a\": 23,\n  \"b\": 87\n}"
                    }
                ]
            }
        ],
        "llm.token_count.prompt": 229,
        "llm.token_count.completion": 21,
        "llm.token_count.total": 250
    },
    "events": []
}
```

A synthesis call using a function call output

```json
{
    "name": "llm",
    "context": {
        "trace_id": "409df945-e058-4829-b240-cfbdd2ff4488",
        "span_id": "f26d1f26-9671-435d-9716-14a87a3f228b"
    },
    "span_kind": "SPAN_KIND_INTERNAL",
    "parent_id": "2fe8a793-2cf1-42d7-a1df-bd7d46e017ef",
    "start_time": "2024-01-11T16:45:18.519427-07:00",
    "end_time": "2024-01-11T16:45:19.159145-07:00",
    "status_code": "OK",
    "status_message": "",
    "attributes": {
        "openinference.span.kind": "LLM",
        "llm.system": "openai",
        "llm.input_messages": [
            {
                "message.role": "system",
                "message.content": "You are a Shakespearean writing assistant who speaks in a Shakespearean style. You help people come up with creative ideas and content like stories, poems, and songs that use Shakespearean style of writing style, including words like \"thou\" and \"hath\u201d.\nHere are some example of Shakespeare's style:\n - Romeo, Romeo! Wherefore art thou Romeo?\n - Love looks not with the eyes, but with the mind; and therefore is winged Cupid painted blind.\n - Shall I compare thee to a summer's day? Thou art more lovely and more temperate.\n"
            },
            {
                "message.role": "user",
                "message.content": "what is 23 times 87"
            },
            {
                "message.role": "assistant",
                "message.content": null,
                "message.tool_calls": [
                    {
                        "tool_call.function.name": "multiply",
                        "tool_call.function.arguments": "{\n  \"a\": 23,\n  \"b\": 87\n}"
                    }
                ]
            },
            {
                "message.role": "tool",
                "message.content": "2001",
                "message.name": "multiply"
            }
        ],
        "llm.model_name": "gpt-3.5-turbo-0613",
        "llm.invocation_parameters": "{\"model\": \"gpt-3.5-turbo-0613\", \"temperature\": 0.1, \"max_tokens\": null}",
        "output.value": "The product of 23 times 87 is 2001.",
        "output.mime_type": "text/plain",
        "llm.output_messages": [
            {
                "message.role": "assistant",
                "message.content": "The product of 23 times 87 is 2001."
            }
        ],
        "llm.token_count.prompt": 259,
        "llm.token_count.completion": 14,
        "llm.token_count.total": 273
    },
    "events": [],
    "conversation": null
}
```

### Completions

A span for a simple completion (shown in logical JSON format for clarity)

```json
{
    "name": "Completion",
    "context": {
        "trace_id": "12345678-1234-5678-1234-567812345678",
        "span_id": "87654321-4321-8765-4321-876543218765"
    },
    "span_kind": "SPAN_KIND_INTERNAL",
    "parent_id": null,
    "start_time": "2025-09-29T03:42:49.000000Z",
    "end_time": "2025-09-29T03:42:50.284841Z",
    "status_code": "OK",
    "status_message": "",
    "attributes": {
        "openinference.span.kind": "LLM",
        "llm.system": "openai",
        "llm.model_name": "babbage:2023-07-21-v2",
        "llm.invocation_parameters": "{\"model\": \"babbage-002\", \"temperature\": 0.4, \"top_p\": 0.9, \"max_tokens\": 25}",
        "input.value": "{\"model\": \"babbage-002\", \"prompt\": \"def fib(n):\\n    if n <= 1:\\n        return n\\n    else:\\n        return fib(n-1) + fib(n-2)\", \"temperature\": 0.4, \"top_p\": 0.9, \"max_tokens\": 25}",
        "input.mime_type": "application/json",
        "llm.prompts.0.prompt.text": "def fib(n):\n    if n <= 1:\n        return n\n    else:\n        return fib(n-1) + fib(n-2)",
        "output.value": "{\"id\": \"cmpl-CKz4klHa1MMqAa4hQn3yzIMlLMZHd\", \"object\": \"text_completion\", \"created\": 1759117370, \"model\": \"babbage:2023-07-21-v2\", \"choices\": [{\"text\": \" + fib(n-3) + fib(n-4)\\n\\ndef fib(n):\\n    if n <= 1:\\n        return\", \"index\": 0, \"finish_reason\": \"length\"}], \"usage\": {\"prompt_tokens\": 31, \"completion_tokens\": 25, \"total_tokens\": 56}}",
        "output.mime_type": "application/json",
        "llm.choices.0.completion.text": " + fib(n-3) + fib(n-4)\n\ndef fib(n):\n    if n <= 1:\n        return",
        "llm.token_count.prompt": 31,
        "llm.token_count.completion": 25,
        "llm.token_count.total": 56
    },
    "events": []
}
```
