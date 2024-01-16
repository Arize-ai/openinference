# LLM Spans

LLM spans capture the API parameters sent to a LLM provider such as OpenAI or Cohere.

## Examples

A span for a tool call with OpenAI

```json
{
    "name": "llm",
    "context": {
        "trace_id": "409df945-e058-4829-b240-cfbdd2ff4488",
        "span_id": "01fa9612-01b8-4358-85d6-e3e067305ec3"
    },
    "span_kind": "LLM",
    "parent_id": "2fe8a793-2cf1-42d7-a1df-bd7d46e017ef",
    "start_time": "2024-01-11T16:45:17.982858-07:00",
    "end_time": "2024-01-11T16:45:18.517639-07:00",
    "status_code": "OK",
    "status_message": "",
    "attributes": {
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
    "span_kind": "LLM",
    "parent_id": "2fe8a793-2cf1-42d7-a1df-bd7d46e017ef",
    "start_time": "2024-01-11T16:45:18.519427-07:00",
    "end_time": "2024-01-11T16:45:19.159145-07:00",
    "status_code": "OK",
    "status_message": "",
    "attributes": {
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
