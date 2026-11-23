# LLM Spans

An LLM span (`openinference.span.kind` = `"LLM"`) records one call to a large language model, such as a chat
completion or a text completion request to OpenAI, Anthropic, or Cohere. It captures the request sent to the provider
and the response that came back.

## Attributes

[Requirement levels](../attributes/conventions.md#requirement-levels) follow the OpenTelemetry definitions.

<!-- weaver ["openinference.span.kind", "llm.system", "llm.model_name", "llm.provider", "llm.request.model_name", "llm.response.model_name", "llm.invocation_parameters", "llm.finish_reason", "input.value", "input.mime_type", "output.value", "output.mime_type", "llm.input_messages", "llm.output_messages", "llm.tools", "llm.token_count.prompt", "llm.token_count.completion", "llm.token_count.total"] as $keys | .registry.spans[] | select(.type == "openinference.llm") | {span: .type, span_attributes: [$keys[] as $k | .attributes[] | select(.key == $k)]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Requirement level | Type | Notes |
| --- | --- | --- | --- | --- |
| `openinference.span.kind` | ![Development](https://img.shields.io/badge/-development-blue) | Required | String | `"LLM"` |
| `llm.system` | ![Development](https://img.shields.io/badge/-development-blue) | Required | String | The AI product as identified by the client or server instrumentation. See [well-known values](../attributes/semantic-conventions.md#llm-system-values). |
| `llm.model_name` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | String | The name of the language model being utilized |
| `llm.provider` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | String | The hosting provider of the LLM, e.g. `azure`. See [well-known values](../attributes/semantic-conventions.md#llm-provider-values). When the hosting provider differs from the system. |
| `llm.request.model_name` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | String | The model requested by the caller, as sent in the request. May differ from `llm.response.model_name` when the provider routes the request to a different model. When the requested model can differ from the model that served the response. |
| `llm.response.model_name` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | String | The model that actually generated the response, as reported by the provider. May differ from `llm.request.model_name` when the provider routes the request to a different model. When the requested model can differ from the model that served the response. |
| `llm.invocation_parameters` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | JSON String | Parameters used during the invocation of an LLM or API |
| `llm.finish_reason` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | String | The reason the model stopped generating tokens. |
| `input.value` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | String | The raw input, usually a JSON string. |
| `input.mime_type` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | String | MIME type of `input.value`, usually `application/json`. When `input.value` is set and is not plain text. |
| `output.value` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | String | The raw output, usually a JSON string. |
| `output.mime_type` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | String | MIME type of `output.value`, usually `application/json`. When `output.value` is set and is not plain text. |
| `llm.input_messages` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | List of objects | List of messages sent to the LLM in a chat API request, e.g. `llm.input_messages.0.message.role`. See [Messages](../content/messages.md). For chat API calls. |
| `llm.output_messages` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | List of objects | List of messages received from the LLM in a chat API response, e.g. `llm.output_messages.0.message.role`. See [Messages](../content/messages.md). For chat API calls. |
| `llm.tools` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | List of objects | List of tools that are advertised to the LLM to be able to call, e.g. `llm.tools.0.tool.name`, `llm.tools.0.tool.description`, `llm.tools.0.tool.json_schema`. See [Tool Calling](../content/tool-calling.md). When tools are advertised to the model. |
| `llm.token_count.prompt` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | Integer | The number of tokens in the prompt |
| `llm.token_count.completion` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | Integer | The number of tokens in the completion |
| `llm.token_count.total` | ![Development](https://img.shields.io/badge/-development-blue) | Recommended | Integer | Total number of tokens, including prompt and completion |

<!-- endweaver -->

A completions (non-chat) API call records `llm.prompts` and `llm.choices` in place of the message lists; see
[Completions API](../content/messages.md#completions-api). LLM spans also carry the
[context attributes](../attributes/context.md) in scope when they are created.

## Model Identification

`llm.system`, `llm.provider`, and `llm.model_name` answer different questions:

- `llm.system` identifies the AI product or vendor, i.e. the shape of the API the client speaks. It should use a
  [well-known value](../attributes/semantic-conventions.md#llm-system-values) when one applies (e.g. `"openai"`, `"anthropic"`,
  `"cohere"`).
- `llm.provider` identifies the hosting provider, i.e. who runs the hardware, when it differs from the system (e.g.
  `"azure"` for Azure-hosted OpenAI). See the [well-known values](../attributes/semantic-conventions.md#llm-provider-values).
- `llm.model_name` contains the actual model name returned by the API (e.g. `"gpt-4-0613"`,
  `"claude-3-opus-20240229"`).

### Requested and Response Models

Some providers can route a request to a different model than the one the caller specified (for example,
classifier-triggered fallback). `llm.request.model_name` and `llm.response.model_name` let instrumentation record both
values as distinct, queryable attributes when it can tell them apart:

- `llm.request.model_name` is the model string the caller sent in the request.
- `llm.response.model_name` is the model the provider reports as having generated the response.
- Both are optional — only set them when the underlying API response actually distinguishes requested from
  effective model. Most providers echo the same model back, so these attributes will typically be unset.
- `llm.model_name` keeps its existing meaning and remains required where applicable: it should equal
  `llm.response.model_name` when known, falling back to `llm.request.model_name` otherwise. This keeps
  `llm.model_name` backward compatible for consumers that don't yet read the new attributes.

Embedding and decision spans do not use these `llm.*` identification attributes. See
[Embedding Spans](./embedding.md#attributes-not-used-in-embedding-spans) and
[Decision Spans](./decision.md#model-identification).

## Token Counts

`llm.token_count.prompt`, `llm.token_count.completion`, and `llm.token_count.total` record the token usage the
provider reports. The extended token count attributes break these down:

- `llm.token_count.prompt_details.*` for prompt-related token counts
- `llm.token_count.completion_details.*` for completion-related token counts

These enable tracking of multimodal (audio) token usage, monitoring of reasoning token consumption, cache efficiency
analysis, and billing reconciliation.

### Cache Tokens

`llm.token_count.prompt_details.cache_read` and `llm.token_count.prompt_details.cache_write` record cache operations:

- `cache_read` is the number of prompt tokens successfully retrieved from cache (cache hits). For OpenAI, this
  corresponds to `usage.prompt_tokens_details.cached_tokens` in Chat Completions API responses and
  `usage.input_tokens_details.cached_tokens` in Responses API responses. For Anthropic, when using a `cache_control`
  block, this maps to the `cache_read_input_tokens` field in Messages API responses.
- `cache_write` is the number of prompt tokens not found in cache (cache misses) that were subsequently written to
  cache. For OpenAI, this corresponds to the `usage.prompt_tokens_details.cache_write_tokens` field in Chat Completions
  API responses and the `usage.input_tokens_details.cache_write_tokens` field in Responses API responses. For
  Anthropic, when using a `cache_control` block, this maps to the `cache_creation_input_tokens` field in Messages API
  responses.

### Sub-Counts Are Inclusive

The `prompt_details.*` values are sub-counts of `llm.token_count.prompt`: they are already included in it, so
`llm.token_count.prompt` is expected to be greater than or equal to their sum. For providers whose reported input
token count excludes cache tokens (e.g. Anthropic's `input_tokens`), instrumentations should fold the cache read/write
tokens back into `llm.token_count.prompt` (and `llm.token_count.total`) rather than reporting the exclusive value.

## Cost

The `llm.cost.*` attributes record the cost of the call in USD, as floating point values. They mirror the token count
breakdown. Viewed as a JSON-like structure:

```json
{
  "prompt": 0.0021,  # Cost in USD
  "completion": 0.0045,  # Cost in USD
  "total": 0.0066,  # Cost in USD
  "completion_details": {
    "output": 0.0009,  # Cost in USD
    "reasoning": 0.0024,    # Cost in USD (e.g., 80 tokens * $0.03/1K tokens)
    "audio": 0.0012  # Cost in USD (e.g., 40 tokens * $0.03/1K tokens)
  },
  "prompt_details": {
    "input": 0.0003,  # Cost in USD
    "cache_write": 0.0006,  # Cost in USD (e.g., 20 tokens * $0.03/1K tokens)
    "cache_read": 0.0003,   # Cost in USD (e.g., 10 tokens * $0.03/1K tokens)
    "cache_input": 0.0006,  # Cost in USD (e.g., 20 tokens * $0.03/1K tokens)
    "audio": 0.0003   # Cost in USD (e.g., 10 tokens * $0.03/1K tokens)
  }
}
```

## Examples

The examples show list attributes in nested JSON form for readability. On the wire they are
[flattened](../attributes/conventions.md#flattening), e.g. `llm.input_messages.0.message.role`. Examples of reasoning
content are in [Messages](../content/messages.md#reasoning-content).

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
