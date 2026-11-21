# Semantic Conventions

The semantic conventions define the keys and values that describe commonly observed concepts and operations in AI
applications. They populate the attributes of spans and span events. This page lists every OpenInference attribute,
grouped by namespace. All OpenInference tracing SDKs MUST support these attributes. The [span conventions](../spans/llm.md) say which attributes a span of each kind carries, and
[Attribute Conventions](./conventions.md) explains how names, types, and lists are encoded.

The tables on this page are generated from the machine-readable
[semantic convention registry](https://github.com/Arize-ai/openinference/tree/main/semantic-conventions). To change
one, edit the registry and regenerate; do not edit the tables by hand.

Types marked "List of objects" are [flattened](./conventions.md#flattening) with zero-based indices, e.g.
`llm.input_messages.0.message.role`.

## OpenInference

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("openinference."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `openinference.project.name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The name of the project that the telemetry belongs to. Recorded as a resource attribute, not a span attribute. Identifies the `openinference.project` entity. | `"default"`, `"my-llm-app"` |
| `openinference.span.kind` | ![Development](https://img.shields.io/badge/-development-blue) | String | The role an operation plays in an AI application. Required on every OpenInference span. It is distinct from OpenTelemetry's own `SpanKind`. See [Span Kinds](../general/span-kinds.md). | `"LLM"`, `"CHAIN"`, `"TOOL"`, `"RETRIEVER"`, `"RERANKER"`, `"EMBEDDING"`, `"AGENT"`, `"GUARDRAIL"`, `"EVALUATOR"`, `"PROMPT"`, `"DECISION"`, `"UNKNOWN"` |

<!-- endweaver -->

## Input and Output

Valid on spans of any kind.

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("input.") or startswith("output."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `input.documents` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | File document (e.g. PDF) inputs to an operation, independent of span kind, e.g. `input.documents.0.document.url`. See [Multimodal Content](../content/multimodal.md#span-kind-independent-documents) | `[{"document.url": "https://example.com/report.pdf", "document.mime_type": "application/pdf"}]` |
| `input.images` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | Image inputs to an operation, independent of span kind, e.g. `input.images.0.image.url`. See [Multimodal Content](../content/multimodal.md#span-kind-independent-images) | `[{"image.url": "https://sample-link-to-image.jpg"}]` |
| `input.mime_type` | ![Development](https://img.shields.io/badge/-development-blue) | String | MIME type representing the format of `input.value` | `"text/plain"`, `"application/json"` |
| `input.value` | ![Development](https://img.shields.io/badge/-development-blue) | String | The input value to an operation | `"{'query': 'What is the weather today?'}"` |
| `output.documents` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | File document (e.g. PDF) outputs of an operation, independent of span kind, e.g. `output.documents.0.document.url`. See [Multimodal Content](../content/multimodal.md#span-kind-independent-documents) | `[{"document.url": "https://example.com/filled-form.pdf", "document.mime_type": "application/pdf"}]` |
| `output.images` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | Image outputs of an operation, independent of span kind, e.g. `output.images.0.image.url`. See [Multimodal Content](../content/multimodal.md#span-kind-independent-images) | `[{"image.url": "data:image/png;base64,iVBORw0KGgo..."}]` |
| `output.mime_type` | ![Development](https://img.shields.io/badge/-development-blue) | String | MIME type representing the format of `output.value` | `"text/plain"`, `"application/json"` |
| `output.value` | ![Development](https://img.shields.io/badge/-development-blue) | String | The output value of an operation | `"Hello, World!"` |

<!-- endweaver -->

## Context

Usually set through the instrumentation context API, which attaches them to every span in scope. See
[Context Attributes](./context.md).

<!-- weaver ["session.id", "user.id", "metadata", "tag.tags"] as $keys | {attributes: [$keys[] as $k | .registry.spans[] | select(.type == "openinference.chain") | .attributes[] | select(.key == $k)]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `session.id` | ![Development](https://img.shields.io/badge/-development-blue) | String | Unique identifier for a session, such as a multi-turn conversation. | `"00112233-4455-6677-8899-aabbccddeeff"` |
| `user.id` | ![Development](https://img.shields.io/badge/-development-blue) | String | Unique identifier for a user. | `"S-1-5-21-202424912787-2692429404-2351956786-1000"` |
| `metadata` | ![Development](https://img.shields.io/badge/-development-blue) | JSON String | Metadata associated with a span | `"{'author': 'John Doe', 'date': '2023-09-09'}"` |
| `tag.tags` | ![Development](https://img.shields.io/badge/-development-blue) | List of strings | List of tags to give the span a category | `["shopping", "travel"]` |

<!-- endweaver -->

## LLM

### Model Identification

See [LLM Spans](../spans/llm.md#model-identification) for how these attributes relate.

<!-- weaver ["llm.system", "llm.provider", "llm.model_name", "llm.request.model_name", "llm.response.model_name"] as $keys | {attributes: [$keys[] as $k | .registry.attributes[] | select(.key == $k)]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `llm.system` | ![Development](https://img.shields.io/badge/-development-blue) | String | The AI product as identified by the client or server instrumentation. See [well-known values](../attributes/semantic-conventions.md#llm-system-values). | `"anthropic"`, `"openai"` |
| `llm.provider` | ![Development](https://img.shields.io/badge/-development-blue) | String | The hosting provider of the LLM, e.g. `azure`. See [well-known values](../attributes/semantic-conventions.md#llm-provider-values). | `"openai"`, `"azure"` |
| `llm.model_name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The name of the language model being utilized | `"gpt-3.5-turbo"` |
| `llm.request.model_name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The model requested by the caller, as sent in the request. May differ from `llm.response.model_name` when the provider routes the request to a different model. | `"claude-opus-5"` |
| `llm.response.model_name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The model that actually generated the response, as reported by the provider. May differ from `llm.request.model_name` when the provider routes the request to a different model. | `"claude-opus-4-8"` |

<!-- endweaver -->

### Invocation

<!-- weaver ["llm.invocation_parameters", "llm.finish_reason", "llm.function_call"] as $keys | {attributes: [$keys[] as $k | .registry.attributes[] | select(.key == $k)]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `llm.invocation_parameters` | ![Development](https://img.shields.io/badge/-development-blue) | JSON String | Parameters used during the invocation of an LLM or API | `"{model_name: 'gpt-3', temperature: 0.7}"` |
| `llm.finish_reason` | ![Development](https://img.shields.io/badge/-development-blue) | String | The reason the model stopped generating tokens. | `"stop"`, `"length"` |
| `llm.function_call` | ![Deprecated](https://img.shields.io/badge/-deprecated-red) | JSON String | Deprecated: Use `message.tool_calls` instead. Object recording details of a function call in models or APIs | `"{function_name: 'add', args: [1, 2]}"` |

<!-- endweaver -->

### Messages, Prompts, and Tools

<!-- weaver ["llm.input_messages", "llm.output_messages", "llm.prompts", "llm.choices", "llm.tools"] as $keys | {attributes: [$keys[] as $k | .registry.attributes[] | select(.key == $k)]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `llm.input_messages` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | List of messages sent to the LLM in a chat API request, e.g. `llm.input_messages.0.message.role`. See [Messages](../content/messages.md) | `[{"message.role": "user", "message.content": "hello"}]` |
| `llm.output_messages` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | List of messages received from the LLM in a chat API response, e.g. `llm.output_messages.0.message.role`. See [Messages](../content/messages.md) | `[{"message.role": "assistant", "message.content": "hello"}]` |
| `llm.prompts` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | Prompts provided to a completions API, e.g. `llm.prompts.0.prompt.text` | `[{"prompt.text": "def fib(n):..."}]` |
| `llm.choices` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | Text choices returned from a completions API, e.g. `llm.choices.0.completion.text` | `[{"completion.text": " + fib(n-3)..."}]` |
| `llm.tools` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | List of tools that are advertised to the LLM to be able to call, e.g. `llm.tools.0.tool.name`, `llm.tools.0.tool.description`, `llm.tools.0.tool.json_schema`. See [Tool Calling](../content/tool-calling.md) | `[{"tool.name": "get_weather", "tool.json_schema": "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"}}"}]` |

<!-- endweaver -->

### Prompt Template

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("llm.prompt_template."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `llm.prompt_template.template` | ![Development](https://img.shields.io/badge/-development-blue) | String | Template used to generate prompts (e.g., using `{variable}` placeholder syntax) | `"Weather forecast for {city} on {date}"` |
| `llm.prompt_template.variables` | ![Development](https://img.shields.io/badge/-development-blue) | JSON String | JSON of key value pairs applied to the prompt template | `"{ context: \"<context from retrieval>\", subject: \"math\" }"` |
| `llm.prompt_template.version` | ![Development](https://img.shields.io/badge/-development-blue) | String | The version of the prompt template | `"v1.0"` |

<!-- endweaver -->

### Token Counts

All token counts are integers. The `*_details.*` values are sub-counts of their parent count; see
[LLM Spans](../spans/llm.md#token-counts).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("llm.token_count."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `llm.token_count.completion` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | The number of tokens in the completion | `15` |
| `llm.token_count.completion_details.audio` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | The number of audio tokens generated by the model | `10` |
| `llm.token_count.completion_details.reasoning` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | The number of tokens used for model reasoning | `10` |
| `llm.token_count.prompt` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | The number of tokens in the prompt | `10` |
| `llm.token_count.prompt_details.audio` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | The number of audio input tokens presented in the prompt | `10` |
| `llm.token_count.prompt_details.cache_input` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | The number of input tokens in the prompt that were cached | `5` |
| `llm.token_count.prompt_details.cache_read` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | The number of prompt tokens successfully retrieved from cache (cache hits). Maps to `cached_tokens` in OpenAI responses | `5` |
| `llm.token_count.prompt_details.cache_write` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | The number of prompt tokens not found in cache that were written to cache (cache misses). Maps to `cache_write_tokens` in OpenAI responses | `0` |
| `llm.token_count.total` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | Total number of tokens, including prompt and completion | `20` |

<!-- endweaver -->

### Cost

All costs are floating point values in USD. See [LLM Spans](../spans/llm.md#cost).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("llm.cost."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `llm.cost.completion` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Total cost of all output tokens generated by the LLM in USD | `0.0045` |
| `llm.cost.completion_details.audio` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Cost of audio tokens in the completion in USD | `0.0012` |
| `llm.cost.completion_details.output` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Total cost of output tokens in USD | `0.0009` |
| `llm.cost.completion_details.reasoning` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Cost of reasoning steps in the completion in USD | `0.0024` |
| `llm.cost.prompt` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Total cost of all input tokens sent to the LLM in USD | `0.0021` |
| `llm.cost.prompt_details.audio` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Cost of audio tokens in the prompt in USD | `0.0003` |
| `llm.cost.prompt_details.cache_input` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Cost of input tokens in the prompt that were cached in USD | `0.0006` |
| `llm.cost.prompt_details.cache_read` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Cost of prompt tokens read from cache in USD | `0.0003` |
| `llm.cost.prompt_details.cache_write` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Cost of prompt tokens written to cache in USD | `0.0006` |
| `llm.cost.prompt_details.input` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Total cost of input tokens in USD | `0.0003` |
| `llm.cost.total` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Total cost of the LLM call in USD (prompt + completion) | `0.0066` |

<!-- endweaver -->

### LLM System Values

[Well-known values](./conventions.md#well-known-values) for `llm.system`:

<!-- weaver .registry.attributes[] | select(.key == "llm.system") | {members: .type.members} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Value | Description | Stability |
| --- | --- | --- |
| `openai` | OpenAI | ![Development](https://img.shields.io/badge/-development-blue) |
| `anthropic` | Anthropic | ![Development](https://img.shields.io/badge/-development-blue) |
| `mistralai` | Mistral AI | ![Development](https://img.shields.io/badge/-development-blue) |
| `cohere` | Cohere | ![Development](https://img.shields.io/badge/-development-blue) |
| `vertexai` | Vertex AI | ![Development](https://img.shields.io/badge/-development-blue) |
| `deepseek` | DeepSeek | ![Development](https://img.shields.io/badge/-development-blue) |
| `amazon` | Amazon Bedrock native | ![Development](https://img.shields.io/badge/-development-blue) |
| `meta` | Meta (Llama) | ![Development](https://img.shields.io/badge/-development-blue) |
| `ai21` | AI21 Labs | ![Development](https://img.shields.io/badge/-development-blue) |
| `typesafe` | TypeSafe AI | ![Development](https://img.shields.io/badge/-development-blue) |
| `xai` | xAI (Grok) | ![Development](https://img.shields.io/badge/-development-blue) |

<!-- endweaver -->

### LLM Provider Values

[Well-known values](./conventions.md#well-known-values) for `llm.provider`:

<!-- weaver .registry.attributes[] | select(.key == "llm.provider") | {members: .type.members} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Value | Description | Stability |
| --- | --- | --- |
| `openai` | OpenAI | ![Development](https://img.shields.io/badge/-development-blue) |
| `anthropic` | Anthropic | ![Development](https://img.shields.io/badge/-development-blue) |
| `mistralai` | Mistral AI | ![Development](https://img.shields.io/badge/-development-blue) |
| `cohere` | Cohere | ![Development](https://img.shields.io/badge/-development-blue) |
| `google` | Google (Vertex) | ![Development](https://img.shields.io/badge/-development-blue) |
| `aws` | AWS Bedrock | ![Development](https://img.shields.io/badge/-development-blue) |
| `azure` | Azure | ![Development](https://img.shields.io/badge/-development-blue) |
| `xai` | xAI | ![Development](https://img.shields.io/badge/-development-blue) |
| `deepseek` | DeepSeek | ![Development](https://img.shields.io/badge/-development-blue) |
| `groq` | Groq | ![Development](https://img.shields.io/badge/-development-blue) |
| `fireworks` | Fireworks AI | ![Development](https://img.shields.io/badge/-development-blue) |
| `moonshot` | Moonshot AI | ![Development](https://img.shields.io/badge/-development-blue) |
| `cerebras` | Cerebras | ![Development](https://img.shields.io/badge/-development-blue) |
| `perplexity` | Perplexity | ![Development](https://img.shields.io/badge/-development-blue) |
| `together` | Together AI | ![Development](https://img.shields.io/badge/-development-blue) |
| `ollama` | Ollama | ![Development](https://img.shields.io/badge/-development-blue) |
| `meta` | Meta AI | ![Development](https://img.shields.io/badge/-development-blue) |
| `zai` | Z.ai (GLM) | ![Development](https://img.shields.io/badge/-development-blue) |
| `minimax` | MiniMax | ![Development](https://img.shields.io/badge/-development-blue) |
| `oracle` | Oracle (OCI Generative AI) | ![Development](https://img.shields.io/badge/-development-blue) |
| `ibm` | IBM (watsonx.ai) | ![Development](https://img.shields.io/badge/-development-blue) |
| `typesafe` | TypeSafe AI | ![Development](https://img.shields.io/badge/-development-blue) |
| `doubleword` | Doubleword | ![Development](https://img.shields.io/badge/-development-blue) |

<!-- endweaver -->

## Message

Items of `llm.input_messages` and `llm.output_messages`. See [Messages](../content/messages.md).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("message."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `message.content` | ![Development](https://img.shields.io/badge/-development-blue) | String | The content of a message in a chat | `"What's the weather today?"` |
| `message.contents` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | The message contents to the LLM, an array of `message_content` objects. Use this form when a message has multiple content items or when item ordering matters. | `[{"message_content.type": "text", "message_content.text": "Hello"}]` |
| `message.function_call_arguments_json` | ![Deprecated](https://img.shields.io/badge/-deprecated-red) | JSON String | Deprecated: Use `message.tool_calls` instead. The arguments to the function call in JSON | `"{ 'x': 2 }"` |
| `message.function_call_name` | ![Deprecated](https://img.shields.io/badge/-deprecated-red) | String | Deprecated: Use `message.tool_calls` instead. Function call function name | `"multiply"` |
| `message.name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The name of the function or tool that produced a tool/function role message. Used when `message.role` is `"tool"` or `"function"` to identify the origin. | `"multiply"` |
| `message.role` | ![Development](https://img.shields.io/badge/-development-blue) | String | Role of the entity in a message (e.g., user, system) | `"user"`, `"system"` |
| `message.tool_call_id` | ![Development](https://img.shields.io/badge/-development-blue) | String | Tool call result identifier corresponding to `tool_call.id` | `"call_62136355"` |
| `message.tool_calls` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | List of tool calls (e.g. function calls) generated by the LLM, e.g. `llm.output_messages.0.message.tool_calls.0.tool_call.id` | `[{"tool_call.function.name": "get_current_weather"}]` |

<!-- endweaver -->

## Message Content

Items of `message.contents`. See [Multimodal Content](../content/multimodal.md) and
[Reasoning Content](../content/messages.md#reasoning-content).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("message_content."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `message_content.audio` | ![Development](https://img.shields.io/badge/-development-blue) | Audio Object | The audio content of the message, if the type is `"audio"`. Nested leaves are `audio.url` and `audio.transcript`. | `{"audio.url": "https://example.com/audio.mp3"}` |
| `message_content.data` | ![Development](https://img.shields.io/badge/-development-blue) | String | Opaque vendor-issued data captured verbatim. Maps to Anthropic `redacted_thinking.data`. | `"EmwKAhgBEgy3..."` |
| `message_content.document` | ![Development](https://img.shields.io/badge/-development-blue) | Document Object | The document content of the message (e.g. a PDF, an uploaded file, or inline text), if the type is `"document"`. Nested leaves are `document.url`, `document.content`, `document.mime_type`, `document.id`, and `document.name`. | `{"document.url": "data:application/pdf;base64,JVBERi0...", "document.mime_type": "application/pdf"}` |
| `message_content.encrypted_content` | ![Development](https://img.shields.io/badge/-development-blue) | String | OpenAI `encrypted_content` captured verbatim. | `"gAAAAA...=="` |
| `message_content.id` | ![Development](https://img.shields.io/badge/-development-blue) | String | Provider-assigned identifier for this message content item. For OpenAI Responses reasoning items, this maps to `ResponseReasoningItem.id` and should be preserved for stateless replay. | `"rs_abc123"` |
| `message_content.image` | ![Development](https://img.shields.io/badge/-development-blue) | Image Object | The image content of the message, if the type is `"image"`. | `{"image.url": "https://sample-link-to-image.jpg"}` |
| `message_content.signature` | ![Development](https://img.shields.io/badge/-development-blue) | String | Opaque vendor-issued signature captured verbatim. Maps to provider `signature` fields, and to Gemini `thoughtSignature` fields when the signature is attached to a non-tool content part such as text. | `"EuYBCkQYAiJA..."` |
| `message_content.text` | ![Development](https://img.shields.io/badge/-development-blue) | String | The text content of the message. Used when `message_content.type` is `"text"` or `"reasoning"` (visible reasoning or summary text). For OpenAI `summary_text` arrays, concatenate the array items into this single text value for now. | `"This is a sample text"` |
| `message_content.type` | ![Development](https://img.shields.io/badge/-development-blue) | String | The type of the content. Use `"reasoning"` for reasoning/thinking content, including Anthropic `redacted_thinking` blocks. Use `"tool_use"` when a provider returns a tool call as an ordered content part. | `"text"`, `"image"`, `"audio"`, `"video"`, `"document"`, `"reasoning"`, `"tool_use"` |
| `message_content.video` | ![Development](https://img.shields.io/badge/-development-blue) | Video Object | The video content of the message, if the type is `"video"`. The nested leaf is `video.url`. Consumers infer MIME type from the URL path extension or a data URI prefix. | `{"video.url": "gs://bucket/clip.mp4"}` |

<!-- endweaver -->

## Prompt

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("prompt."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `prompt.id` | ![Development](https://img.shields.io/badge/-development-blue) | String | A vendor-specific id used to identify the prompt. | `"1234"` |
| `prompt.text` | ![Development](https://img.shields.io/badge/-development-blue) | String | The text of a prompt in a completions API request. Used under `llm.prompts`. | `"def fib(n):..."` |
| `prompt.url` | ![Development](https://img.shields.io/badge/-development-blue) | String | A vendor-specific URL used to locate the prompt via the web | `"https://smith.langchain.com/prompts/naive-prompt"` |
| `prompt.vendor` | ![Development](https://img.shields.io/badge/-development-blue) | String | The vendor or origin of the prompt, e.g. `langsmith`, `portkey`, `arize-phoenix` | `"langchain"` |

<!-- endweaver -->

## Completion

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("completion."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `completion.text` | ![Development](https://img.shields.io/badge/-development-blue) | String | The text of a choice returned from a completions API. Used under `llm.choices`. | `" + fib(n-3)..."` |

<!-- endweaver -->

## Tool

On a `TOOL` span, these attributes describe the tool being invoked; under `llm.tools.<index>`, an advertised tool
definition. See [Tool Calling](../content/tool-calling.md).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("tool."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `tool.description` | ![Development](https://img.shields.io/badge/-development-blue) | String | Description of the tool's purpose and functionality. On a `TOOL` span, the tool being invoked; under `llm.tools.<index>`, an advertised tool definition | `"An API to get weather data."` |
| `tool.id` | ![Development](https://img.shields.io/badge/-development-blue) | String | The identifier for the result of the tool call (corresponding to `tool_call.id`) | `"call_62136355"` |
| `tool.json_schema` | ![Development](https://img.shields.io/badge/-development-blue) | JSON String | The JSON schema of a tool input | `"{'type': 'function', 'function': {'name': 'get_weather'}}"` |
| `tool.name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The name of the tool. On a `TOOL` span, the tool being invoked; under `llm.tools.<index>`, the name of an advertised tool definition | `"WeatherAPI"` |
| `tool.parameters` | ![Development](https://img.shields.io/badge/-development-blue) | JSON String | The parameters definition for invoking the tool | `"{ 'a': 'int' }"` |

<!-- endweaver -->

## Tool Call

Items of `message.tool_calls`, and of `message.contents` when `message_content.type` is `"tool_use"`. See
[Tool Calling](../content/tool-calling.md).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("tool_call."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `tool_call.function.arguments` | ![Development](https://img.shields.io/badge/-development-blue) | JSON String | The arguments for the function being invoked by a tool call | `"{'city': 'London'}"` |
| `tool_call.function.name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The name of the function being invoked by a tool call | `"get_current_weather"` |
| `tool_call.id` | ![Development](https://img.shields.io/badge/-development-blue) | String | The id of a tool call (useful when there is more than one call at the same time) | `"call_62136355"` |
| `tool_call.reasoning_signature` | ![Development](https://img.shields.io/badge/-development-blue) | String | Opaque vendor-issued reasoning echo token attached to a tool call. Used by Gemini, which puts `thoughtSignature` on `functionCall` parts rather than on the reasoning summary. | `"CiQB..."` |

<!-- endweaver -->

## Image

Composed under `message_content.image` and `<input|output>.images`. See [Multimodal Content](../content/multimodal.md).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("image."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `image.url` | ![Development](https://img.shields.io/badge/-development-blue) | String | The link to the image or its base64 encoding | `"https://sample-link-to-image.jpg"` |

<!-- endweaver -->

## Audio

Composed under `message_content.audio`. See [Multimodal Content](../content/multimodal.md#audio-content).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("audio."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `audio.mime_type` | ![Development](https://img.shields.io/badge/-development-blue) | String | The MIME type of the audio file. | `"audio/mpeg"` |
| `audio.transcript` | ![Development](https://img.shields.io/badge/-development-blue) | String | The transcript of the audio file (e.g. a Whisper transcription) | `"Hello, how are you?"` |
| `audio.url` | ![Development](https://img.shields.io/badge/-development-blue) | String | The URL of an audio file (e.g. cloud storage), or a base64 data URI | `"https://storage.com/buckets/1/file.wav"` |

<!-- endweaver -->

## Video

Composed under `message_content.video`. See [Multimodal Content](../content/multimodal.md#video-content).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("video."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `video.url` | ![Development](https://img.shields.io/badge/-development-blue) | String | The URL, object-store URI, or base64 data URI of a video. Consumers infer MIME type from the path extension (`.mp4` to `video/mp4`, `.webm` to `video/webm`, `.ogv` to `video/ogg`, `.avi` to `video/x-msvideo`) or from a data URI prefix. | `"gs://bucket/clip.mp4"` |

<!-- endweaver -->

## Document

Items of `retrieval.documents`, `reranker.input_documents`, `reranker.output_documents`, and
`<input|output>.documents`, and the leaves of `message_content.document`. See
[Document Content](../content/multimodal.md#document-content).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("document."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `document.content` | ![Development](https://img.shields.io/badge/-development-blue) | String | The text content of a document: a retrieved passage, or an inline text document passed to a model (e.g. an Anthropic `text` source) | `"This is a sample document content."` |
| `document.id` | ![Development](https://img.shields.io/badge/-development-blue) | String/Integer | Unique identifier for a document: a retrieved document or chunk in a retrieval system, or the provider file id (e.g. an OpenAI or Anthropic Files API id) for a document passed to a model. Instrumentations MAY record an integer id; consumers SHOULD accept both strings and integers. | `"1234"`, `"1"`, `"file-abc123"` |
| `document.metadata` | ![Development](https://img.shields.io/badge/-development-blue) | JSON String | Metadata associated with a document | `"{'author': 'John Doe', 'date': '2023-09-09'}"` |
| `document.mime_type` | ![Development](https://img.shields.io/badge/-development-blue) | String | The MIME type of a file document such as a PDF | `"application/pdf"` |
| `document.name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The name supplied with a document, copied verbatim: a file name (OpenAI `filename`) or a title (Anthropic `title`, Bedrock `name`, Gemini `display_name`); not a unique identifier | `"invoice.pdf"` |
| `document.score` | ![Development](https://img.shields.io/badge/-development-blue) | Float | Score representing the relevance of a document | `0.98` |
| `document.url` | ![Development](https://img.shields.io/badge/-development-blue) | String | The URL, object-store URI, or base64 data URI of a file document such as a PDF | `"https://example.com/report.pdf"` |

<!-- endweaver -->

## Retrieval

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("retrieval."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `retrieval.documents` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | List of retrieved documents | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]` |

<!-- endweaver -->

## Reranker

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("reranker."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `reranker.input_documents` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | List of documents as input to the reranker | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]` |
| `reranker.model_name` | ![Development](https://img.shields.io/badge/-development-blue) | String | Model name of the reranker | `"cross-encoder/ms-marco-MiniLM-L-12-v2"` |
| `reranker.output_documents` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | List of documents outputted by the reranker | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]` |
| `reranker.query` | ![Development](https://img.shields.io/badge/-development-blue) | String | Query parameter of the reranker | `"How to format timestamp?"` |
| `reranker.top_k` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | Top K parameter of the reranker | `3` |

<!-- endweaver -->

## Embedding

See [Embedding Spans](../spans/embedding.md).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("embedding."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `embedding.embeddings` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | List of embedding objects including text and vector data | `[{"embedding.vector": [...], "embedding.text": "hello"}]` |
| `embedding.invocation_parameters` | ![Development](https://img.shields.io/badge/-development-blue) | JSON String | Parameters used during the invocation of an embedding model or API (excluding input) | `"{\"model\": \"text-embedding-3-small\", \"encoding_format\": \"float\"}"` |
| `embedding.model_name` | ![Development](https://img.shields.io/badge/-development-blue) | String | Name of the embedding model used | `"BERT-base"` |
| `embedding.text` | ![Development](https://img.shields.io/badge/-development-blue) | String | The text represented in the embedding | `"hello world"` |
| `embedding.vector` | ![Development](https://img.shields.io/badge/-development-blue) | List of floats | The embedding vector consisting of a list of floats | `[0.123, 0.456]` |

<!-- endweaver -->

## Decision

See [Decision Spans](../spans/decision.md).

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("decision."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `decision.model_name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The name of the decision model being utilized. Mirrors `llm.model_name` for `DECISION` spans. | `"jev-1.13.0"` |
| `decision.provider` | ![Development](https://img.shields.io/badge/-development-blue) | String | The hosting provider of the decision model: who runs the hardware that answered, e.g. `typesafe` when calling TypeSafe directly, `openai` for the OpenAI Decisions API, or the cloud or self-hosting provider for a model served elsewhere. Distinct from `decision.system`, which names the API shape. The well-known values alias the matching `llm.provider` values. | `"typesafe"` |
| `decision.request.model_name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The decision model requested by the caller, as sent in the request. May differ from `decision.response.model_name` when the provider resolves an alias or routes the request. | `"jev-latest"` |
| `decision.response.model_name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The decision model that actually produced the response, as reported by the provider. May differ from `decision.request.model_name`. | `"jev-1.13.0"` |
| `decision.system` | ![Development](https://img.shields.io/badge/-development-blue) | String | The decision API ecosystem the call conforms to, i.e. which request/response shape the client speaks, as identified by the client or server instrumentation: `typesafe` for the TypeSafe System One / Jev API, `openai` for the OpenAI Decisions API. A self-hosted vLLM server answering the Jev-compatible `/v1/systemone` shape is still `typesafe`. Distinct from `decision.provider`, which says who hosts the model. The well-known values alias the matching `llm.system` values. | `"typesafe"` |
| `decision.token_count.input` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | The number of input tokens consumed by a decision model call: the state, questions, and candidate options. Maps to `usage.input_tokens` in TypeSafe System One responses. | `412` |
| `decision.token_count.output` | ![Development](https://img.shields.io/badge/-development-blue) | Integer | The number of output tokens produced by a decision model call. Decision models emit typed answers rather than text, so this is typically small. Maps to `usage.output_tokens` in TypeSafe System One responses. | `2` |

<!-- endweaver -->

### Decision System and Provider Values

`decision.system` and `decision.provider` draw from the same identifier space as `llm.system` and `llm.provider`: the
same string names the same vendor in both attribute families, and the semantic convention packages expose the
decision values as aliases of the matching LLM values (for example `OpenInferenceDecisionSystemValues.TYPESAFE` is
`OpenInferenceLLMSystemValues.TYPESAFE`). The decision lists are the subset of vendors currently known to offer a
decision API. They are [well-known values](./conventions.md#well-known-values).

`decision.system`:

<!-- weaver .registry.attributes[] | select(.key == "decision.system") | {members: .type.members} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Value | Description | Stability |
| --- | --- | --- |
| `typesafe` | [TypeSafe AI System One / Jev API](https://typesafe.ai/blog/introducing-system-one-models-and-jev), including Jev-compatible servers such as vLLM's `/v1/systemone` | ![Development](https://img.shields.io/badge/-development-blue) |
| `openai` | [OpenAI Decisions API](https://openai.com/index/devday-2026-recap/) | ![Development](https://img.shields.io/badge/-development-blue) |

<!-- endweaver -->

`decision.provider`:

<!-- weaver .registry.attributes[] | select(.key == "decision.provider") | {members: .type.members} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Value | Description | Stability |
| --- | --- | --- |
| `typesafe` | TypeSafe AI | ![Development](https://img.shields.io/badge/-development-blue) |
| `openai` | OpenAI | ![Development](https://img.shields.io/badge/-development-blue) |

<!-- endweaver -->

vLLM's native `/v1/decisions` API ([RFC](https://github.com/vllm-project/vllm/issues/59365)) does not yet have a
well-known value; a custom value MAY be used until it is added.

## Agent

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("agent."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `agent.name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The name of the agent that this span represents. | `"researcher"` |

<!-- endweaver -->

## Graph

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("graph."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `graph.node.id` | ![Development](https://img.shields.io/badge/-development-blue) | String | The id of the node in the execution graph. This along with `graph.node.parent_id` are used to visualize the execution graph. | `"search_api_0"` |
| `graph.node.name` | ![Development](https://img.shields.io/badge/-development-blue) | String | The name of the node in the execution graph. Use this to present a human readable name for the node. Optional | `"Search API"` |
| `graph.node.parent_id` | ![Development](https://img.shields.io/badge/-development-blue) | String | This references the id of the parent node. Leaving this unset or set as empty string implies that the current span is the root node. | `"router_0"` |

<!-- endweaver -->

## Annotation and Evaluation

Feedback from human, LLM, and code judges. The `annotation.*` and `evaluation.*` forms are equivalent; see
[Annotations and Evaluations](../feedback/annotations.md).

### Feedback Lists

<!-- weaver ["annotations", "trace.annotations", "session.annotations", "evaluations", "trace.evaluations", "session.evaluations"] as $keys | {attributes: [$keys[] as $k | .registry.attributes[] | select(.key == $k)]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `annotations` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | Span-scoped feedback using annotation terminology, flattened as `annotations.0.annotation.*` | `[{"annotation.name": "hallucination", "annotation.label": "hallucinated"}]` |
| `trace.annotations` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | Trace-scoped feedback using annotation terminology, flattened as `trace.annotations.0.annotation.*` | `[{"annotation.name": "retrieval_quality", "annotation.score": 0.92}]` |
| `session.annotations` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | Session-scoped feedback using annotation terminology, flattened as `session.annotations.0.annotation.*`; requires `session.id` | `[{"annotation.name": "conversational_coherence", "annotation.label": "coherent"}]` |
| `evaluations` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | Span-scoped feedback using evaluation or eval terminology, flattened as `evaluations.0.evaluation.*` | `[{"evaluation.name": "relevance", "evaluation.score": 0.95}]` |
| `trace.evaluations` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | Trace-scoped feedback using evaluation terminology, flattened as `trace.evaluations.0.evaluation.*` | `[{"evaluation.name": "retrieval_quality", "evaluation.score": 0.92}]` |
| `session.evaluations` | ![Development](https://img.shields.io/badge/-development-blue) | List of objects | Session-scoped feedback using evaluation terminology, flattened as `session.evaluations.0.evaluation.*`; requires `session.id` | `[{"evaluation.name": "conversational_coherence", "evaluation.label": "coherent"}]` |

<!-- endweaver -->

### Result Fields

Each item carries these fields, under `annotation.*` or `evaluation.*`.

<!-- weaver {attributes: [.registry.attributes[] | select(.key | startswith("annotation.") or startswith("evaluation."))]} -->
<!-- Generated from semantic-conventions/model. Do not edit by hand: change the model and run `make -C semantic-conventions generate`. -->

| Attribute | Stability | Type | Description | Examples |
| --- | --- | --- | --- | --- |
| `annotation.annotator_kind` | ![Development](https://img.shields.io/badge/-development-blue) | String | Judge kind: `HUMAN`, `LLM`, `CODE`, or custom. See [Annotator Kinds](../feedback/annotations.md#annotator-kinds) | `"LLM"`, `"HUMAN"` |
| `annotation.explanation` | ![Development](https://img.shields.io/badge/-development-blue) | String | Reason or evidence for the result | `"The claim is not supported by the context."` |
| `annotation.identifier` | ![Development](https://img.shields.io/badge/-development-blue) | String | Stable result ID for the same name and target | `"judge-v2"`, `"reviewer-42"` |
| `annotation.label` | ![Development](https://img.shields.io/badge/-development-blue) | String | Categorical result | `"hallucinated"`, `"relevant"` |
| `annotation.metadata` | ![Development](https://img.shields.io/badge/-development-blue) | JSON String | Additional result or judge data | `"{\"evaluator\":\"judge-v2\"}"` |
| `annotation.name` | ![Development](https://img.shields.io/badge/-development-blue) | String | Criterion or metric name | `"hallucination"`, `"relevance"` |
| `annotation.score` | ![Development](https://img.shields.io/badge/-development-blue) | Integer/Float | Numeric result. Instrumentations MAY record an integer; consumers SHOULD accept both integers and floats. | `1.0`, `0.95` |
| `evaluation.annotator_kind` | ![Development](https://img.shields.io/badge/-development-blue) | String | Judge kind: `HUMAN`, `LLM`, `CODE`, or custom. See [Annotator Kinds](../feedback/annotations.md#annotator-kinds) | `"LLM"`, `"HUMAN"` |
| `evaluation.explanation` | ![Development](https://img.shields.io/badge/-development-blue) | String | Reason or evidence for the result | `"The claim is not supported by the context."` |
| `evaluation.identifier` | ![Development](https://img.shields.io/badge/-development-blue) | String | Stable result ID for the same name and target | `"judge-v2"`, `"reviewer-42"` |
| `evaluation.label` | ![Development](https://img.shields.io/badge/-development-blue) | String | Categorical result | `"hallucinated"`, `"relevant"` |
| `evaluation.metadata` | ![Development](https://img.shields.io/badge/-development-blue) | JSON String | Additional result or judge data | `"{\"evaluator\":\"judge-v2\"}"` |
| `evaluation.name` | ![Development](https://img.shields.io/badge/-development-blue) | String | Criterion or metric name | `"hallucination"`, `"relevance"` |
| `evaluation.score` | ![Development](https://img.shields.io/badge/-development-blue) | Integer/Float | Numeric result. Instrumentations MAY record an integer; consumers SHOULD accept both integers and floats. | `1.0`, `0.95` |

<!-- endweaver -->

## Exception

These attributes are defined by OpenTelemetry's
[exception semantic conventions](https://opentelemetry.io/docs/specs/semconv/exceptions/exceptions-spans/) and recorded
on an `exception` span event.

| Attribute | Type | Description | Examples |
| --- | --- | --- | --- |
| `exception.type` | String | The type of exception that was thrown | `"NullPointerException"` |
| `exception.message` | String | Detailed message describing the exception | `"Null value encountered"` |
| `exception.stacktrace` | String | The stack trace of the exception | `"at app.main(app.java:16)"` |
| `exception.escaped` | Boolean | Indicator if the exception has escaped the span's scope | `true` |
