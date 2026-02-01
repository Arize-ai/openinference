# Semantic Conventions

The **Semantic Conventions** define the keys and values which describe commonly observed concepts, protocols, and
operations used by applications. These conventions are used to populate the `attributes` of `spans` and span `events`.

## Span Kinds

The `openinference.span.kind` attribute is **required** for all OpenInference spans and identifies the type of operation being traced. The span kind provides a hint to the tracing backend as to how the trace should be assembled. Valid values include:

| Span Kind Value | Description                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LLM`           | A span that represents a call to a Large Language Model (LLM). For example, an LLM span could be used to represent a call to OpenAI or Llama for chat completions or text generation.                                                                                                                                                                                                                                                       |
| `EMBEDDING`     | A span that represents a call to an LLM or embedding service for generating embeddings. For example, an Embedding span could be used to represent a call to OpenAI to get an ada embedding for retrieval.                                                                                                                                                                                                                                   |
| `CHAIN`         | A span that represents a starting point or a link between different LLM application steps. For example, a Chain span could be used to represent the beginning of a request to an LLM application or the glue code that passes context from a retriever to an LLM call.                                                                                                                                                                      |
| `RETRIEVER`     | A span that represents a data retrieval step. For example, a Retriever span could be used to represent a call to a vector store or a database to fetch documents or information.                                                                                                                                                                                                                                                            |
| `RERANKER`      | A span that represents the reranking of a set of input documents. For example, a cross-encoder may be used to compute the input documents' relevance scores with respect to a user query, and the top K documents with the highest scores are then returned by the Reranker.                                                                                                                                                                |
| `TOOL`          | A span that represents a call to an external tool such as a calculator, weather API, or any function execution that is invoked by an LLM or agent.                                                                                                                                                                                                                                                                                          |
| `AGENT`         | A span that encompasses calls to LLMs and Tools. An agent describes a reasoning block that acts on tools using the guidance of an LLM.                                                                                                                                                                                                                                                                                                      |
| `GUARDRAIL`     | A span that represents calls to a component to protect against jailbreak user input prompts by taking action to modify or reject an LLM's response if it contains undesirable content. For example, a Guardrail span could involve checking if an LLM's output response contains inappropriate language, via a custom or external guardrail library, and then amending the LLM response to remove references to the inappropriate language. |
| `EVALUATOR`     | A span that represents a call to a function or process performing an evaluation of the language model's outputs. Examples include assessing the relevance, correctness, or helpfulness of the language model's answers.                                                                                                                                                                                                                     |
| `PROMPT`        | A span that represents the rendering of a prompt template. For example, a Prompt span could be used to represent the rendering a template with variables.                                                                                                                                                                                                                                                                                   |

## Reserved Attributes

The following attributes are reserved and MUST be supported by all OpenInference Tracing SDKs:

| Attribute                                      | Type                        | Example                                                                                                                       | Description                                                                                                                                                                         |
| ---------------------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `document.content`                             | String                      | `"This is a sample document content."`                                                                                        | The content of a retrieved document                                                                                                                                                 |
| `document.id`                                  | String/Integer              | `"1234"` or `1`                                                                                                               | Unique identifier for a document                                                                                                                                                    |
| `document.metadata`                            | JSON String                 | `"{'author': 'John Doe', 'date': '2023-09-09'}"`                                                                              | Metadata associated with a document                                                                                                                                                 |
| `document.score`                               | Float                       | `0.98`                                                                                                                        | Score representing the relevance of a document                                                                                                                                      |
| `embedding.embeddings`                         | List of objects<sup>†</sup> | `[{"embedding.vector": [...], "embedding.text": "hello"}]`                                                                    | List of embedding objects including text and vector data                                                                                                                            |
| `embedding.invocation_parameters`              | JSON String                 | `"{\"model\": \"text-embedding-3-small\", \"encoding_format\": \"float\"}"`                                                   | Parameters used during the invocation of an embedding model or API (excluding input)                                                                                                |
| `embedding.model_name`                         | String                      | `"BERT-base"`                                                                                                                 | Name of the embedding model used                                                                                                                                                    |
| `embedding.text`                               | String                      | `"hello world"`                                                                                                               | The text represented in the embedding                                                                                                                                               |
| `embedding.vector`                             | List of floats              | `[0.123, 0.456, ...]`                                                                                                         | The embedding vector consisting of a list of floats                                                                                                                                 |
| `exception.escaped`                            | Boolean                     | `true`                                                                                                                        | Indicator if the exception has escaped the span's scope                                                                                                                             |
| `exception.message`                            | String                      | `"Null value encountered"`                                                                                                    | Detailed message describing the exception                                                                                                                                           |
| `exception.stacktrace`                         | String                      | `"at app.main(app.java:16)"`                                                                                                  | The stack trace of the exception                                                                                                                                                    |
| `exception.type`                               | String                      | `"NullPointerException"`                                                                                                      | The type of exception that was thrown                                                                                                                                               |
| `image.url`                                    | String                      | `"https://sample-link-to-image.jpg"`                                                                                          | The link to the image or its base64 encoding                                                                                                                                        |
| `input.mime_type`                              | String                      | `"text/plain"` or `"application/json"`                                                                                        | MIME type representing the format of `input.value`                                                                                                                                  |
| `input.value`                                  | String                      | `"{'query': 'What is the weather today?'}"`                                                                                   | The input value to an operation                                                                                                                                                     |
| `llm.prompts`                                  | List of objects<sup>†</sup> | `[{"prompt.text": "def fib(n):..."}]`                                                                                         | Prompts provided to a completions API                                                                                                                                               |
| `llm.choices`                                  | List of objects<sup>†</sup> | `[{"completion.text": " + fib(n-3)..."}]`                                                                                     | Text choices returned from a completions API                                                                                                                                        |
| `llm.function_call`                            | JSON String                 | `"{function_name: 'add', args: [1, 2]}"`                                                                                      | Object recording details of a function call in models or APIs                                                                                                                       |
| `llm.input_messages`                           | List of objects<sup>†</sup> | `[{"message.role": "user", "message.content": "hello"}]`                                                                      | List of messages sent to the LLM in a chat API request. Uses flattened attributes with indexed prefixes (e.g., `llm.input_messages.0.message.role`)                                 |
| `llm.invocation_parameters`                    | JSON string                 | `"{model_name: 'gpt-3', temperature: 0.7}"`                                                                                   | Parameters used during the invocation of an LLM or API                                                                                                                              |
| `llm.provider`                                 | String                      | `openai`, `azure`                                                                                                             | The hosting provider of the llm, e.x. `azure`                                                                                                                                       |
| `llm.system`                                   | String                      | `anthropic`, `openai`                                                                                                         | The AI product as identified by the client or server instrumentation.                                                                                                               |
| `llm.model_name`                               | String                      | `"gpt-3.5-turbo"`                                                                                                             | The name of the language model being utilized                                                                                                                                       |
| `llm.output_messages`                          | List of objects<sup>†</sup> | `[{"message.role": "assistant", "message.content": "hello"}]`                                                                 | List of messages received from the LLM in a chat API response. Uses flattened attributes with indexed prefixes (e.g., `llm.output_messages.0.message.role`)                         |
| `llm.prompt_template.template`                 | String                      | `"Weather forecast for {city} on {date}"`                                                                                     | Template used to generate prompts as Python f-strings                                                                                                                               |
| `llm.prompt_template.variables`                | JSON String                 | `{ context: "<context from retrieval>", subject: "math" }`                                                                    | JSON of key value pairs applied to the prompt template                                                                                                                              |
| `llm.prompt_template.version`                  | String                      | `"v1.0"`                                                                                                                      | The version of the prompt template                                                                                                                                                  |
| `llm.token_count.completion`                   | Integer                     | `15`                                                                                                                          | The number of tokens in the completion                                                                                                                                              |
| `llm.token_count.completion_details.reasoning` | Integer                     | `10`                                                                                                                          | The number of tokens used for model reasoning                                                                                                                                       |
| `llm.token_count.completion_details.audio`     | Integer                     | `10`                                                                                                                          | The number of audio input tokens generated by the model                                                                                                                             |
| `llm.token_count.prompt`                       | Integer                     | `10`                                                                                                                          | The number of tokens in the prompt                                                                                                                                                  |
| `llm.token_count.prompt_details.cache_read`    | Integer                     | `5`                                                                                                                           | The number of prompt tokens successfully retrieved from cache (cache hits). Maps to `cached_tokens` in OpenAI responses                                                             |
| `llm.token_count.prompt_details.cache_write`   | Integer                     | `0`                                                                                                                           | The number of prompt tokens not found in cache that were written to cache (cache misses). Specific to Anthropic Claude                                                              |
| `llm.token_count.prompt_details.audio`         | Integer                     | `10`                                                                                                                          | The number of audio input tokens presented in the prompt                                                                                                                            |
| `llm.token_count.total`                        | Integer                     | `20`                                                                                                                          | Total number of tokens, including prompt and completion                                                                                                                             |
| `llm.cost.prompt`                              | Float                       | `0.0021`                                                                                                                      | Total cost of all input tokens sent to the LLM in USD                                                                                                                               |
| `llm.cost.completion`                          | Float                       | `0.0045`                                                                                                                      | Total cost of all output tokens generated by the LLM in USD                                                                                                                         |
| `llm.cost.total`                               | Float                       | `0.0066`                                                                                                                      | Total cost of the LLM call in USD (prompt + completion)                                                                                                                             |
| `llm.cost.prompt_details.input`                | Float                       | `0.0003`                                                                                                                      | Total cost of input tokens in USD                                                                                                                                                   |
| `llm.cost.completion_details.output`           | Float                       | `0.0009`                                                                                                                      | Total cost of output tokens in USD                                                                                                                                                  |
| `llm.cost.completion_details.reasoning`        | Float                       | `0.0024`                                                                                                                      | Cost of reasoning steps in the completion in USD                                                                                                                                    |
| `llm.cost.completion_details.audio`            | Float                       | `0.0012`                                                                                                                      | Cost of audio tokens in the completion in USD                                                                                                                                       |
| `llm.cost.prompt_details.cache_write`          | Float                       | `0.0006`                                                                                                                      | Cost of prompt tokens written to cache in USD                                                                                                                                       |
| `llm.cost.prompt_details.cache_read`           | Float                       | `0.0003`                                                                                                                      | Cost of prompt tokens read from cache in USD                                                                                                                                        |
| `llm.cost.prompt_details.cache_input`          | Float                       | `0.0006`                                                                                                                      | Cost of input tokens in the prompt that were cached in USD                                                                                                                          |
| `llm.cost.prompt_details.audio`                | Float                       | `0.0003`                                                                                                                      | Cost of audio tokens in the prompt in USD                                                                                                                                           |
| `llm.tools`                                    | List of objects<sup>†</sup> | `[{"tool.json_schema": "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\"}}"}]`                               | List of tools that are advertised to the LLM to be able to call. Uses flattened attributes with indexed prefixes (e.g., `llm.tools.0.tool.json_schema`)                             |
| `message.content`                              | String                      | `"What's the weather today?"`                                                                                                 | The content of a message in a chat                                                                                                                                                  |
| `message.contents`                             | List of objects<sup>†</sup> | `[{"message_content.type": "text", "message_content.text": "Hello"}, ...]`                                                    | The message contents to the llm, it is an array of `message_content` objects.                                                                                                       |
| `message.function_call_arguments_json`         | JSON String                 | `"{ 'x': 2 }"`                                                                                                                | The arguments to the function call in JSON                                                                                                                                          |
| `message.function_call_name`                   | String                      | `"multiply"` or `"subtract"`                                                                                                  | Function call function name                                                                                                                                                         |
| `message.tool_call_id`                         | String                      | `"call_62136355"`                                                                                                             | Tool call result identifier corresponding to `tool_call.id`                                                                                                                         |
| `message.role`                                 | String                      | `"user"` or `"system"`                                                                                                        | Role of the entity in a message (e.g., user, system)                                                                                                                                |
| `message.tool_calls`                           | List of objects<sup>†</sup> | `[{"tool_call.function.name": "get_current_weather"}]`                                                                        | List of tool calls (e.g. function calls) generated by the LLM. When in output messages, uses flattened attributes (e.g., `llm.output_messages.0.message.tool_calls.0.tool_call.id`) |
| `messagecontent.type`                          | String                      | `"text"` or `"image"`                                                                                                         | The type of the content, such as "text" or "image".                                                                                                                                 |
| `messagecontent.text`                          | String                      | `"This is a sample text"`                                                                                                     | The text content of the message, if the type is "text".                                                                                                                             |
| `messagecontent.image`                         | Image Object                | `{"image.url": "https://sample-link-to-image.jpg"}`                                                                           | The image content of the message, if the type is "image".                                                                                                                           |
| `metadata`                                     | JSON String                 | `"{'author': 'John Doe', 'date': '2023-09-09'}"`                                                                              | Metadata associated with a span                                                                                                                                                     |
| `openinference.span.kind`                      | String                      | `"LLM"`, `"EMBEDDING"`, `"CHAIN"`, `"RETRIEVER"`, `"RERANKER"`, `"TOOL"`, `"AGENT"`, `"GUARDRAIL"`, `"EVALUATOR"`, `"PROMPT"` | Required for all OpenInference spans. Identifies the type of operation. See [Span Kinds](#span-kinds) for detailed descriptions of each kind.                                       |
| `output.mime_type`                             | String                      | `"text/plain"` or `"application/json"`                                                                                        | MIME type representing the format of `output.value`                                                                                                                                 |
| `output.value`                                 | String                      | `"Hello, World!"`                                                                                                             | The output value of an operation                                                                                                                                                    |
| `reranker.input_documents`                     | List of objects<sup>†</sup> | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]`                                                    | List of documents as input to the reranker                                                                                                                                          |
| `reranker.model_name`                          | String                      | `"cross-encoder/ms-marco-MiniLM-L-12-v2"`                                                                                     | Model name of the reranker                                                                                                                                                          |
| `reranker.output_documents`                    | List of objects<sup>†</sup> | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]`                                                    | List of documents outputted by the reranker                                                                                                                                         |
| `reranker.query`                               | String                      | `"How to format timestamp?"`                                                                                                  | Query parameter of the reranker                                                                                                                                                     |
| `reranker.top_k`                               | Integer                     | 3                                                                                                                             | Top K parameter of the reranker                                                                                                                                                     |
| `retrieval.documents`                          | List of objects<sup>†</sup> | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]`                                                    | List of retrieved documents                                                                                                                                                         |
| `session.id`                                   | String                      | `"26bcd3d2-cad2-443d-a23c-625e47f3324a"`                                                                                      | Unique identifier for a session                                                                                                                                                     |
| `tag.tags`                                     | List of strings             | ["shopping", "travel"]                                                                                                        | List of tags to give the span a category                                                                                                                                            |
| `tool.description`                             | String                      | `"An API to get weather data."`                                                                                               | Description of the tool's purpose and functionality                                                                                                                                 |
| `tool.json_schema`                             | JSON String                 | `"{'type': 'function', 'function': {'name': 'get_weather'}}"`                                                                 | The json schema of a tool input                                                                                                                                                     |
| `tool.name`                                    | String                      | `"WeatherAPI"`                                                                                                                | The name of the tool being utilized                                                                                                                                                 |
| `tool.id`                                      | String                      | `"WeatherAPI"`                                                                                                                | The identifier for the result of the tool call (corresponding to `tool_call.id`)                                                                                                    |
| `tool.parameters`                              | JSON string                 | `"{ 'a': 'int' }"`                                                                                                            | The parameters definition for invoking the tool                                                                                                                                     |
| `tool_call.function.arguments`                 | JSON string                 | `"{'city': 'London'}"`                                                                                                        | The arguments for the function being invoked by a tool call                                                                                                                         |
| `tool_call.function.name`                      | String                      | `"get_current_weather"`                                                                                                       | The name of the function being invoked by a tool call                                                                                                                               |
| `tool_call.id`                                 | string                      | `"call_62136355"`                                                                                                             | The id of the a tool call (useful when there are more than one call at the same time)                                                                                               |
| `user.id`                                      | String                      | `"9328ae73-7141-4f45-a044-8e06192aa465"`                                                                                      | Unique identifier for a user                                                                                                                                                        |
| `audio.url`                                    | String                      | `https://storage.com/buckets/1/file.wav`                                                                                      | The url to an audio file (e.x. cloud storage)                                                                                                                                       |
| `audio.mime_type`                              | String                      | `audio/mpeg`                                                                                                                  | The mime type of the audio file (e.x. `audio/mpeg`, `audio/wav` )                                                                                                                   |
| `audio.transcript`                             | String                      | `"Hello, how are you?"`                                                                                                       | The transcript of the audio file (e.x. whisper transcription)                                                                                                                       |
| `prompt.vendor`                                | String                      | `"langchain"`                                                                                                                 | The vendor or origin of the prompt, (e.x. 'langsmith', 'portkey' 'arize-phoenix', etc)                                                                                              |
| `prompt.id`                                    | String                      | `"1234"`                                                                                                                      | A vendor-specific id used to identify the prompt.                                                                                                                                   |
| `prompt.url`                                   | String                      | `https://smith.langchain.com/prompts/naive-prompt`                                                                            | A vendor-specific URL used to locate the prompt via the web                                                                                                                         |
| `agent.name`                                   | String                      | `researcher`                                                                                                                  | The name of the agent that this span represents.                                                                                                                                    |
| `graph.node.id`                                | String                      | `search_api_0`                                                                                                                | The id of the node in the execution graph. This along with graph.node.parent_id are used to visualize the execution graph.                                                          |
| `graph.node.name`                              | String                      | `Search API`                                                                                                                  | The name of the node in the execution graph. Use this to present a human readable name for the node. Optional                                                                       |
| `graph.node.parent_id`                         | String                      | `router_0`                                                                                                                    | This references the id of the parent node. Leaving this unset or set as empty string implies that the current span is the root node.                                                |

<sup>†</sup> To get a list of objects exported as OpenTelemetry span attributes, flattening of the list is necessary as
shown in the examples below. All list-based attributes use zero-based indexing in their flattened form.

`llm.system` has the following list of well-known values. If one of them applies, then the respective value MUST be
used; otherwise, a custom value MAY be used.

| Value       | Description |
| ----------- | ----------- |
| `anthropic` | Anthropic   |
| `openai`    | OpenAI      |
| `vertexai`  | Vertex AI   |
| `cohere`    | Cohere      |
| `mistralai` | Mistral AI  |

`llm.provider` has the following list of well-known values. If one of them applies, then the respective value MUST be
used; otherwise, a custom value MAY be used.

| Value       | Description     |
| ----------- | --------------- |
| `anthropic` | Anthropic       |
| `openai`    | OpenAI          |
| `cohere`    | Cohere          |
| `mistralai` | Mistral AI      |
| `azure`     | Azure           |
| `google`    | Google (Vertex) |
| `aws`       | AWS Bedrock     |

### Token Count Details

`llm.token_count.prompt_details.cache_read` and `llm.token_count.prompt_details.cache_write` provide granular token count information for cache operations, enabling detailed API usage tracking and cost analysis.

- `cache_read` represents the number of prompt tokens successfully retrieved from cache (cache hits). For OpenAI, this
  corresponds to the `usage.prompt_tokens_details.cached_tokens` field in completion API responses. For Anthropic, when
  using a cache_control block, this maps to the `cache_read_input_tokens` field in Messages API responses.
- `cache_write` represents the number of prompt tokens not found in cache (cache misses) that were subsequently written
  to cache. This metric is specific to Anthropic and corresponds to the `cache_write_input_tokens` field in their
  Messages API responses.

The extended token count attributes follow the naming pattern:

- `llm.token_count.prompt_details.*` for prompt-related token counts
- `llm.token_count.completion_details.*` for completion-related token counts

These attributes enable:

- Tracking of multimodal token usage (audio tokens)
- Monitoring reasoning token consumption for models with chain-of-thought capabilities
- Cache efficiency analysis for cost optimization
- Detailed billing reconciliation

Note: All token count attributes store integer values representing the count of tokens. Cost attributes store floating point values in USD currency.

### System and Model Identification

The `llm.system` attribute identifies the AI product/vendor, while `llm.model_name` contains the specific model identifier:

- `llm.system` should use well-known values when applicable (e.g., "openai", "anthropic", "cohere")
- `llm.model_name` should contain the actual model name returned by the API (e.g., "gpt-4-0613", "claude-3-opus-20240229")
- The `llm.provider` attribute can be used to identify the hosting provider when different from the system (e.g., "azure" for Azure-hosted OpenAI)

**For embedding operations (`openinference.span.kind: "EMBEDDING"`):**

- `llm.system` and `llm.provider` are **not used**
- Use `embedding.model_name` to identify the embedding model (e.g., "text-embedding-3-small", "text-embedding-ada-002")
- See the [Embedding Spans](./embedding_spans.md#attributes-not-used-in-embedding-spans) specification for the rationale

The `llm.cost` prefix is used to group cost-related attributes. When these keys are transformed into a JSON-like structure, it would look like:

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

## Attribute Naming Conventions

### Indexed Attribute Prefixes

When dealing with lists of structured data, OpenInference uses indexed prefixes to create flattened attribute names. The general pattern is:

`<prefix>.<index>.<suffix>`

Where:

- `<prefix>` is the base attribute name (e.g., `llm.input_messages`, `llm.tools`)
- `<index>` is a zero-based integer index
- `<suffix>` is the nested attribute path

### Common Flattened Attribute Patterns

#### LLM Input/Output Messages

- `llm.input_messages.<index>.message.role` - Role of the message (e.g., "user", "assistant", "system")
- `llm.input_messages.<index>.message.content` - Text content of the message
- `llm.output_messages.<index>.message.role` - Role of the output message
- `llm.output_messages.<index>.message.content` - Text content of the output message

#### Completions API (Legacy Text Completion)

For the legacy completions API (non-chat):

- `llm.prompts.<index>.prompt.text` - Input prompt(s) provided to the completions API (e.g., `llm.prompts.0.prompt.text`, `llm.prompts.1.prompt.text`)
- `llm.choices.<index>.completion.text` - Text choice(s) returned from the completions API (e.g., `llm.choices.0.completion.text`, `llm.choices.1.completion.text`)

These attributes use a nested indexed format with discriminated union structure, allowing for future expansion. The nested structure (`.prompt.text` and `.completion.text`) mirrors the pattern used for `llm.input_messages` and `llm.output_messages`.

#### Message Content Arrays (Multimodal)

For messages containing multiple content items (text, images, audio):

- `llm.input_messages.<messageIndex>.message.contents.<contentIndex>.message_content.text` - Text content item
- `llm.input_messages.<messageIndex>.message.contents.<contentIndex>.message_content.type` - Content type ("text", "image", "audio")
- `llm.input_messages.<messageIndex>.message.contents.<contentIndex>.message_content.image.image.url` - Image URL or base64 data

#### Tool Calls in Output Messages

- `llm.output_messages.<messageIndex>.message.tool_calls.<toolCallIndex>.tool_call.id` - Unique identifier for the tool call
- `llm.output_messages.<messageIndex>.message.tool_calls.<toolCallIndex>.tool_call.function.name` - Name of the function being called
- `llm.output_messages.<messageIndex>.message.tool_calls.<toolCallIndex>.tool_call.function.arguments` - JSON string of function arguments

#### Available Tools

- `llm.tools.<index>.tool.json_schema` - Complete JSON schema of the tool, including type, function definition, and parameters

#### Python

```python
messages = [{"message.role": "user", "message.content": "hello"},
            {"message.role": "assistant", "message.content": "hi"}]

for i, obj in enumerate(messages):
    for key, value in obj.items():
        span.set_attribute(f"llm.input_messages.{i}.{key}", value)
```

#### JavaScript/TypeScript (ES6)

```javascript
const messages = [
  { "message.role": "user", "message.content": "hello" },
  {
    "message.role": "assistant",
    "message.content": "hi",
  },
];

for (const [i, obj] of messages.entries()) {
  for (const [key, value] of Object.entries(obj)) {
    span.setAttribute(`llm.input_messages.${i}.${key}`, value);
  }
}
```

#### Go Example (Helper Functions)

```go
// Helper function for input message attributes
func InputMessageAttribute(index int, suffix string) string {
    return fmt.Sprintf("%s.%d.%s", LLMInputMessages, index, suffix)
}

// Helper function for multimodal content attributes
func InputMessageContentAttribute(messageIndex, contentIndex int, suffix string) string {
    return fmt.Sprintf("%s.%d.message.contents.%d.message_content.%s",
        LLMInputMessages, messageIndex, contentIndex, suffix)
}

// Helper function for tool call attributes in output messages
func OutputMessageToolCallAttribute(messageIndex, toolCallIndex int, suffix string) string {
    return fmt.Sprintf("%s.%d.%s.%d.%s",
        LLMOutputMessages, messageIndex, MessageToolCalls, toolCallIndex, suffix)
}
```

If the objects are further nested, flattening should continue until the attribute values are either simple values,
i.e. `bool`, `str`, `bytes`, `int`, `float` or simple lists,
i.e. `List[bool]`, `List[str]`, `List[bytes]`, `List[int]`, `List[float]`.
