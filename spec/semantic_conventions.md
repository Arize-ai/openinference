# Semantic Conventions

The **Semantic Conventions** define the keys and values which describe commonly observed concepts, protocols, and
operations used by applications. These conventions are used to populate the `attributes` of `spans` and span `events`.

## Reserved Attributes

The following attributes are reserved and MUST be supported by all OpenInference Tracing SDKs:

| Attribute                              | Type            | Example                                                                    | Description                                                      |
| -------------------------------------- | --------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `openinference.span.kind`              | String          | `"CHAIN"`                                                                  | The kind of span (e.g., `CHAIN`, `LLM`, `RETRIEVER`, `RERANKER`) |
| `exception.type`                       | String          | `"NullPointerException"`                                                   | The type of exception that was thrown                            |
| `exception.message`                    | String          | `"Null value encountered"`                                                 | Detailed message describing the exception                        |
| `exception.escaped`                    | Boolean         | `true`                                                                     | Indicator if the exception has escaped the span's scope          |
| `exception.stacktrace`                 | String          | `"at app.main(app.java:16)"`                                               | The stack trace of the exception                                 |
| `output.value`                         | String          | `"Hello, World!"`                                                          | The output value of an operation                                 |
| `output.mime_type`                     | String          | `"text/plain"` or `"application/json"`                                     | MIME type representing the format of `output.value`              |
| `input.value`                          | String          | `"{'query': 'What is the weather today?'}"`                                | The input value to an operation                                  |
| `input.mime_type`                      | String          | `"text/plain"` or `"application/json"`                                     | MIME type representing the format of `input.value`               |
| `embedding.embeddings`                 | List of objects | `[{"embedding.vector": [...], "embedding.text": "hello"}]`                 | List of embedding objects including text and vector data         |
| `embedding.model_name`                 | String          | `"BERT-base"`                                                              | Name of the embedding model used                                 |
| `embedding.text`                       | String          | `"hello world"`                                                            | The text represented in the embedding                            |
| `embedding.vector`                     | List of floats  | `[0.123, 0.456, ...]`                                                      | The embedding vector consisting of a list of floats              |
| `llm.function_call`                    | JSON String     | `"{function_name: 'add', args: [1, 2]}"`                                   | Object recording details of a function call in models or APIs    |
| `llm.invocation_parameters`            | JSON string     | `"{model_name: 'gpt-3', temperature: 0.7}"`                                | Parameters used during the invocation of an LLM or API           |
| `llm.input_messages`                   | List of objects | `[{"message.role": "user", "message.content": "hello"}]`                   | List of messages sent to the LLM in a chat API request           |
| `llm.output_messages`                  | List of objects | `[{"message.role": "user", "message.content": "hello"}]`                   | List of messages received from the LLM in a chat API request     |
| `message.role`                         | String          | `"user"` or `"system"`                                                     | Role of the entity in a message (e.g., user, system)             |
| `message.function_call_name`           | String          | `"multiply"` or `"subtract"`                                               | Function call function name                                      |
| `message.function_call_arguments_json` | JSON String     | `"{ 'x': 2 }"`                                                             | The arguments to the function call in JSON                       |
| `message.content`                      | String          | `"What's the weather today?"`                                              | The content of a message in a chat                               |
| `message.tool_calls`                   | List of objects | `[{"tool_call.function.name": "get_current_weather"}]`                     | List of tool calls (e.g. function calls) generated by the LLM    |
| `tool_call.function.name`              | String          | `get_current_weather`                                                      | The name of the function being invoked by a tool call            |
| `tool_call.function.arguments`         | JSON string     | `"{'city': 'London'}"`                                                     | The arguments for the function being invoked by a tool call      |
| `llm.model_name`                       | String          | `"gpt-3.5-turbo"`                                                          | The name of the language model being utilized                    |
| `llm.prompt_template.template`         | String          | `"Weather forecast for {city} on {date}"`                                  | Template used to generate prompts as Python f-strings            |
| `llm.prompt_template.variables`        | JSON String     | `{ context: "<context from retrieval>", subject: "math" }`                 | JSON of key value pairs applied to the prompt template           |
| `llm.prompt_template.version`          | String          | `"v1.0"`                                                                   | The version of the prompt template                               |
| `llm.token_count.prompt`               | Integer         | `5`                                                                        | The number of tokens in the prompt                               |
| `llm.token_count.completion`           | Integer         | `15`                                                                       | The number of tokens in the completion                           |
| `llm.token_count.total`                | Integer         | `20`                                                                       | Total number of tokens, including prompt and completion          |
| `tool.name`                            | String          | `"WeatherAPI"`                                                             | The name of the tool being utilized                              |
| `tool.description`                     | String          | `"An API to get weather data."`                                            | Description of the tool's purpose and functionality              |
| `tool.parameters`                      | JSON string     | `"{ 'a': 'int' }"`                                                         | The parameters definition for invoking the tool                  |
| `retrieval.documents`                  | List of objects | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]` | List of retrieved documents                                      |
| `document.id`                          | String/Integer  | `"1234"` or `1`                                                            | Unique identifier for a document                                 |
| `document.score`                       | Float           | `0.98`                                                                     | Score representing the relevance of a document                   |
| `document.content`                     | String          | `"This is a sample document content."`                                     | The content of a retrieved document                              |
| `document.metadata`                    | JSON String     | `"{'author': 'John Doe', 'date': '2023-09-09'}"`                           | Metadata associated with a document                              |
| `reranker.input_documents`             | List of objects | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]` | List of documents as input to the reranker                       |
| `reranker.output_documents`            | List of objects | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]` | List of documents outputted by the reranker                      |
| `reranker.query`                       | String          | `"How to format timestamp?"`                                               | Query parameter of the reranker                                  |
| `reranker.model_name`                  | String          | `"cross-encoder/ms-marco-MiniLM-L-12-v2"`                                  | Model name of the reranker                                       |
| `reranker.top_k`                       | Integer         | 3                                                                          | Top K parameter of the reranker                                  |
| `tag.tags`                             | List of strings | ["shopping", "travel"]                                                     | List of tags to give the span a category                         |
| `metadata.*`                           | Any             | Any OTEL-compatible value                                                  | User-defined metadata for a chain or other span kind             |

Note: the `object` type refers to a set of key-value pairs also known as a `struct`, `mapping`, `dictionary`, etc.
