# Semantic Conventions

The **Semantic Conventions** define the keys and values which describe commonly observed concepts, protocols, and
operations used by applications. These conventions are used to populate the `attributes` of `spans` and span `events`.

## Reserved Attributes

The following attributes are reserved and MUST be supported by all OpenInference Tracing SDKs:

| Attribute                              | Type                        | Example                                                                    | Description                                                                           |
| -------------------------------------- | --------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `document.content`                     | String                      | `"This is a sample document content."`                                     | The content of a retrieved document                                                   |
| `document.id`                          | String/Integer              | `"1234"` or `1`                                                            | Unique identifier for a document                                                      |
| `document.metadata`                    | JSON String                 | `"{'author': 'John Doe', 'date': '2023-09-09'}"`                           | Metadata associated with a document                                                   |
| `document.score`                       | Float                       | `0.98`                                                                     | Score representing the relevance of a document                                        |
| `embedding.embeddings`                 | List of objects<sup>†</sup> | `[{"embedding.vector": [...], "embedding.text": "hello"}]`                 | List of embedding objects including text and vector data                              |
| `embedding.model_name`                 | String                      | `"BERT-base"`                                                              | Name of the embedding model used                                                      |
| `embedding.text`                       | String                      | `"hello world"`                                                            | The text represented in the embedding                                                 |
| `embedding.vector`                     | List of floats              | `[0.123, 0.456, ...]`                                                      | The embedding vector consisting of a list of floats                                   |
| `exception.escaped`                    | Boolean                     | `true`                                                                     | Indicator if the exception has escaped the span's scope                               |
| `exception.message`                    | String                      | `"Null value encountered"`                                                 | Detailed message describing the exception                                             |
| `exception.stacktrace`                 | String                      | `"at app.main(app.java:16)"`                                               | The stack trace of the exception                                                      |
| `exception.type`                       | String                      | `"NullPointerException"`                                                   | The type of exception that was thrown                                                 |
| `image.url`                            | String                      | `"https://sample-link-to-image.jpg"`                                       | The link to the image or its base64 encoding                                          |
| `input.mime_type`                      | String                      | `"text/plain"` or `"application/json"`                                     | MIME type representing the format of `input.value`                                    |
| `input.value`                          | String                      | `"{'query': 'What is the weather today?'}"`                                | The input value to an operation                                                       |
| `llm.function_call`                    | JSON String                 | `"{function_name: 'add', args: [1, 2]}"`                                   | Object recording details of a function call in models or APIs                         |
| `llm.input_messages`                   | List of objects<sup>†</sup> | `[{"message.role": "user", "message.content": "hello"}]`                   | List of messages sent to the LLM in a chat API request                                |
| `llm.invocation_parameters`            | JSON string                 | `"{model_name: 'gpt-3', temperature: 0.7}"`                                | Parameters used during the invocation of an LLM or API                                |
| `llm.provider`                         | String                      | `openai`, `azure`                                                          | The hosting provider of the llm, e.x. `azure`                                         |
| `llm.system`                           | String                      | `anthropic`, `openai`                                                      | The AI product as identified by the client or server instrumentation.                 |
| `llm.model_name`                       | String                      | `"gpt-3.5-turbo"`                                                          | The name of the language model being utilized                                         |
| `llm.output_messages`                  | List of objects<sup>†</sup> | `[{"message.role": "user", "message.content": "hello"}]`                   | List of messages received from the LLM in a chat API request                          |
| `llm.prompt_template.template`         | String                      | `"Weather forecast for {city} on {date}"`                                  | Template used to generate prompts as Python f-strings                                 |
| `llm.prompt_template.variables`        | JSON String                 | `{ context: "<context from retrieval>", subject: "math" }`                 | JSON of key value pairs applied to the prompt template                                |
| `llm.prompt_template.version`          | String                      | `"v1.0"`                                                                   | The version of the prompt template                                                    |
| `llm.token_count.completion`           | Integer                     | `15`                                                                       | The number of tokens in the completion                                                |
| `llm.token_count.prompt`               | Integer                     | `5`                                                                        | The number of tokens in the prompt                                                    |
| `llm.token_count.total`                | Integer                     | `20`                                                                       | Total number of tokens, including prompt and completion                               |
| `llm.tools`                            | List of objects<sup>†</sup> | `[{"tool": {"json_schema": "{}"}, ...]`                                    | List of tools that are advertised to the LLM to be able to call                       |
| `message.content`                      | String                      | `"What's the weather today?"`                                              | The content of a message in a chat                                                    |
| `message.contents`                     | List of objects<sup>†</sup> | `[{"message_content.type": "text", "message_content.text": "Hello"}, ...]` | The message contents to the llm, it is an array of `message_content` objects.         |
| `message.function_call_arguments_json` | JSON String                 | `"{ 'x': 2 }"`                                                             | The arguments to the function call in JSON                                            |
| `message.function_call_name`           | String                      | `"multiply"` or `"subtract"`                                               | Function call function name                                                           |
| `message.tool_call_id`                 | String                      | `"call_62136355"`                                                          | Tool call result identifier corresponding to `tool_call.id`                           |
| `message.role`                         | String                      | `"user"` or `"system"`                                                     | Role of the entity in a message (e.g., user, system)                                  |
| `message.tool_calls`                   | List of objects<sup>†</sup> | `[{"tool_call.function.name": "get_current_weather"}]`                     | List of tool calls (e.g. function calls) generated by the LLM                         |
| `messagecontent.type`                  | String                      | `"text"` or `"image"`                                                      | The type of the content, such as "text" or "image".                                   |
| `messagecontent.text`                  | String                      | `"This is a sample text"`                                                  | The text content of the message, if the type is "text".                               |
| `messagecontent.image`                 | Image Object                | `{"image.url": "https://sample-link-to-image.jpg"}`                        | The image content of the message, if the type is "image".                             |
| `metadata`                             | JSON String                 | `"{'author': 'John Doe', 'date': '2023-09-09'}"`                           | Metadata associated with a span                                                       |
| `openinference.span.kind`              | String                      | `"CHAIN"`                                                                  | The kind of span (e.g., `CHAIN`, `LLM`, `RETRIEVER`, `RERANKER`)                      |
| `output.mime_type`                     | String                      | `"text/plain"` or `"application/json"`                                     | MIME type representing the format of `output.value`                                   |
| `output.value`                         | String                      | `"Hello, World!"`                                                          | The output value of an operation                                                      |
| `reranker.input_documents`             | List of objects<sup>†</sup> | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]` | List of documents as input to the reranker                                            |
| `reranker.model_name`                  | String                      | `"cross-encoder/ms-marco-MiniLM-L-12-v2"`                                  | Model name of the reranker                                                            |
| `reranker.output_documents`            | List of objects<sup>†</sup> | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]` | List of documents outputted by the reranker                                           |
| `reranker.query`                       | String                      | `"How to format timestamp?"`                                               | Query parameter of the reranker                                                       |
| `reranker.top_k`                       | Integer                     | 3                                                                          | Top K parameter of the reranker                                                       |
| `retrieval.documents`                  | List of objects<sup>†</sup> | `[{"document.id": "1", "document.score": 0.9, "document.content": "..."}]` | List of retrieved documents                                                           |
| `session.id`                           | String                      | `"26bcd3d2-cad2-443d-a23c-625e47f3324a"`                                   | Unique identifier for a session                                                       |
| `tag.tags`                             | List of strings             | ["shopping", "travel"]                                                     | List of tags to give the span a category                                              |
| `tool.description`                     | String                      | `"An API to get weather data."`                                            | Description of the tool's purpose and functionality                                   |
| `tool.json_schema`                     | JSON String                 | `"{'type': 'function', 'function': {'name': 'get_weather'}}"`              | The json schema of a tool input                                                       |
| `tool.name`                            | String                      | `"WeatherAPI"`                                                             | The name of the tool being utilized                                                   |
| `tool.id`                              | String                      | `"WeatherAPI"`                                                             | The identifier for the result of the tool call (corresponding to `tool_call.id`)      |
| `tool.parameters`                      | JSON string                 | `"{ 'a': 'int' }"`                                                         | The parameters definition for invoking the tool                                       |
| `tool_call.function.arguments`         | JSON string                 | `"{'city': 'London'}"`                                                     | The arguments for the function being invoked by a tool call                           |
| `tool_call.function.name`              | String                      | `"get_current_weather"`                                                    | The name of the function being invoked by a tool call                                 |
| `tool_call.id`                         | string                      | `"call_62136355"`                                                          | The id of the a tool call (useful when there are more than one call at the same time) |
| `user.id`                              | String                      | `"9328ae73-7141-4f45-a044-8e06192aa465"`                                   | Unique identifier for a user                                                          |
| `audio.url`                            | String                      | `https://storage.com/buckets/1/file.wav`                                   | The url to an audio file (e.x. cloud storage)                                         |
| `audio.mime_type`                      | String                      | `audio/mpeg`                                                               | The mime type of the audio file (e.x. `audio/mpeg`, `audio/wav` )                     |
| `audio.transcript`                     | String                      | `"Hello, how are you?"`                                                    | The transcript of the audio file (e.x. whisper transcription)                         |

<sup>†</sup> To get a list of objects exported as OpenTelemetry span attributes, flattening of the list is necessary as
shown in the examples below.

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

#### Python

```python
messages = [{"message.role": "user", "message.content": "hello"},
            {"message.role": "assistant", "message.content": "hi"}]

for i, obj in enumerate(messages):
    for key, value in obj.items():
        span.set_attribute(f"input.messages.{i}.{key}", value)
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
        span.setAttribute(`input.messages.${i}.${key}`, value);
    }
}
```

If the objects are further nested, flattening should continue until the attribute values are either simple values,
i.e. `bool`, `str`, `bytes`, `int`, `float` or simple lists,
i.e. `List[bool]`, `List[str]`, `List[bytes]`, `List[int]`, `List[float]`.
