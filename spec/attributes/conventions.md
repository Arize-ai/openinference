# Attribute Conventions

These rules apply to every OpenInference attribute. The [semantic conventions](./semantic-conventions.md) list the
attributes themselves.

## Naming

Attribute names are dot-separated namespaces, with each segment in `snake_case`: `llm.token_count.prompt`,
`message_content.image`. The first segment names the domain the attribute belongs to (`llm`, `embedding`, `tool`,
`document`), and later segments narrow it.

OpenInference reuses OpenTelemetry's attributes where one exists for the same concept, such as the `exception.*`
attributes.

## Value Types

OTLP attribute values are strings, booleans, integers, floating point numbers, or homogeneous arrays of one of these.
The semantic conventions use these types:

| Type | Encoding |
| --- | --- |
| String, Boolean, Integer, Float | The OTLP primitive of the same name. |
| List of strings, List of floats | An OTLP array. |
| JSON String | A string holding serialized JSON, such as `metadata` or `llm.invocation_parameters`. Consumers parse it to recover the object. |
| List of objects | A list of structured items, [flattened](#flattening) into one attribute per leaf. |
| Image, Audio, Video, Document Object | A single structured item, flattened under its parent attribute, e.g. `message_content.image.image.url`. |

## Flattening

OTLP span attributes cannot hold objects, so structured values are flattened into one attribute per leaf. Lists of
objects use a zero-based integer index:

`<prefix>.<index>.<suffix>`

- `<prefix>` is the list attribute, such as `llm.input_messages` or `llm.tools`.
- `<index>` is the zero-based position of the item in the list.
- `<suffix>` is the attribute path within the item, which starts with the item's own namespace, such as
  `message.role` or `tool.name`.

For example, the first input message's role is `llm.input_messages.0.message.role`. Nested lists repeat the pattern:
`llm.output_messages.0.message.tool_calls.1.tool_call.function.name` is the function name of the second tool call in
the first output message.

Flattening continues until every value is a primitive (`bool`, `str`, `bytes`, `int`, `float`) or a list of
primitives.

Examples in this specification often show list attributes in a nested JSON form for readability, such as
`"llm.input_messages": [{"message.role": "user"}]`. On the wire they are always flattened.

### List Attributes

| List attribute | Item namespace | Flattened form |
| --- | --- | --- |
| `llm.input_messages`, `llm.output_messages` | `message.*` | `llm.input_messages.<i>.message.role` |
| `message.contents` | `message_content.*` | `llm.input_messages.<i>.message.contents.<j>.message_content.type` |
| `message.tool_calls` | `tool_call.*` | `llm.output_messages.<i>.message.tool_calls.<j>.tool_call.id` |
| `llm.tools` | `tool.*` | `llm.tools.<i>.tool.name` |
| `llm.prompts` | `prompt.*` | `llm.prompts.<i>.prompt.text` |
| `llm.choices` | `completion.*` | `llm.choices.<i>.completion.text` |
| `embedding.embeddings` | `embedding.*` | `embedding.embeddings.<i>.embedding.vector` |
| `retrieval.documents` | `document.*` | `retrieval.documents.<i>.document.content` |
| `reranker.input_documents`, `reranker.output_documents` | `document.*` | `reranker.output_documents.<i>.document.score` |
| `input.images`, `output.images` | `image.*` | `input.images.<i>.image.url` |
| `input.documents`, `output.documents` | `document.*` | `output.documents.<i>.document.url` |
| `annotations`, `trace.annotations`, `session.annotations` | `annotation.*` | `annotations.<i>.annotation.name` |
| `evaluations`, `trace.evaluations`, `session.evaluations` | `evaluation.*` | `evaluations.<i>.evaluation.score` |

The pages for each domain describe these structures in full: [Messages](../content/messages.md),
[Multimodal Content](../content/multimodal.md), [Tool Calling](../content/tool-calling.md), and
[Annotations and Evaluations](../feedback/annotations.md).

### Examples

Python:

```python
messages = [{"message.role": "user", "message.content": "hello"},
            {"message.role": "assistant", "message.content": "hi"}]

for i, obj in enumerate(messages):
    for key, value in obj.items():
        span.set_attribute(f"llm.input_messages.{i}.{key}", value)
```

JavaScript/TypeScript:

```javascript
const messages = [
  { "message.role": "user", "message.content": "hello" },
  { "message.role": "assistant", "message.content": "hi" },
];

for (const [i, obj] of messages.entries()) {
  for (const [key, value] of Object.entries(obj)) {
    span.setAttribute(`llm.input_messages.${i}.${key}`, value);
  }
}
```

Go helper functions:

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

## Requirement Levels

The span conventions give each attribute a requirement level, following the
[OpenTelemetry requirement levels](https://opentelemetry.io/docs/specs/semconv/general/attribute-requirement-level/):

| Level | Meaning |
| --- | --- |
| Required | The attribute MUST be set on every span the convention applies to. |
| Recommended | The attribute SHOULD be set when the information is available to the instrumentation. |
| Not used | The attribute SHOULD NOT be set on spans of this kind. The convention explains what to use instead. |

## Well-Known Values

Some attributes, such as `llm.system`, `llm.provider`, and `annotation.annotator_kind`, have a list of well-known
values in the [semantic conventions](./semantic-conventions.md). If one of the well-known values applies, it MUST be used; otherwise, a custom value MAY be
used.
