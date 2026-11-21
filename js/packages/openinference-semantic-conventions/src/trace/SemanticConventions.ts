/**
 * Semantic conventions for OpenInference tracing
 *
 * Generated from the semantic convention registry in semantic-conventions/model.
 * Do not edit by hand: change the model or
 * semantic-conventions/templates/registry/typescript, then run
 * `make -C semantic-conventions generate`.
 */

export const SemanticAttributePrefixes = {
  agent: "agent",
  annotation: "annotation",
  annotations: "annotations",
  audio: "audio",
  completion: "completion",
  decision: "decision",
  document: "document",
  embedding: "embedding",
  evaluation: "evaluation",
  evaluations: "evaluations",
  graph: "graph",
  image: "image",
  input: "input",
  llm: "llm",
  message: "message",
  message_content: "message_content",
  messages: "messages",
  metadata: "metadata",
  openinference: "openinference",
  output: "output",
  prompt: "prompt",
  reranker: "reranker",
  retrieval: "retrieval",
  session: "session",
  tag: "tag",
  tool: "tool",
  tool_call: "tool_call",
  trace: "trace",
  user: "user",
  video: "video",
} as const;

export const LLMAttributePostfixes = {
  choices: "choices",
  cost: "cost",
  finish_reason: "finish_reason",
  function_call: "function_call",
  input_messages: "input_messages",
  invocation_parameters: "invocation_parameters",
  model_name: "model_name",
  output_messages: "output_messages",
  prompt_template: "prompt_template",
  prompts: "prompts",
  provider: "provider",
  request: "request",
  response: "response",
  system: "system",
  token_count: "token_count",
  tools: "tools",
} as const;

export const DecisionAttributePostfixes = {
  model_name: "model_name",
  provider: "provider",
  request: "request",
  response: "response",
  system: "system",
  token_count: "token_count",
} as const;

export const LLMPromptTemplateAttributePostfixes = {
  template: "template",
  variables: "variables",
  version: "version",
} as const;

export const RetrievalAttributePostfixes = {
  documents: "documents",
} as const;

export const RerankerAttributePostfixes = {
  input_documents: "input_documents",
  model_name: "model_name",
  output_documents: "output_documents",
  query: "query",
  top_k: "top_k",
} as const;

export const EmbeddingAttributePostfixes = {
  embeddings: "embeddings",
  invocation_parameters: "invocation_parameters",
  model_name: "model_name",
  text: "text",
  vector: "vector",
} as const;

export const ToolAttributePostfixes = {
  description: "description",
  id: "id",
  json_schema: "json_schema",
  name: "name",
  parameters: "parameters",
} as const;

export const MessageAttributePostfixes = {
  content: "content",
  contents: "contents",
  function_call_arguments_json: "function_call_arguments_json",
  function_call_name: "function_call_name",
  name: "name",
  role: "role",
  tool_call_id: "tool_call_id",
  tool_calls: "tool_calls",
} as const;

export const MessageContentsAttributePostfixes = {
  audio: "audio",
  data: "data",
  document: "document",
  encrypted_content: "encrypted_content",
  id: "id",
  image: "image",
  signature: "signature",
  text: "text",
  type: "type",
  video: "video",
} as const;

export const ImageAttributesPostfixes = {
  url: "url",
} as const;

export const VideoAttributesPostfixes = {
  url: "url",
} as const;

export const ToolCallAttributePostfixes = {
  function_arguments_json: "function.arguments",
  function_name: "function.name",
  id: "id",
  reasoning_signature: "reasoning_signature",
} as const;

export const DocumentAttributePostfixes = {
  content: "content",
  id: "id",
  metadata: "metadata",
  mime_type: "mime_type",
  name: "name",
  score: "score",
  url: "url",
} as const;

export const TagAttributePostfixes = {
  tags: "tags",
} as const;

export const SessionAttributePostfixes = {
  annotations: "annotations",
  evaluations: "evaluations",
  id: "id",
} as const;

export const UserAttributePostfixes = {
  id: "id",
} as const;

export const AudioAttributesPostfixes = {
  mime_type: "mime_type",
  transcript: "transcript",
  url: "url",
} as const;

export const PromptAttributePostfixes = {
  id: "id",
  text: "text",
  url: "url",
  vendor: "vendor",
} as const;

export const AgentPostfixes = {
  name: "name",
} as const;

export const GraphPostfixes = {
  node_id: "node.id",
  node_name: "node.name",
  node_parent_id: "node.parent_id",
} as const;

export const FeedbackAttributePostfixes = {
  annotator_kind: "annotator_kind",
  explanation: "explanation",
  identifier: "identifier",
  label: "label",
  metadata: "metadata",
  name: "name",
  score: "score",
} as const;

/**
 * The name of the agent that this span represents.
 */
export const AGENT_NAME = "agent.name";

/**
 * Judge kind: `HUMAN`, `LLM`, `CODE`, or custom. See [Annotator Kinds](https://github.com/Arize-ai/openinference/blob/main/spec/feedback/annotations.md#annotator-kinds)
 */
export const ANNOTATION_ANNOTATOR_KIND = "annotation.annotator_kind";

/**
 * Reason or evidence for the result
 */
export const ANNOTATION_EXPLANATION = "annotation.explanation";

/**
 * Stable result ID for the same name and target
 */
export const ANNOTATION_IDENTIFIER = "annotation.identifier";

/**
 * Categorical result
 */
export const ANNOTATION_LABEL = "annotation.label";

/**
 * Additional result or judge data
 */
export const ANNOTATION_METADATA = "annotation.metadata";

/**
 * Criterion or metric name
 */
export const ANNOTATION_NAME = "annotation.name";

/**
 * Numeric result.
 */
export const ANNOTATION_SCORE = "annotation.score";

/**
 * Span-scoped feedback using annotation terminology, flattened as `annotations.0.annotation.*`
 */
export const ANNOTATIONS = "annotations";

/**
 * The MIME type of the audio file.
 */
export const AUDIO_MIME_TYPE = "audio.mime_type";

/**
 * The transcript of the audio file (e.g. a Whisper transcription)
 */
export const AUDIO_TRANSCRIPT = "audio.transcript";

/**
 * The URL of an audio file (e.g. cloud storage), or a base64 data URI
 */
export const AUDIO_URL = "audio.url";

/**
 * The text of a choice returned from a completions API. Used under `llm.choices`.
 */
export const COMPLETION_TEXT = "completion.text";

/**
 * The name of the decision model being utilized. Mirrors `llm.model_name` for `DECISION` spans.
 */
export const DECISION_MODEL_NAME = "decision.model_name";

/**
 * The hosting provider of the decision model: who runs the hardware that answered, e.g. `typesafe` when calling TypeSafe directly, `openai` for the OpenAI Decisions API, or the cloud or self-hosting provider for a model served elsewhere. Distinct from `decision.system`, which names the API shape. The well-known values alias the matching `llm.provider` values.
 */
export const DECISION_PROVIDER = "decision.provider";

/**
 * The decision model requested by the caller, as sent in the request. May differ from `decision.response.model_name` when the provider resolves an alias or routes the request.
 */
export const DECISION_REQUEST_MODEL_NAME = "decision.request.model_name";

/**
 * The decision model that actually produced the response, as reported by the provider. May differ from `decision.request.model_name`.
 */
export const DECISION_RESPONSE_MODEL_NAME = "decision.response.model_name";

/**
 * The decision API ecosystem the call conforms to, i.e. which request/response shape the client speaks, as identified by the client or server instrumentation: `typesafe` for the TypeSafe System One / Jev API, `openai` for the OpenAI Decisions API. A self-hosted vLLM server answering the Jev-compatible `/v1/systemone` shape is still `typesafe`. Distinct from `decision.provider`, which says who hosts the model. The well-known values alias the matching `llm.system` values.
 */
export const DECISION_SYSTEM = "decision.system";

/**
 * The number of input tokens consumed by a decision model call: the state, questions, and candidate options. Maps to `usage.input_tokens` in TypeSafe System One responses.
 */
export const DECISION_TOKEN_COUNT_INPUT = "decision.token_count.input";

/**
 * The number of output tokens produced by a decision model call. Decision models emit typed answers rather than text, so this is typically small. Maps to `usage.output_tokens` in TypeSafe System One responses.
 */
export const DECISION_TOKEN_COUNT_OUTPUT = "decision.token_count.output";

/**
 * The text content of a document: a retrieved passage, or an inline text document passed to a model (e.g. an Anthropic `text` source)
 */
export const DOCUMENT_CONTENT = "document.content";

/**
 * Unique identifier for a document: a retrieved document or chunk in a retrieval system, or the provider file id (e.g. an OpenAI or Anthropic Files API id) for a document passed to a model.
 */
export const DOCUMENT_ID = "document.id";

/**
 * Metadata associated with a document
 */
export const DOCUMENT_METADATA = "document.metadata";

/**
 * The MIME type of a file document such as a PDF
 */
export const DOCUMENT_MIME_TYPE = "document.mime_type";

/**
 * The name supplied with a document, copied verbatim: a file name (OpenAI `filename`) or a title (Anthropic `title`, Bedrock `name`, Gemini `display_name`); not a unique identifier
 */
export const DOCUMENT_NAME = "document.name";

/**
 * Score representing the relevance of a document
 */
export const DOCUMENT_SCORE = "document.score";

/**
 * The URL, object-store URI, or base64 data URI of a file document such as a PDF
 */
export const DOCUMENT_URL = "document.url";

/**
 * List of embedding objects including text and vector data
 */
export const EMBEDDING_EMBEDDINGS = "embedding.embeddings";

/**
 * Parameters used during the invocation of an embedding model or API (excluding input)
 */
export const EMBEDDING_INVOCATION_PARAMETERS = "embedding.invocation_parameters";

/**
 * Name of the embedding model used
 */
export const EMBEDDING_MODEL_NAME = "embedding.model_name";

/**
 * The text represented in the embedding
 */
export const EMBEDDING_TEXT = "embedding.text";

/**
 * The embedding vector consisting of a list of floats
 */
export const EMBEDDING_VECTOR = "embedding.vector";

/**
 * Judge kind: `HUMAN`, `LLM`, `CODE`, or custom. See [Annotator Kinds](https://github.com/Arize-ai/openinference/blob/main/spec/feedback/annotations.md#annotator-kinds)
 */
export const EVALUATION_ANNOTATOR_KIND = "evaluation.annotator_kind";

/**
 * Reason or evidence for the result
 */
export const EVALUATION_EXPLANATION = "evaluation.explanation";

/**
 * Stable result ID for the same name and target
 */
export const EVALUATION_IDENTIFIER = "evaluation.identifier";

/**
 * Categorical result
 */
export const EVALUATION_LABEL = "evaluation.label";

/**
 * Additional result or judge data
 */
export const EVALUATION_METADATA = "evaluation.metadata";

/**
 * Criterion or metric name
 */
export const EVALUATION_NAME = "evaluation.name";

/**
 * Numeric result.
 */
export const EVALUATION_SCORE = "evaluation.score";

/**
 * Span-scoped feedback using evaluation or eval terminology, flattened as `evaluations.0.evaluation.*`
 */
export const EVALUATIONS = "evaluations";

/**
 * The id of the node in the execution graph. This along with `graph.node.parent_id` are used to visualize the execution graph.
 */
export const GRAPH_NODE_ID = "graph.node.id";

/**
 * The name of the node in the execution graph. Use this to present a human readable name for the node. Optional
 */
export const GRAPH_NODE_NAME = "graph.node.name";

/**
 * This references the id of the parent node. Leaving this unset or set as empty string implies that the current span is the root node.
 */
export const GRAPH_NODE_PARENT_ID = "graph.node.parent_id";

/**
 * The link to the image or its base64 encoding
 */
export const IMAGE_URL = "image.url";

/**
 * File document (e.g. PDF) inputs to an operation, independent of span kind, e.g. `input.documents.0.document.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-documents)
 */
export const INPUT_DOCUMENTS = "input.documents";

/**
 * Image inputs to an operation, independent of span kind, e.g. `input.images.0.image.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-images)
 */
export const INPUT_IMAGES = "input.images";

/**
 * MIME type representing the format of `input.value`
 */
export const INPUT_MIME_TYPE = "input.mime_type";

/**
 * The input value to an operation
 */
export const INPUT_VALUE = "input.value";

/**
 * Text choices returned from a completions API, e.g. `llm.choices.0.completion.text`
 */
export const LLM_CHOICES = "llm.choices";

/**
 * Total cost of all output tokens generated by the LLM in USD
 */
export const LLM_COST_COMPLETION = "llm.cost.completion";

/**
 * Cost of audio tokens in the completion in USD
 */
export const LLM_COST_COMPLETION_DETAILS_AUDIO = "llm.cost.completion_details.audio";

/**
 * Total cost of output tokens in USD
 */
export const LLM_COST_OUTPUT = "llm.cost.completion_details.output";

/**
 * Cost of reasoning steps in the completion in USD
 */
export const LLM_COST_COMPLETION_DETAILS_REASONING = "llm.cost.completion_details.reasoning";

/**
 * Total cost of all input tokens sent to the LLM in USD
 */
export const LLM_COST_PROMPT = "llm.cost.prompt";

/**
 * Cost of audio tokens in the prompt in USD
 */
export const LLM_COST_PROMPT_DETAILS_AUDIO = "llm.cost.prompt_details.audio";

/**
 * Cost of input tokens in the prompt that were cached in USD
 */
export const LLM_COST_PROMPT_DETAILS_CACHE_INPUT = "llm.cost.prompt_details.cache_input";

/**
 * Cost of prompt tokens read from cache in USD
 */
export const LLM_COST_PROMPT_DETAILS_CACHE_READ = "llm.cost.prompt_details.cache_read";

/**
 * Cost of prompt tokens written to cache in USD
 */
export const LLM_COST_PROMPT_DETAILS_CACHE_WRITE = "llm.cost.prompt_details.cache_write";

/**
 * Total cost of input tokens in USD
 */
export const LLM_COST_INPUT = "llm.cost.prompt_details.input";

/**
 * Total cost of the LLM call in USD (prompt + completion)
 */
export const LLM_COST_TOTAL = "llm.cost.total";

/**
 * The reason the model stopped generating tokens.
 */
export const LLM_FINISH_REASON = "llm.finish_reason";

/**
 * Object recording details of a function call in models or APIs
 * @deprecated Use `message.tool_calls` instead.
 */
export const LLM_FUNCTION_CALL = "llm.function_call";

/**
 * List of messages sent to the LLM in a chat API request, e.g. `llm.input_messages.0.message.role`. See [Messages](https://github.com/Arize-ai/openinference/blob/main/spec/content/messages.md)
 */
export const LLM_INPUT_MESSAGES = "llm.input_messages";

/**
 * Parameters used during the invocation of an LLM or API
 */
export const LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters";

/**
 * The name of the language model being utilized
 */
export const LLM_MODEL_NAME = "llm.model_name";

/**
 * List of messages received from the LLM in a chat API response, e.g. `llm.output_messages.0.message.role`. See [Messages](https://github.com/Arize-ai/openinference/blob/main/spec/content/messages.md)
 */
export const LLM_OUTPUT_MESSAGES = "llm.output_messages";

/**
 * Template used to generate prompts (e.g., using `{variable}` placeholder syntax)
 */
export const PROMPT_TEMPLATE_TEMPLATE = "llm.prompt_template.template";

/**
 * JSON of key value pairs applied to the prompt template
 */
export const PROMPT_TEMPLATE_VARIABLES = "llm.prompt_template.variables";

/**
 * The version of the prompt template
 */
export const PROMPT_TEMPLATE_VERSION = "llm.prompt_template.version";

/**
 * Prompts provided to a completions API, e.g. `llm.prompts.0.prompt.text`
 */
export const LLM_PROMPTS = "llm.prompts";

/**
 * The hosting provider of the LLM, e.g. `azure`. See [well-known values](https://github.com/Arize-ai/openinference/blob/main/spec/attributes/semantic-conventions.md#llm-provider-values).
 */
export const LLM_PROVIDER = "llm.provider";

/**
 * The model requested by the caller, as sent in the request. May differ from `llm.response.model_name` when the provider routes the request to a different model.
 */
export const LLM_REQUEST_MODEL_NAME = "llm.request.model_name";

/**
 * The model that actually generated the response, as reported by the provider. May differ from `llm.request.model_name` when the provider routes the request to a different model.
 */
export const LLM_RESPONSE_MODEL_NAME = "llm.response.model_name";

/**
 * The AI product as identified by the client or server instrumentation. See [well-known values](https://github.com/Arize-ai/openinference/blob/main/spec/attributes/semantic-conventions.md#llm-system-values).
 */
export const LLM_SYSTEM = "llm.system";

/**
 * The number of tokens in the completion
 */
export const LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion";

/**
 * The number of audio tokens generated by the model
 */
export const LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO = "llm.token_count.completion_details.audio";

/**
 * The number of tokens used for model reasoning
 */
export const LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING =
  "llm.token_count.completion_details.reasoning";

/**
 * The number of tokens in the prompt
 */
export const LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt";

/**
 * The number of audio input tokens presented in the prompt
 */
export const LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO = "llm.token_count.prompt_details.audio";

/**
 * The number of input tokens in the prompt that were cached
 */
export const LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_INPUT =
  "llm.token_count.prompt_details.cache_input";

/**
 * The number of prompt tokens successfully retrieved from cache (cache hits). Maps to `cached_tokens` in OpenAI responses
 */
export const LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ =
  "llm.token_count.prompt_details.cache_read";

/**
 * The number of prompt tokens not found in cache that were written to cache (cache misses). Maps to `cache_write_tokens` in OpenAI responses
 */
export const LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE =
  "llm.token_count.prompt_details.cache_write";

/**
 * Total number of tokens, including prompt and completion
 */
export const LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total";

/**
 * List of tools that are advertised to the LLM to be able to call, e.g. `llm.tools.0.tool.name`, `llm.tools.0.tool.description`, `llm.tools.0.tool.json_schema`. See [Tool Calling](https://github.com/Arize-ai/openinference/blob/main/spec/content/tool-calling.md)
 */
export const LLM_TOOLS = "llm.tools";

/**
 * The content of a message in a chat
 */
export const MESSAGE_CONTENT = "message.content";

/**
 * The message contents to the LLM, an array of `message_content` objects. Use this form when a message has multiple content items or when item ordering matters.
 */
export const MESSAGE_CONTENTS = "message.contents";

/**
 * The arguments to the function call in JSON
 * @deprecated Use `message.tool_calls` instead.
 */
export const MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = "message.function_call_arguments_json";

/**
 * Function call function name
 * @deprecated Use `message.tool_calls` instead.
 */
export const MESSAGE_FUNCTION_CALL_NAME = "message.function_call_name";

/**
 * The name of the function or tool that produced a tool/function role message. Used when `message.role` is `"tool"` or `"function"` to identify the origin.
 */
export const MESSAGE_NAME = "message.name";

/**
 * Role of the entity in a message (e.g., user, system)
 */
export const MESSAGE_ROLE = "message.role";

/**
 * Tool call result identifier corresponding to `tool_call.id`
 */
export const MESSAGE_TOOL_CALL_ID = "message.tool_call_id";

/**
 * List of tool calls (e.g. function calls) generated by the LLM, e.g. `llm.output_messages.0.message.tool_calls.0.tool_call.id`
 */
export const MESSAGE_TOOL_CALLS = "message.tool_calls";

/**
 * The audio content of the message, if the type is `"audio"`. Nested leaves are `audio.url` and `audio.transcript`.
 */
export const MESSAGE_CONTENT_AUDIO = "message_content.audio";

/**
 * Opaque vendor-issued data captured verbatim. Maps to Anthropic `redacted_thinking.data`.
 */
export const MESSAGE_CONTENT_DATA = "message_content.data";

/**
 * The document content of the message (e.g. a PDF, an uploaded file, or inline text), if the type is `"document"`. Nested leaves are `document.url`, `document.content`, `document.mime_type`, `document.id`, and `document.name`.
 */
export const MESSAGE_CONTENT_DOCUMENT = "message_content.document";

/**
 * OpenAI `encrypted_content` captured verbatim.
 */
export const MESSAGE_CONTENT_ENCRYPTED_CONTENT = "message_content.encrypted_content";

/**
 * Provider-assigned identifier for this message content item. For OpenAI Responses reasoning items, this maps to `ResponseReasoningItem.id` and should be preserved for stateless replay.
 */
export const MESSAGE_CONTENT_ID = "message_content.id";

/**
 * The image content of the message, if the type is `"image"`.
 */
export const MESSAGE_CONTENT_IMAGE = "message_content.image";

/**
 * Opaque vendor-issued signature captured verbatim. Maps to provider `signature` fields, and to Gemini `thoughtSignature` fields when the signature is attached to a non-tool content part such as text.
 */
export const MESSAGE_CONTENT_SIGNATURE = "message_content.signature";

/**
 * The text content of the message. Used when `message_content.type` is `"text"` or `"reasoning"` (visible reasoning or summary text). For OpenAI `summary_text` arrays, concatenate the array items into this single text value for now.
 */
export const MESSAGE_CONTENT_TEXT = "message_content.text";

/**
 * The type of the content. Use `"reasoning"` for reasoning/thinking content, including Anthropic `redacted_thinking` blocks. Use `"tool_use"` when a provider returns a tool call as an ordered content part.
 */
export const MESSAGE_CONTENT_TYPE = "message_content.type";

/**
 * The video content of the message, if the type is `"video"`. The nested leaf is `video.url`. Consumers infer MIME type from the URL path extension or a data URI prefix.
 */
export const MESSAGE_CONTENT_VIDEO = "message_content.video";

/**
 * Metadata associated with a span
 */
export const METADATA = "metadata";

/**
 * File document (e.g. PDF) outputs of an operation, independent of span kind, e.g. `output.documents.0.document.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-documents)
 */
export const OUTPUT_DOCUMENTS = "output.documents";

/**
 * Image outputs of an operation, independent of span kind, e.g. `output.images.0.image.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-images)
 */
export const OUTPUT_IMAGES = "output.images";

/**
 * MIME type representing the format of `output.value`
 */
export const OUTPUT_MIME_TYPE = "output.mime_type";

/**
 * The output value of an operation
 */
export const OUTPUT_VALUE = "output.value";

/**
 * A vendor-specific id used to identify the prompt.
 */
export const PROMPT_ID = "prompt.id";

/**
 * The text of a prompt in a completions API request. Used under `llm.prompts`.
 */
export const PROMPT_TEXT = "prompt.text";

/**
 * A vendor-specific URL used to locate the prompt via the web
 */
export const PROMPT_URL = "prompt.url";

/**
 * The vendor or origin of the prompt, e.g. `langsmith`, `portkey`, `arize-phoenix`
 */
export const PROMPT_VENDOR = "prompt.vendor";

/**
 * List of documents as input to the reranker
 */
export const RERANKER_INPUT_DOCUMENTS = "reranker.input_documents";

/**
 * Model name of the reranker
 */
export const RERANKER_MODEL_NAME = "reranker.model_name";

/**
 * List of documents outputted by the reranker
 */
export const RERANKER_OUTPUT_DOCUMENTS = "reranker.output_documents";

/**
 * Query parameter of the reranker
 */
export const RERANKER_QUERY = "reranker.query";

/**
 * Top K parameter of the reranker
 */
export const RERANKER_TOP_K = "reranker.top_k";

/**
 * List of retrieved documents
 */
export const RETRIEVAL_DOCUMENTS = "retrieval.documents";

/**
 * Session-scoped feedback using annotation terminology, flattened as `session.annotations.0.annotation.*`; requires `session.id`
 */
export const SESSION_ANNOTATIONS = "session.annotations";

/**
 * Session-scoped feedback using evaluation terminology, flattened as `session.evaluations.0.evaluation.*`; requires `session.id`
 */
export const SESSION_EVALUATIONS = "session.evaluations";

/**
 * Unique identifier for a session, such as a multi-turn conversation.
 */
export const SESSION_ID = "session.id";

/**
 * List of tags to give the span a category
 */
export const TAG_TAGS = "tag.tags";

/**
 * Description of the tool's purpose and functionality. On a `TOOL` span, the tool being invoked; under `llm.tools.<index>`, an advertised tool definition
 */
export const TOOL_DESCRIPTION = "tool.description";

/**
 * The identifier for the result of the tool call (corresponding to `tool_call.id`)
 */
export const TOOL_ID = "tool.id";

/**
 * The JSON schema of a tool input
 */
export const TOOL_JSON_SCHEMA = "tool.json_schema";

/**
 * The name of the tool. On a `TOOL` span, the tool being invoked; under `llm.tools.<index>`, the name of an advertised tool definition
 */
export const TOOL_NAME = "tool.name";

/**
 * The parameters definition for invoking the tool
 */
export const TOOL_PARAMETERS = "tool.parameters";

/**
 * The arguments for the function being invoked by a tool call
 */
export const TOOL_CALL_FUNCTION_ARGUMENTS_JSON = "tool_call.function.arguments";

/**
 * The name of the function being invoked by a tool call
 */
export const TOOL_CALL_FUNCTION_NAME = "tool_call.function.name";

/**
 * The id of a tool call (useful when there is more than one call at the same time)
 */
export const TOOL_CALL_ID = "tool_call.id";

/**
 * Opaque vendor-issued reasoning echo token attached to a tool call. Used by Gemini, which puts `thoughtSignature` on `functionCall` parts rather than on the reasoning summary.
 */
export const TOOL_CALL_REASONING_SIGNATURE = "tool_call.reasoning_signature";

/**
 * Trace-scoped feedback using annotation terminology, flattened as `trace.annotations.0.annotation.*`
 */
export const TRACE_ANNOTATIONS = "trace.annotations";

/**
 * Trace-scoped feedback using evaluation terminology, flattened as `trace.evaluations.0.evaluation.*`
 */
export const TRACE_EVALUATIONS = "trace.evaluations";

/**
 * Unique identifier for a user.
 */
export const USER_ID = "user.id";

/**
 * The URL, object-store URI, or base64 data URI of a video. Consumers infer MIME type from the path extension (`.mp4` to `video/mp4`, `.webm` to `video/webm`, `.ogv` to `video/ogg`, `.avi` to `video/x-msvideo`) or from a data URI prefix.
 */
export const VIDEO_URL = "video.url";

/**
 * Key prefix for additional prompt token count details. Each detail is a separate attribute with this prefix, e.g. llm.token_count.prompt_details.audio. All values are in tokens.
 */
export const LLM_TOKEN_COUNT_PROMPT_DETAILS = "llm.token_count.prompt_details";

/**
 * Key prefix for additional completion token count details. Each detail is a separate attribute with this prefix, e.g. llm.token_count.completion_details.reasoning. All values are in tokens.
 */
export const LLM_TOKEN_COUNT_COMPLETION_DETAILS = "llm.token_count.completion_details";

/**
 * Key prefix for cost information. All values are in USD.
 */
export const LLM_COST = "llm.cost";

export const SemanticConventions = {
  AGENT_NAME,
  ANNOTATION_ANNOTATOR_KIND,
  ANNOTATION_EXPLANATION,
  ANNOTATION_IDENTIFIER,
  ANNOTATION_LABEL,
  ANNOTATION_METADATA,
  ANNOTATION_NAME,
  ANNOTATION_SCORE,
  ANNOTATIONS,
  AUDIO_MIME_TYPE,
  AUDIO_TRANSCRIPT,
  AUDIO_URL,
  COMPLETION_TEXT,
  DECISION_MODEL_NAME,
  DECISION_PROVIDER,
  DECISION_REQUEST_MODEL_NAME,
  DECISION_RESPONSE_MODEL_NAME,
  DECISION_SYSTEM,
  DECISION_TOKEN_COUNT_INPUT,
  DECISION_TOKEN_COUNT_OUTPUT,
  DOCUMENT_CONTENT,
  DOCUMENT_ID,
  DOCUMENT_METADATA,
  DOCUMENT_MIME_TYPE,
  DOCUMENT_NAME,
  DOCUMENT_SCORE,
  DOCUMENT_URL,
  EMBEDDING_EMBEDDINGS,
  EMBEDDING_INVOCATION_PARAMETERS,
  EMBEDDING_MODEL_NAME,
  EMBEDDING_TEXT,
  EMBEDDING_VECTOR,
  EVALUATION_ANNOTATOR_KIND,
  EVALUATION_EXPLANATION,
  EVALUATION_IDENTIFIER,
  EVALUATION_LABEL,
  EVALUATION_METADATA,
  EVALUATION_NAME,
  EVALUATION_SCORE,
  EVALUATIONS,
  GRAPH_NODE_ID,
  GRAPH_NODE_NAME,
  GRAPH_NODE_PARENT_ID,
  IMAGE_URL,
  INPUT_DOCUMENTS,
  INPUT_IMAGES,
  INPUT_MIME_TYPE,
  INPUT_VALUE,
  LLM_CHOICES,
  LLM_COST_COMPLETION,
  LLM_COST_COMPLETION_DETAILS_AUDIO,
  LLM_COST_OUTPUT,
  LLM_COST_COMPLETION_DETAILS_REASONING,
  LLM_COST_PROMPT,
  LLM_COST_PROMPT_DETAILS_AUDIO,
  LLM_COST_PROMPT_DETAILS_CACHE_INPUT,
  LLM_COST_PROMPT_DETAILS_CACHE_READ,
  LLM_COST_PROMPT_DETAILS_CACHE_WRITE,
  LLM_COST_INPUT,
  LLM_COST_TOTAL,
  LLM_FINISH_REASON,
  LLM_FUNCTION_CALL,
  LLM_INPUT_MESSAGES,
  LLM_INVOCATION_PARAMETERS,
  LLM_MODEL_NAME,
  LLM_OUTPUT_MESSAGES,
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VARIABLES,
  PROMPT_TEMPLATE_VERSION,
  LLM_PROMPTS,
  LLM_PROVIDER,
  LLM_REQUEST_MODEL_NAME,
  LLM_RESPONSE_MODEL_NAME,
  LLM_SYSTEM,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO,
  LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_INPUT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
  LLM_TOKEN_COUNT_TOTAL,
  LLM_TOOLS,
  MESSAGE_CONTENT,
  MESSAGE_CONTENTS,
  MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON,
  MESSAGE_FUNCTION_CALL_NAME,
  MESSAGE_NAME,
  MESSAGE_ROLE,
  MESSAGE_TOOL_CALL_ID,
  MESSAGE_TOOL_CALLS,
  MESSAGE_CONTENT_AUDIO,
  MESSAGE_CONTENT_DATA,
  MESSAGE_CONTENT_DOCUMENT,
  MESSAGE_CONTENT_ENCRYPTED_CONTENT,
  MESSAGE_CONTENT_ID,
  MESSAGE_CONTENT_IMAGE,
  MESSAGE_CONTENT_SIGNATURE,
  MESSAGE_CONTENT_TEXT,
  MESSAGE_CONTENT_TYPE,
  MESSAGE_CONTENT_VIDEO,
  METADATA,
  OPENINFERENCE_SPAN_KIND: "openinference.span.kind",
  OUTPUT_DOCUMENTS,
  OUTPUT_IMAGES,
  OUTPUT_MIME_TYPE,
  OUTPUT_VALUE,
  PROMPT_ID,
  PROMPT_TEXT,
  PROMPT_URL,
  PROMPT_VENDOR,
  RERANKER_INPUT_DOCUMENTS,
  RERANKER_MODEL_NAME,
  RERANKER_OUTPUT_DOCUMENTS,
  RERANKER_QUERY,
  RERANKER_TOP_K,
  RETRIEVAL_DOCUMENTS,
  SESSION_ANNOTATIONS,
  SESSION_EVALUATIONS,
  SESSION_ID,
  TAG_TAGS,
  TOOL_DESCRIPTION,
  TOOL_ID,
  TOOL_JSON_SCHEMA,
  TOOL_NAME,
  TOOL_PARAMETERS,
  TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
  TOOL_CALL_FUNCTION_NAME,
  TOOL_CALL_ID,
  TOOL_CALL_REASONING_SIGNATURE,
  TRACE_ANNOTATIONS,
  TRACE_EVALUATIONS,
  USER_ID,
  VIDEO_URL,
  LLM_TOKEN_COUNT_PROMPT_DETAILS,
  LLM_TOKEN_COUNT_COMPLETION_DETAILS,
  LLM_COST,
} as const;

export enum AnnotatorKind {
  HUMAN = "HUMAN",
  LLM = "LLM",
  CODE = "CODE",
}

export enum OpenInferenceSpanKind {
  LLM = "LLM",
  CHAIN = "CHAIN",
  TOOL = "TOOL",
  RETRIEVER = "RETRIEVER",
  RERANKER = "RERANKER",
  EMBEDDING = "EMBEDDING",
  AGENT = "AGENT",
  GUARDRAIL = "GUARDRAIL",
  EVALUATOR = "EVALUATOR",
  PROMPT = "PROMPT",
  DECISION = "DECISION",
  UNKNOWN = "UNKNOWN",
}

/**
 * An enum of common mime types. Not exhaustive.
 */
export enum MimeType {
  TEXT = "text/plain",
  JSON = "application/json",
  AUDIO_WAV = "audio/wav",
}

export enum LLMSystem {
  OPENAI = "openai",
  ANTHROPIC = "anthropic",
  MISTRALAI = "mistralai",
  COHERE = "cohere",
  VERTEXAI = "vertexai",
  DEEPSEEK = "deepseek",
  AMAZON = "amazon",
  META = "meta",
  AI21 = "ai21",
  TYPESAFE = "typesafe",
  XAI = "xai",
}

export enum LLMProvider {
  OPENAI = "openai",
  ANTHROPIC = "anthropic",
  MISTRALAI = "mistralai",
  COHERE = "cohere",
  GOOGLE = "google",
  AWS = "aws",
  AZURE = "azure",
  XAI = "xai",
  DEEPSEEK = "deepseek",
  GROQ = "groq",
  FIREWORKS = "fireworks",
  MOONSHOT = "moonshot",
  CEREBRAS = "cerebras",
  PERPLEXITY = "perplexity",
  TOGETHER = "together",
  OLLAMA = "ollama",
  META = "meta",
  ZAI = "zai",
  MINIMAX = "minimax",
  ORACLE = "oracle",
  IBM = "ibm",
  TYPESAFE = "typesafe",
  DOUBLEWORD = "doubleword",
}

/**
 * Well-known values for `decision.system`. Each value equals the `LLMSystem` value for the same vendor.
 */
export enum DecisionSystem {
  TYPESAFE = "typesafe",
  OPENAI = "openai",
}

/**
 * Well-known values for `decision.provider`. Each value equals the `LLMProvider` value for the same vendor.
 */
export enum DecisionProvider {
  TYPESAFE = "typesafe",
  OPENAI = "openai",
}
