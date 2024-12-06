/**
 * Semantic conventions for OpenInference tracing
 */

export const SemanticAttributePrefixes = {
  input: "input",
  output: "output",
  llm: "llm",
  retrieval: "retrieval",
  reranker: "reranker",
  messages: "messages",
  message: "message",
  document: "document",
  embedding: "embedding",
  tool: "tool",
  tool_call: "tool_call",
  metadata: "metadata",
  tag: "tag",
  session: "session",
  user: "user",
  openinference: "openinference",
  message_content: "message_content",
  image: "image",
  audio: "audio",
} as const;

export const LLMAttributePostfixes = {
  provider: "provider",
  system: "system",
  model_name: "model_name",
  token_count: "token_count",
  input_messages: "input_messages",
  output_messages: "output_messages",
  invocation_parameters: "invocation_parameters",
  prompts: "prompts",
  prompt_template: "prompt_template",
  function_call: "function_call",
  tools: "tools",
} as const;

export const LLMPromptTemplateAttributePostfixes = {
  variables: "variables",
  template: "template",
} as const;

export const RetrievalAttributePostfixes = {
  documents: "documents",
} as const;

export const RerankerAttributePostfixes = {
  input_documents: "input_documents",
  output_documents: "output_documents",
  query: "query",
  model_name: "model_name",
  top_k: "top_k",
} as const;

export const EmbeddingAttributePostfixes = {
  embeddings: "embeddings",
  text: "text",
  model_name: "model_name",
  vector: "vector",
} as const;

export const ToolAttributePostfixes = {
  name: "name",
  description: "description",
  parameters: "parameters",
  json_schema: "json_schema",
} as const;

export const MessageAttributePostfixes = {
  role: "role",
  content: "content",
  contents: "contents",
  name: "name",
  function_call_name: "function_call_name",
  function_call_arguments_json: "function_call_arguments_json",
  tool_calls: "tool_calls",
  tool_call_id: "tool_call_id",
} as const;

export const MessageContentsAttributePostfixes = {
  type: "type",
  text: "text",
  image: "image",
} as const;

export const ImageAttributesPostfixes = {
  url: "url",
} as const;

export const ToolCallAttributePostfixes = {
  function_name: "function.name",
  function_arguments_json: "function.arguments",
  id: "id",
} as const;

export const DocumentAttributePostfixes = {
  id: "id",
  content: "content",
  score: "score",
  metadata: "metadata",
} as const;

export const TagAttributePostfixes = {
  tags: "tags",
} as const;

export const SessionAttributePostfixes = {
  id: "id",
} as const;

export const UserAttributePostfixes = {
  id: "id",
} as const;

export const AudioAttributesPostfixes = {
  url: "url",
  mime_type: "mime_type",
  transcript: "transcript",
} as const;

/**
 * The input to any span
 */
export const INPUT_VALUE = `${SemanticAttributePrefixes.input}.value` as const;
export const INPUT_MIME_TYPE =
  `${SemanticAttributePrefixes.input}.mime_type` as const;
/**
 * The output of any span
 */
export const OUTPUT_VALUE =
  `${SemanticAttributePrefixes.output}.value` as const;
export const OUTPUT_MIME_TYPE =
  `${SemanticAttributePrefixes.output}.mime_type` as const;
/**
 * The messages sent to the LLM for completions
 * Typically seen in OpenAI chat completions
 * @see https://beta.openai.com/docs/api-reference/completions/create
 */
export const LLM_INPUT_MESSAGES =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.input_messages}` as const;

/**
 * The prompts sent to the LLM for completions
 * Typically seen in OpenAI legacy completions
 * @see https://beta.openai.com/docs/api-reference/completions/create
 */
export const LLM_PROMPTS =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.prompts}` as const;

/**
 * The JSON representation of the parameters passed to the LLM
 */
export const LLM_INVOCATION_PARAMETERS =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.invocation_parameters}` as const;

/**
 * The messages received from the LLM for completions
 * Typically seen in OpenAI chat completions
 * @see https://platform.openai.com/docs/api-reference/chat/object#choices-message
 */
export const LLM_OUTPUT_MESSAGES =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.output_messages}` as const;

/**
 * The name of the LLM model
 */
export const LLM_MODEL_NAME =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.model_name}` as const;

/**
 * The provider of the inferences. E.g. the cloud provider
 */
export const LLM_PROVIDER =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.provider}` as const;

/**
 * The AI product as identified by the client or server
 */
export const LLM_SYSTEM =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.system}` as const;

/** Token count for the completion by the llm */
export const LLM_TOKEN_COUNT_COMPLETION =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.token_count}.completion` as const;

/** Token count for the prompt to the llm */
export const LLM_TOKEN_COUNT_PROMPT =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.token_count}.prompt` as const;

/** Token count for the entire transaction with the llm */
export const LLM_TOKEN_COUNT_TOTAL =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.token_count}.total` as const;
/**
 * The role that the LLM assumes the message is from
 * during the LLM invocation
 */
export const MESSAGE_ROLE =
  `${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.role}` as const;

/**
 * The name of the message. This is only used for role 'function' where the name
 * of the function is captured in the name field and the parameters are captured in the
 * content.
 */
export const MESSAGE_NAME =
  `${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.name}` as const;

/**
 * The tool calls generated by the model, such as function calls.
 */
export const MESSAGE_TOOL_CALLS =
  `${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.tool_calls}` as const;

/**
 * The id of the tool call on a "tool" role message
 */
export const MESSAGE_TOOL_CALL_ID =
  `${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.tool_call_id}` as const;

/**
 * tool_call.function.name
 */
export const TOOL_CALL_FUNCTION_NAME =
  `${SemanticAttributePrefixes.tool_call}.${ToolCallAttributePostfixes.function_name}` as const;

/**
 * tool_call.function.argument (JSON string)
 */
export const TOOL_CALL_FUNCTION_ARGUMENTS_JSON =
  `${SemanticAttributePrefixes.tool_call}.${ToolCallAttributePostfixes.function_arguments_json}` as const;

/**
 * The id of the tool call
 */
export const TOOL_CALL_ID =
  `${SemanticAttributePrefixes.tool_call}.${ToolCallAttributePostfixes.id}` as const;

/**
 * The LLM function call function name
 */
export const MESSAGE_FUNCTION_CALL_NAME =
  `${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.function_call_name}` as const;

/**
 * The LLM function call function arguments in a json string
 */
export const MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON =
  `${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.function_call_arguments_json}` as const;
/**
 * The content of the message sent to the LLM
 */
export const MESSAGE_CONTENT =
  `${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.content}` as const;
/**
 * The array of contents for the message sent to the LLM. Each element of the array is
 * an `message_content` object.
 */
export const MESSAGE_CONTENTS =
  `${SemanticAttributePrefixes.message}.${MessageAttributePostfixes.contents}` as const;
/**
 * The type of content sent to the LLM
 */
export const MESSAGE_CONTENT_TYPE =
  `${SemanticAttributePrefixes.message_content}.${MessageContentsAttributePostfixes.type}` as const;
/**
 * The text content of the message sent to the LLM
 */
export const MESSAGE_CONTENT_TEXT =
  `${SemanticAttributePrefixes.message_content}.${MessageContentsAttributePostfixes.text}` as const;
/**
 * The image content of the message sent to the LLM
 */
export const MESSAGE_CONTENT_IMAGE =
  `${SemanticAttributePrefixes.message_content}.${MessageContentsAttributePostfixes.image}` as const;
/**
 * The http or base64 link to the image
 */
export const IMAGE_URL =
  `${SemanticAttributePrefixes.image}.${ImageAttributesPostfixes.url}` as const;

export const DOCUMENT_ID =
  `${SemanticAttributePrefixes.document}.${DocumentAttributePostfixes.id}` as const;

export const DOCUMENT_CONTENT =
  `${SemanticAttributePrefixes.document}.${DocumentAttributePostfixes.content}` as const;

export const DOCUMENT_SCORE =
  `${SemanticAttributePrefixes.document}.${DocumentAttributePostfixes.score}` as const;

export const DOCUMENT_METADATA =
  `${SemanticAttributePrefixes.document}.${DocumentAttributePostfixes.metadata}` as const;

/**
 * The text that was embedded to create the vector
 */
export const EMBEDDING_TEXT =
  `${SemanticAttributePrefixes.embedding}.${EmbeddingAttributePostfixes.text}` as const;

/**
 * The name of the model that was used to create the vector
 */
export const EMBEDDING_MODEL_NAME =
  `${SemanticAttributePrefixes.embedding}.${EmbeddingAttributePostfixes.model_name}` as const;

/**
 * The embedding vector. Typically a high dimensional vector of floats or ints
 */
export const EMBEDDING_VECTOR =
  `${SemanticAttributePrefixes.embedding}.${EmbeddingAttributePostfixes.vector}` as const;

/**
 * The embedding list root
 */
export const EMBEDDING_EMBEDDINGS =
  `${SemanticAttributePrefixes.embedding}.${EmbeddingAttributePostfixes.embeddings}` as const;

/**
 * The retrieval documents list root
 */
export const RETRIEVAL_DOCUMENTS =
  `${SemanticAttributePrefixes.retrieval}.${RetrievalAttributePostfixes.documents}` as const;

const PROMPT_TEMPLATE_PREFIX =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.prompt_template}` as const;

/**
 * The JSON representation of the variables used in the prompt template
 */
export const PROMPT_TEMPLATE_VARIABLES =
  `${PROMPT_TEMPLATE_PREFIX}.variables` as const;

/**
 * A prompt template
 */
export const PROMPT_TEMPLATE_TEMPLATE =
  `${PROMPT_TEMPLATE_PREFIX}.template` as const;

/**
 * The JSON representation of a function call of an LLM
 */
export const LLM_FUNCTION_CALL =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.function_call}` as const;

/**
 * List of tools that are advertised to the LLM to be able to call
 */
export const LLM_TOOLS =
  `${SemanticAttributePrefixes.llm}.${LLMAttributePostfixes.tools}` as const;

/**
 * The name of a tool
 */
export const TOOL_NAME =
  `${SemanticAttributePrefixes.tool}.${ToolAttributePostfixes.name}` as const;

/**
 * The description of a tool
 */
export const TOOL_DESCRIPTION =
  `${SemanticAttributePrefixes.tool}.${ToolAttributePostfixes.description}` as const;

/**
 * The parameters of the tool represented as a JSON string
 */
export const TOOL_PARAMETERS =
  `${SemanticAttributePrefixes.tool}.${ToolAttributePostfixes.parameters}` as const;

/**
 * The json schema of a tool input, It is RECOMMENDED that this be in the
 * OpenAI tool calling format: https://platform.openai.com/docs/assistants/tools
 */
export const TOOL_JSON_SCHEMA =
  `${SemanticAttributePrefixes.tool}.${ToolAttributePostfixes.json_schema}` as const;

/**
 * The session id of a trace. Used to correlate spans in a single session.
 */
export const SESSION_ID =
  `${SemanticAttributePrefixes.session}.${SessionAttributePostfixes.id}` as const;

/**
 * The user id of a trace. Used to correlate spans for a single user.
 */
export const USER_ID =
  `${SemanticAttributePrefixes.user}.${UserAttributePostfixes.id}` as const;

/**
 * The documents used as input to the reranker
 */
export const RERANKER_INPUT_DOCUMENTS =
  `${SemanticAttributePrefixes.reranker}.${RerankerAttributePostfixes.input_documents}` as const;

/**
 * The documents output by the reranker
 */
export const RERANKER_OUTPUT_DOCUMENTS =
  `${SemanticAttributePrefixes.reranker}.${RerankerAttributePostfixes.output_documents}` as const;

/**
 * The query string for the reranker
 */
export const RERANKER_QUERY =
  `${SemanticAttributePrefixes.reranker}.${RerankerAttributePostfixes.query}` as const;

/**
 * The model name for the reranker
 */
export const RERANKER_MODEL_NAME =
  `${SemanticAttributePrefixes.reranker}.${RerankerAttributePostfixes.model_name}` as const;

/**
 * The top k parameter for the reranker
 */
export const RERANKER_TOP_K =
  `${SemanticAttributePrefixes.reranker}.${RerankerAttributePostfixes.top_k}` as const;

/**
 * Metadata for a span, used to store user-defined key-value pairs
 */
export const METADATA = "metadata" as const;

/**
 * A prompt template version
 */
export const PROMPT_TEMPLATE_VERSION =
  `${PROMPT_TEMPLATE_PREFIX}.version` as const;

/**
 * The tags associated with a span
 */
export const TAG_TAGS =
  `${SemanticAttributePrefixes.tag}.${TagAttributePostfixes.tags}` as const;

/**
 * The url of an audio file
 */
export const AUDIO_URL =
  `${SemanticAttributePrefixes.audio}.${AudioAttributesPostfixes.url}` as const;

/**
 * The audio mime type
 */
export const AUDIO_MIME_TYPE =
  `${SemanticAttributePrefixes.audio}.${AudioAttributesPostfixes.mime_type}` as const;

/**
 * The audio transcript as text
 */
export const AUDIO_TRANSCRIPT =
  `${SemanticAttributePrefixes.audio}.${AudioAttributesPostfixes.transcript}` as const;

export const SemanticConventions = {
  IMAGE_URL,
  INPUT_VALUE,
  INPUT_MIME_TYPE,
  OUTPUT_VALUE,
  OUTPUT_MIME_TYPE,
  LLM_INPUT_MESSAGES,
  LLM_OUTPUT_MESSAGES,
  LLM_MODEL_NAME,
  LLM_PROMPTS,
  LLM_INVOCATION_PARAMETERS,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_TOTAL,
  LLM_SYSTEM,
  LLM_PROVIDER,
  LLM_TOOLS,
  MESSAGE_ROLE,
  MESSAGE_NAME,
  MESSAGE_TOOL_CALLS,
  MESSAGE_TOOL_CALL_ID,
  TOOL_CALL_ID,
  TOOL_CALL_FUNCTION_NAME,
  TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
  MESSAGE_FUNCTION_CALL_NAME,
  MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON,
  MESSAGE_CONTENT,
  MESSAGE_CONTENTS,
  MESSAGE_CONTENT_IMAGE,
  MESSAGE_CONTENT_TEXT,
  MESSAGE_CONTENT_TYPE,
  DOCUMENT_ID,
  DOCUMENT_CONTENT,
  DOCUMENT_SCORE,
  DOCUMENT_METADATA,
  EMBEDDING_EMBEDDINGS,
  EMBEDDING_TEXT,
  EMBEDDING_MODEL_NAME,
  EMBEDDING_VECTOR,
  TOOL_DESCRIPTION,
  TOOL_NAME,
  TOOL_PARAMETERS,
  TOOL_JSON_SCHEMA,
  PROMPT_TEMPLATE_VARIABLES,
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VERSION,
  RERANKER_INPUT_DOCUMENTS,
  RERANKER_OUTPUT_DOCUMENTS,
  RERANKER_QUERY,
  RERANKER_MODEL_NAME,
  RERANKER_TOP_K,
  LLM_FUNCTION_CALL,
  RETRIEVAL_DOCUMENTS,
  SESSION_ID,
  USER_ID,
  METADATA,
  TAG_TAGS,
  OPENINFERENCE_SPAN_KIND: `${SemanticAttributePrefixes.openinference}.span.kind`,
} as const;

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
}

export enum LLMProvider {
  OPENAI = "openai",
  ANTHROPIC = "anthropic",
  MISTRALAI = "mistralai",
  COHERE = "cohere",
  // Cloud Providers of LLM systems
  GOOGLE = "google",
  AWS = "aws",
  AZURE = "azure",
}
