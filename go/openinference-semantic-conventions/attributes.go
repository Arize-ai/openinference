package semconv

// Span-level attributes — set on the span representing the operation.
const (
	// OpenInferenceSpanKind classifies the span (LLM, CHAIN, TOOL, etc.).
	// Use the SpanKind* constants below as the value.
	OpenInferenceSpanKind = "openinference.span.kind"

	// InputValue is the input to the operation. Plain string by default;
	// set InputMimeType to "application/json" if the value is a JSON string.
	InputValue    = "input.value"
	InputMimeType = "input.mime_type"

	// OutputValue is the output of the operation. Plain string by default;
	// set OutputMimeType to "application/json" if the value is a JSON string.
	OutputValue    = "output.value"
	OutputMimeType = "output.mime_type"

	// Metadata is a JSON-encoded map of user-defined key-value pairs.
	Metadata = "metadata"

	// TagTags is a list of custom categorical tags.
	TagTags = "tag.tags"

	// SessionID groups spans into a multi-turn conversation.
	SessionID = "session.id"
	// UserID identifies the user the span ran on behalf of.
	UserID = "user.id"

	// AgentName names the agent. Agents performing the same function
	// should share a name.
	AgentName = "agent.name"

	// GraphNodeID is the id of a node in the execution graph. Combined with
	// GraphNodeParentID, the Arize UI renders the agent's execution graph.
	GraphNodeID       = "graph.node.id"
	GraphNodeName     = "graph.node.name"
	GraphNodeParentID = "graph.node.parent_id"
)

// LLM-span attributes — set when the span represents an LLM API call.
const (
	LLMModelName            = "llm.model_name"
	LLMProvider             = "llm.provider"
	LLMSystem               = "llm.system"
	LLMInvocationParameters = "llm.invocation_parameters"
	LLMFunctionCall         = "llm.function_call"
	LLMFinishReason         = "llm.finish_reason"

	// LLMInputMessages and LLMOutputMessages are key *prefixes*; individual
	// messages are addressed via the indexer helpers, e.g.
	// LLMInputMessageRoleKey(0) == "llm.input_messages.0.message.role".
	LLMInputMessages  = "llm.input_messages"
	LLMOutputMessages = "llm.output_messages"

	// LLMPrompts / LLMChoices are key prefixes for the completions API
	// (vs the chat API). Use the indexer helpers below.
	LLMPrompts = "llm.prompts"
	LLMChoices = "llm.choices"

	LLMPromptTemplate          = "llm.prompt_template.template"
	LLMPromptTemplateVariables = "llm.prompt_template.variables"
	LLMPromptTemplateVersion   = "llm.prompt_template.version"

	LLMTools = "llm.tools"
)

// Token-count attributes for LLM spans. Values are integer counts of tokens.
const (
	LLMTokenCountPrompt                     = "llm.token_count.prompt"
	LLMTokenCountPromptDetails              = "llm.token_count.prompt_details"
	LLMTokenCountPromptDetailsAudio         = "llm.token_count.prompt_details.audio"
	LLMTokenCountPromptDetailsCacheInput    = "llm.token_count.prompt_details.cache_input"
	LLMTokenCountPromptDetailsCacheRead     = "llm.token_count.prompt_details.cache_read"
	LLMTokenCountPromptDetailsCacheWrite    = "llm.token_count.prompt_details.cache_write"
	LLMTokenCountCompletion                 = "llm.token_count.completion"
	LLMTokenCountCompletionDetailsAudio     = "llm.token_count.completion_details.audio"
	LLMTokenCountCompletionDetailsReasoning = "llm.token_count.completion_details.reasoning"
	LLMTokenCountTotal                      = "llm.token_count.total"
)

// Cost attributes for LLM spans. Values are USD floats.
const (
	LLMCostPrompt                     = "llm.cost.prompt"
	LLMCostPromptDetails              = "llm.cost.prompt_details"
	LLMCostPromptDetailsAudio         = "llm.cost.prompt_details.audio"
	LLMCostPromptDetailsCacheInput    = "llm.cost.prompt_details.cache_input"
	LLMCostPromptDetailsCacheRead     = "llm.cost.prompt_details.cache_read"
	LLMCostPromptDetailsCacheWrite    = "llm.cost.prompt_details.cache_write"
	LLMCostPromptDetailsInput         = "llm.cost.prompt_details.input"
	LLMCostCompletion                 = "llm.cost.completion"
	LLMCostCompletionDetails          = "llm.cost.completion_details"
	LLMCostCompletionDetailsAudio     = "llm.cost.completion_details.audio"
	LLMCostCompletionDetailsOutput    = "llm.cost.completion_details.output"
	LLMCostCompletionDetailsReasoning = "llm.cost.completion_details.reasoning"
	LLMCostTotal                      = "llm.cost.total"
)

// Embedding-span attributes — set when the span represents an embedding call.
const (
	EmbeddingEmbeddings           = "embedding.embeddings"
	EmbeddingInvocationParameters = "embedding.invocation_parameters"
	EmbeddingModelName            = "embedding.model_name"

	// EmbeddingText / EmbeddingVector are nested under EmbeddingEmbeddings.{i}.
	// Use the indexer helpers below.
	EmbeddingText   = "embedding.text"
	EmbeddingVector = "embedding.vector"
)

// Tool-span attributes — set when the span represents a tool/function invocation.
const (
	ToolName        = "tool.name"
	ToolDescription = "tool.description"
	// ToolParameters is a JSON string of tool parameters; see
	// https://platform.openai.com/docs/guides/gpt/function-calling.
	ToolParameters = "tool.parameters"
	ToolID         = "tool.id"
	// ToolJSONSchema is the JSON schema for the tool input,
	// recommended in OpenAI function-calling format.
	ToolJSONSchema = "tool.json_schema"
)

// Retrieval-span attributes — set when the span represents a retrieval call.
const (
	// RetrievalDocuments is the key prefix for returned documents; use the
	// Document indexer helpers below.
	RetrievalDocuments = "retrieval.documents"
)

// Reranker-span attributes — set when the span represents a reranker call.
const (
	RerankerInputDocuments  = "reranker.input_documents"
	RerankerOutputDocuments = "reranker.output_documents"
	RerankerQuery           = "reranker.query"
	RerankerModelName       = "reranker.model_name"
	RerankerTopK            = "reranker.top_k"
)

// Message attributes — nested under LLMInputMessages.{i} or LLMOutputMessages.{i}.
// In raw form these strings start with "message.*"; use the indexer helpers
// below to build the full keys (e.g., "llm.input_messages.0.message.role").
const (
	MessageRole                      = "message.role"
	MessageContent                   = "message.content"
	MessageContents                  = "message.contents"
	MessageName                      = "message.name"
	MessageToolCalls                 = "message.tool_calls"
	MessageFunctionCallName          = "message.function_call_name"
	MessageFunctionCallArgumentsJSON = "message.function_call_arguments_json"
	MessageToolCallID                = "message.tool_call_id"
)

// Message-content attributes — for the contents array on a message.
const (
	MessageContentType             = "message_content.type"
	MessageContentText             = "message_content.text"
	MessageContentImage            = "message_content.image"
	MessageContentSignature        = "message_content.signature"
	MessageContentData             = "message_content.data"
	MessageContentEncriptedContent = "message_content.encripted_content"
)

// Image attributes — nested under MessageContentImage.
const (
	ImageURL = "image.url"
)

// Audio attributes.
const (
	AudioURL        = "audio.url"
	AudioMimeType   = "audio.mime_type"
	AudioTranscript = "audio.transcript"
)

// Document attributes — nested under RetrievalDocuments.{i}.
const (
	DocumentID       = "document.id"
	DocumentScore    = "document.score"
	DocumentContent  = "document.content"
	DocumentMetadata = "document.metadata"
)

// Tool-call attributes — nested under MessageToolCalls.{i}.
const (
	ToolCallID                    = "tool_call.id"
	ToolCallFunctionName          = "tool_call.function.name"
	ToolCallFunctionArgumentsJSON = "tool_call.function.arguments"
	ToolCallSignature             = "tool_call.signature"
)

// Completions-API attributes — nested under LLMPrompts.{i} and LLMChoices.{i}.
const (
	PromptText     = "prompt.text"
	CompletionText = "completion.text"
)

// Prompt-source attributes — identify a prompt from a prompt registry/library.
const (
	PromptVendor = "prompt.vendor"
	PromptID     = "prompt.id"
	PromptURL    = "prompt.url"
)
