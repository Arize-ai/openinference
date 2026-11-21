// Code generated from the semantic convention registry in semantic-conventions/model. DO NOT EDIT.
// Change the model or semantic-conventions/templates/registry/go, then run
// `make -C semantic-conventions generate`.

package semconv

// Attribute keys. Lists of objects are flattened with zero-based indices, e.g.
// "llm.input_messages.0.message.role"; use the indexer helpers to build those
// keys.
const (
	// AgentName: The name of the agent that this span represents.
	AgentName = "agent.name"

	// AnnotationAnnotatorKind: Judge kind: `HUMAN`, `LLM`, `CODE`, or custom. See [Annotator Kinds](https://github.com/Arize-ai/openinference/blob/main/spec/feedback/annotations.md#annotator-kinds)
	AnnotationAnnotatorKind = "annotation.annotator_kind"

	// AnnotationExplanation: Reason or evidence for the result
	AnnotationExplanation = "annotation.explanation"

	// AnnotationIdentifier: Stable result ID for the same name and target
	AnnotationIdentifier = "annotation.identifier"

	// AnnotationLabel: Categorical result
	AnnotationLabel = "annotation.label"

	// AnnotationMetadata: Additional result or judge data
	AnnotationMetadata = "annotation.metadata"

	// AnnotationName: Criterion or metric name
	AnnotationName = "annotation.name"

	// AnnotationScore: Numeric result.
	AnnotationScore = "annotation.score"

	// Annotations: Span-scoped feedback using annotation terminology, flattened as `annotations.0.annotation.*`
	Annotations = "annotations"

	// AudioMimeType: The MIME type of the audio file.
	AudioMimeType = "audio.mime_type"

	// AudioTranscript: The transcript of the audio file (e.g. a Whisper transcription)
	AudioTranscript = "audio.transcript"

	// AudioURL: The URL of an audio file (e.g. cloud storage), or a base64 data URI
	AudioURL = "audio.url"

	// CompletionText: The text of a choice returned from a completions API. Used under `llm.choices`.
	CompletionText = "completion.text"

	// DecisionModelName: The name of the decision model being utilized. Mirrors `llm.model_name` for `DECISION` spans.
	DecisionModelName = "decision.model_name"

	// DecisionProvider: The hosting provider of the decision model: who runs the hardware that answered, e.g. `typesafe` when calling TypeSafe directly, `openai` for the OpenAI Decisions API, or the cloud or self-hosting provider for a model served elsewhere. Distinct from `decision.system`, which names the API shape. The well-known values alias the matching `llm.provider` values.
	DecisionProvider = "decision.provider"

	// DecisionRequestModelName: The decision model requested by the caller, as sent in the request. May differ from `decision.response.model_name` when the provider resolves an alias or routes the request.
	DecisionRequestModelName = "decision.request.model_name"

	// DecisionResponseModelName: The decision model that actually produced the response, as reported by the provider. May differ from `decision.request.model_name`.
	DecisionResponseModelName = "decision.response.model_name"

	// DecisionSystem: The decision API ecosystem the call conforms to, i.e. which request/response shape the client speaks, as identified by the client or server instrumentation: `typesafe` for the TypeSafe System One / Jev API, `openai` for the OpenAI Decisions API. A self-hosted vLLM server answering the Jev-compatible `/v1/systemone` shape is still `typesafe`. Distinct from `decision.provider`, which says who hosts the model. The well-known values alias the matching `llm.system` values.
	DecisionSystem = "decision.system"

	// DecisionTokenCountInput: The number of input tokens consumed by a decision model call: the state, questions, and candidate options. Maps to `usage.input_tokens` in TypeSafe System One responses.
	DecisionTokenCountInput = "decision.token_count.input"

	// DecisionTokenCountOutput: The number of output tokens produced by a decision model call. Decision models emit typed answers rather than text, so this is typically small. Maps to `usage.output_tokens` in TypeSafe System One responses.
	DecisionTokenCountOutput = "decision.token_count.output"

	// DocumentContent: The text content of a document: a retrieved passage, or an inline text document passed to a model (e.g. an Anthropic `text` source)
	DocumentContent = "document.content"

	// DocumentID: Unique identifier for a document: a retrieved document or chunk in a retrieval system, or the provider file id (e.g. an OpenAI or Anthropic Files API id) for a document passed to a model.
	DocumentID = "document.id"

	// DocumentMetadata: Metadata associated with a document
	DocumentMetadata = "document.metadata"

	// DocumentMimeType: The MIME type of a file document such as a PDF
	DocumentMimeType = "document.mime_type"

	// DocumentName: The name supplied with a document, copied verbatim: a file name (OpenAI `filename`) or a title (Anthropic `title`, Bedrock `name`, Gemini `display_name`); not a unique identifier
	DocumentName = "document.name"

	// DocumentScore: Score representing the relevance of a document
	DocumentScore = "document.score"

	// DocumentURL: The URL, object-store URI, or base64 data URI of a file document such as a PDF
	DocumentURL = "document.url"

	// EmbeddingEmbeddings: List of embedding objects including text and vector data
	EmbeddingEmbeddings = "embedding.embeddings"

	// EmbeddingInvocationParameters: Parameters used during the invocation of an embedding model or API (excluding input)
	EmbeddingInvocationParameters = "embedding.invocation_parameters"

	// EmbeddingModelName: Name of the embedding model used
	EmbeddingModelName = "embedding.model_name"

	// EmbeddingText: The text represented in the embedding
	EmbeddingText = "embedding.text"

	// EmbeddingVector: The embedding vector consisting of a list of floats
	EmbeddingVector = "embedding.vector"

	// EvaluationAnnotatorKind: Judge kind: `HUMAN`, `LLM`, `CODE`, or custom. See [Annotator Kinds](https://github.com/Arize-ai/openinference/blob/main/spec/feedback/annotations.md#annotator-kinds)
	EvaluationAnnotatorKind = "evaluation.annotator_kind"

	// EvaluationExplanation: Reason or evidence for the result
	EvaluationExplanation = "evaluation.explanation"

	// EvaluationIdentifier: Stable result ID for the same name and target
	EvaluationIdentifier = "evaluation.identifier"

	// EvaluationLabel: Categorical result
	EvaluationLabel = "evaluation.label"

	// EvaluationMetadata: Additional result or judge data
	EvaluationMetadata = "evaluation.metadata"

	// EvaluationName: Criterion or metric name
	EvaluationName = "evaluation.name"

	// EvaluationScore: Numeric result.
	EvaluationScore = "evaluation.score"

	// Evaluations: Span-scoped feedback using evaluation or eval terminology, flattened as `evaluations.0.evaluation.*`
	Evaluations = "evaluations"

	// GraphNodeID: The id of the node in the execution graph. This along with `graph.node.parent_id` are used to visualize the execution graph.
	GraphNodeID = "graph.node.id"

	// GraphNodeName: The name of the node in the execution graph. Use this to present a human readable name for the node. Optional
	GraphNodeName = "graph.node.name"

	// GraphNodeParentID: This references the id of the parent node. Leaving this unset or set as empty string implies that the current span is the root node.
	GraphNodeParentID = "graph.node.parent_id"

	// ImageURL: The link to the image or its base64 encoding
	ImageURL = "image.url"

	// InputDocuments: File document (e.g. PDF) inputs to an operation, independent of span kind, e.g. `input.documents.0.document.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-documents)
	InputDocuments = "input.documents"

	// InputImages: Image inputs to an operation, independent of span kind, e.g. `input.images.0.image.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-images)
	InputImages = "input.images"

	// InputMimeType: MIME type representing the format of `input.value`
	InputMimeType = "input.mime_type"

	// InputValue: The input value to an operation
	InputValue = "input.value"

	// LLMChoices: Text choices returned from a completions API, e.g. `llm.choices.0.completion.text`
	LLMChoices = "llm.choices"

	// LLMCostCompletion: Total cost of all output tokens generated by the LLM in USD
	LLMCostCompletion = "llm.cost.completion"

	// LLMCostCompletionDetailsAudio: Cost of audio tokens in the completion in USD
	LLMCostCompletionDetailsAudio = "llm.cost.completion_details.audio"

	// LLMCostCompletionDetailsOutput: Total cost of output tokens in USD
	LLMCostCompletionDetailsOutput = "llm.cost.completion_details.output"

	// LLMCostCompletionDetailsReasoning: Cost of reasoning steps in the completion in USD
	LLMCostCompletionDetailsReasoning = "llm.cost.completion_details.reasoning"

	// LLMCostPrompt: Total cost of all input tokens sent to the LLM in USD
	LLMCostPrompt = "llm.cost.prompt"

	// LLMCostPromptDetailsAudio: Cost of audio tokens in the prompt in USD
	LLMCostPromptDetailsAudio = "llm.cost.prompt_details.audio"

	// LLMCostPromptDetailsCacheInput: Cost of input tokens in the prompt that were cached in USD
	LLMCostPromptDetailsCacheInput = "llm.cost.prompt_details.cache_input"

	// LLMCostPromptDetailsCacheRead: Cost of prompt tokens read from cache in USD
	LLMCostPromptDetailsCacheRead = "llm.cost.prompt_details.cache_read"

	// LLMCostPromptDetailsCacheWrite: Cost of prompt tokens written to cache in USD
	LLMCostPromptDetailsCacheWrite = "llm.cost.prompt_details.cache_write"

	// LLMCostPromptDetailsInput: Total cost of input tokens in USD
	LLMCostPromptDetailsInput = "llm.cost.prompt_details.input"

	// LLMCostTotal: Total cost of the LLM call in USD (prompt + completion)
	LLMCostTotal = "llm.cost.total"

	// LLMFinishReason: The reason the model stopped generating tokens.
	LLMFinishReason = "llm.finish_reason"

	// LLMFunctionCall: Object recording details of a function call in models or APIs
	//
	// Deprecated: Use `message.tool_calls` instead.
	LLMFunctionCall = "llm.function_call"

	// LLMInputMessages: List of messages sent to the LLM in a chat API request, e.g. `llm.input_messages.0.message.role`. See [Messages](https://github.com/Arize-ai/openinference/blob/main/spec/content/messages.md)
	LLMInputMessages = "llm.input_messages"

	// LLMInvocationParameters: Parameters used during the invocation of an LLM or API
	LLMInvocationParameters = "llm.invocation_parameters"

	// LLMModelName: The name of the language model being utilized
	LLMModelName = "llm.model_name"

	// LLMOutputMessages: List of messages received from the LLM in a chat API response, e.g. `llm.output_messages.0.message.role`. See [Messages](https://github.com/Arize-ai/openinference/blob/main/spec/content/messages.md)
	LLMOutputMessages = "llm.output_messages"

	// LLMPromptTemplate: Template used to generate prompts (e.g., using `{variable}` placeholder syntax)
	LLMPromptTemplate = "llm.prompt_template.template"

	// LLMPromptTemplateVariables: JSON of key value pairs applied to the prompt template
	LLMPromptTemplateVariables = "llm.prompt_template.variables"

	// LLMPromptTemplateVersion: The version of the prompt template
	LLMPromptTemplateVersion = "llm.prompt_template.version"

	// LLMPrompts: Prompts provided to a completions API, e.g. `llm.prompts.0.prompt.text`
	LLMPrompts = "llm.prompts"

	// LLMProvider: The hosting provider of the LLM, e.g. `azure`. See [well-known values](https://github.com/Arize-ai/openinference/blob/main/spec/attributes/semantic-conventions.md#llm-provider-values).
	LLMProvider = "llm.provider"

	// LLMRequestModelName: The model requested by the caller, as sent in the request. May differ from `llm.response.model_name` when the provider routes the request to a different model.
	LLMRequestModelName = "llm.request.model_name"

	// LLMResponseModelName: The model that actually generated the response, as reported by the provider. May differ from `llm.request.model_name` when the provider routes the request to a different model.
	LLMResponseModelName = "llm.response.model_name"

	// LLMSystem: The AI product as identified by the client or server instrumentation. See [well-known values](https://github.com/Arize-ai/openinference/blob/main/spec/attributes/semantic-conventions.md#llm-system-values).
	LLMSystem = "llm.system"

	// LLMTokenCountCompletion: The number of tokens in the completion
	LLMTokenCountCompletion = "llm.token_count.completion"

	// LLMTokenCountCompletionDetailsAudio: The number of audio tokens generated by the model
	LLMTokenCountCompletionDetailsAudio = "llm.token_count.completion_details.audio"

	// LLMTokenCountCompletionDetailsReasoning: The number of tokens used for model reasoning
	LLMTokenCountCompletionDetailsReasoning = "llm.token_count.completion_details.reasoning"

	// LLMTokenCountPrompt: The number of tokens in the prompt
	LLMTokenCountPrompt = "llm.token_count.prompt"

	// LLMTokenCountPromptDetailsAudio: The number of audio input tokens presented in the prompt
	LLMTokenCountPromptDetailsAudio = "llm.token_count.prompt_details.audio"

	// LLMTokenCountPromptDetailsCacheInput: The number of input tokens in the prompt that were cached
	LLMTokenCountPromptDetailsCacheInput = "llm.token_count.prompt_details.cache_input"

	// LLMTokenCountPromptDetailsCacheRead: The number of prompt tokens successfully retrieved from cache (cache hits). Maps to `cached_tokens` in OpenAI responses
	LLMTokenCountPromptDetailsCacheRead = "llm.token_count.prompt_details.cache_read"

	// LLMTokenCountPromptDetailsCacheWrite: The number of prompt tokens not found in cache that were written to cache (cache misses). Maps to `cache_write_tokens` in OpenAI responses
	LLMTokenCountPromptDetailsCacheWrite = "llm.token_count.prompt_details.cache_write"

	// LLMTokenCountTotal: Total number of tokens, including prompt and completion
	LLMTokenCountTotal = "llm.token_count.total"

	// LLMTools: List of tools that are advertised to the LLM to be able to call, e.g. `llm.tools.0.tool.name`, `llm.tools.0.tool.description`, `llm.tools.0.tool.json_schema`. See [Tool Calling](https://github.com/Arize-ai/openinference/blob/main/spec/content/tool-calling.md)
	LLMTools = "llm.tools"

	// MessageContent: The content of a message in a chat
	MessageContent = "message.content"

	// MessageContents: The message contents to the LLM, an array of `message_content` objects. Use this form when a message has multiple content items or when item ordering matters.
	MessageContents = "message.contents"

	// MessageFunctionCallArgumentsJSON: The arguments to the function call in JSON
	//
	// Deprecated: Use `message.tool_calls` instead.
	MessageFunctionCallArgumentsJSON = "message.function_call_arguments_json"

	// MessageFunctionCallName: Function call function name
	//
	// Deprecated: Use `message.tool_calls` instead.
	MessageFunctionCallName = "message.function_call_name"

	// MessageName: The name of the function or tool that produced a tool/function role message. Used when `message.role` is `"tool"` or `"function"` to identify the origin.
	MessageName = "message.name"

	// MessageRole: Role of the entity in a message (e.g., user, system)
	MessageRole = "message.role"

	// MessageToolCallID: Tool call result identifier corresponding to `tool_call.id`
	MessageToolCallID = "message.tool_call_id"

	// MessageToolCalls: List of tool calls (e.g. function calls) generated by the LLM, e.g. `llm.output_messages.0.message.tool_calls.0.tool_call.id`
	MessageToolCalls = "message.tool_calls"

	// MessageContentAudio: The audio content of the message, if the type is `"audio"`. Nested leaves are `audio.url` and `audio.transcript`.
	MessageContentAudio = "message_content.audio"

	// MessageContentData: Opaque vendor-issued data captured verbatim. Maps to Anthropic `redacted_thinking.data`.
	MessageContentData = "message_content.data"

	// MessageContentDocument: The document content of the message (e.g. a PDF, an uploaded file, or inline text), if the type is `"document"`. Nested leaves are `document.url`, `document.content`, `document.mime_type`, `document.id`, and `document.name`.
	MessageContentDocument = "message_content.document"

	// MessageContentEncryptedContent: OpenAI `encrypted_content` captured verbatim.
	MessageContentEncryptedContent = "message_content.encrypted_content"

	// MessageContentID: Provider-assigned identifier for this message content item. For OpenAI Responses reasoning items, this maps to `ResponseReasoningItem.id` and should be preserved for stateless replay.
	MessageContentID = "message_content.id"

	// MessageContentImage: The image content of the message, if the type is `"image"`.
	MessageContentImage = "message_content.image"

	// MessageContentSignature: Opaque vendor-issued signature captured verbatim. Maps to provider `signature` fields, and to Gemini `thoughtSignature` fields when the signature is attached to a non-tool content part such as text.
	MessageContentSignature = "message_content.signature"

	// MessageContentText: The text content of the message. Used when `message_content.type` is `"text"` or `"reasoning"` (visible reasoning or summary text). For OpenAI `summary_text` arrays, concatenate the array items into this single text value for now.
	MessageContentText = "message_content.text"

	// MessageContentType: The type of the content. Use `"reasoning"` for reasoning/thinking content, including Anthropic `redacted_thinking` blocks. Use `"tool_use"` when a provider returns a tool call as an ordered content part.
	MessageContentType = "message_content.type"

	// MessageContentVideo: The video content of the message, if the type is `"video"`. The nested leaf is `video.url`. Consumers infer MIME type from the URL path extension or a data URI prefix.
	MessageContentVideo = "message_content.video"

	// Metadata: Metadata associated with a span
	Metadata = "metadata"

	// OpenInferenceProjectName: The name of the project that the telemetry belongs to.
	OpenInferenceProjectName = "openinference.project.name"

	// OpenInferenceSpanKind: The role an operation plays in an AI application.
	OpenInferenceSpanKind = "openinference.span.kind"

	// OutputDocuments: File document (e.g. PDF) outputs of an operation, independent of span kind, e.g. `output.documents.0.document.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-documents)
	OutputDocuments = "output.documents"

	// OutputImages: Image outputs of an operation, independent of span kind, e.g. `output.images.0.image.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-images)
	OutputImages = "output.images"

	// OutputMimeType: MIME type representing the format of `output.value`
	OutputMimeType = "output.mime_type"

	// OutputValue: The output value of an operation
	OutputValue = "output.value"

	// PromptID: A vendor-specific id used to identify the prompt.
	PromptID = "prompt.id"

	// PromptText: The text of a prompt in a completions API request. Used under `llm.prompts`.
	PromptText = "prompt.text"

	// PromptURL: A vendor-specific URL used to locate the prompt via the web
	PromptURL = "prompt.url"

	// PromptVendor: The vendor or origin of the prompt, e.g. `langsmith`, `portkey`, `arize-phoenix`
	PromptVendor = "prompt.vendor"

	// RerankerInputDocuments: List of documents as input to the reranker
	RerankerInputDocuments = "reranker.input_documents"

	// RerankerModelName: Model name of the reranker
	RerankerModelName = "reranker.model_name"

	// RerankerOutputDocuments: List of documents outputted by the reranker
	RerankerOutputDocuments = "reranker.output_documents"

	// RerankerQuery: Query parameter of the reranker
	RerankerQuery = "reranker.query"

	// RerankerTopK: Top K parameter of the reranker
	RerankerTopK = "reranker.top_k"

	// RetrievalDocuments: List of retrieved documents
	RetrievalDocuments = "retrieval.documents"

	// SessionAnnotations: Session-scoped feedback using annotation terminology, flattened as `session.annotations.0.annotation.*`; requires `session.id`
	SessionAnnotations = "session.annotations"

	// SessionEvaluations: Session-scoped feedback using evaluation terminology, flattened as `session.evaluations.0.evaluation.*`; requires `session.id`
	SessionEvaluations = "session.evaluations"

	// SessionID: Unique identifier for a session, such as a multi-turn conversation.
	SessionID = "session.id"

	// TagTags: List of tags to give the span a category
	TagTags = "tag.tags"

	// ToolDescription: Description of the tool's purpose and functionality. On a `TOOL` span, the tool being invoked; under `llm.tools.<index>`, an advertised tool definition
	ToolDescription = "tool.description"

	// ToolID: The identifier for the result of the tool call (corresponding to `tool_call.id`)
	ToolID = "tool.id"

	// ToolJSONSchema: The JSON schema of a tool input
	ToolJSONSchema = "tool.json_schema"

	// ToolName: The name of the tool. On a `TOOL` span, the tool being invoked; under `llm.tools.<index>`, the name of an advertised tool definition
	ToolName = "tool.name"

	// ToolParameters: The parameters definition for invoking the tool
	ToolParameters = "tool.parameters"

	// ToolCallFunctionArgumentsJSON: The arguments for the function being invoked by a tool call
	ToolCallFunctionArgumentsJSON = "tool_call.function.arguments"

	// ToolCallFunctionName: The name of the function being invoked by a tool call
	ToolCallFunctionName = "tool_call.function.name"

	// ToolCallID: The id of a tool call (useful when there is more than one call at the same time)
	ToolCallID = "tool_call.id"

	// ToolCallReasoningSignature: Opaque vendor-issued reasoning echo token attached to a tool call. Used by Gemini, which puts `thoughtSignature` on `functionCall` parts rather than on the reasoning summary.
	ToolCallReasoningSignature = "tool_call.reasoning_signature"

	// TraceAnnotations: Trace-scoped feedback using annotation terminology, flattened as `trace.annotations.0.annotation.*`
	TraceAnnotations = "trace.annotations"

	// TraceEvaluations: Trace-scoped feedback using evaluation terminology, flattened as `trace.evaluations.0.evaluation.*`
	TraceEvaluations = "trace.evaluations"

	// UserID: Unique identifier for a user.
	UserID = "user.id"

	// VideoURL: The URL, object-store URI, or base64 data URI of a video. Consumers infer MIME type from the path extension (`.mp4` to `video/mp4`, `.webm` to `video/webm`, `.ogv` to `video/ogg`, `.avi` to `video/x-msvideo`) or from a data URI prefix.
	VideoURL = "video.url"

	// LLMTokenCountPromptDetails is the key prefix for additional prompt token count details. Each detail is a separate attribute with this prefix, e.g. llm.token_count.prompt_details.audio. All values are in tokens.
	LLMTokenCountPromptDetails = "llm.token_count.prompt_details"

	// LLMCostPromptDetails is the key prefix for additional prompt cost details. Each detail is a separate attribute with this prefix, e.g. llm.cost.prompt_details.audio. All values are in USD.
	LLMCostPromptDetails = "llm.cost.prompt_details"

	// LLMCostCompletionDetails is the key prefix for additional completion cost details. Each detail is a separate attribute with this prefix, e.g. llm.cost.completion_details.audio. All values are in USD.
	LLMCostCompletionDetails = "llm.cost.completion_details"
)
