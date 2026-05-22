package semconv

import "testing"

// TestAttributeKeys asserts the string values of the exported attribute
// constants match the OpenInference spec — these are the wire-format keys
// the Arize collector parses, so they must not drift from the canonical
// values in python/openinference-semantic-conventions and the spec/ dir.
func TestAttributeKeys(t *testing.T) {
	cases := []struct {
		got, want string
	}{
		{OpenInferenceSpanKind, "openinference.span.kind"},
		{InputValue, "input.value"},
		{InputMimeType, "input.mime_type"},
		{OutputValue, "output.value"},
		{OutputMimeType, "output.mime_type"},
		{Metadata, "metadata"},
		{TagTags, "tag.tags"},
		{SessionID, "session.id"},
		{UserID, "user.id"},
		{AgentName, "agent.name"},
		{GraphNodeID, "graph.node.id"},
		{GraphNodeName, "graph.node.name"},
		{GraphNodeParentID, "graph.node.parent_id"},

		{LLMModelName, "llm.model_name"},
		{LLMProvider, "llm.provider"},
		{LLMSystem, "llm.system"},
		{LLMInvocationParameters, "llm.invocation_parameters"},
		{LLMFunctionCall, "llm.function_call"},
		{LLMFinishReason, "llm.finish_reason"},
		{LLMInputMessages, "llm.input_messages"},
		{LLMOutputMessages, "llm.output_messages"},
		{LLMPrompts, "llm.prompts"},
		{LLMChoices, "llm.choices"},
		{LLMPromptTemplate, "llm.prompt_template.template"},
		{LLMPromptTemplateVariables, "llm.prompt_template.variables"},
		{LLMPromptTemplateVersion, "llm.prompt_template.version"},
		{LLMTools, "llm.tools"},

		{LLMTokenCountPrompt, "llm.token_count.prompt"},
		{LLMTokenCountPromptDetails, "llm.token_count.prompt_details"},
		{LLMTokenCountPromptDetailsAudio, "llm.token_count.prompt_details.audio"},
		{LLMTokenCountPromptDetailsCacheInput, "llm.token_count.prompt_details.cache_input"},
		{LLMTokenCountPromptDetailsCacheRead, "llm.token_count.prompt_details.cache_read"},
		{LLMTokenCountPromptDetailsCacheWrite, "llm.token_count.prompt_details.cache_write"},
		{LLMTokenCountCompletion, "llm.token_count.completion"},
		{LLMTokenCountCompletionDetailsAudio, "llm.token_count.completion_details.audio"},
		{LLMTokenCountCompletionDetailsReasoning, "llm.token_count.completion_details.reasoning"},
		{LLMTokenCountTotal, "llm.token_count.total"},

		{LLMCostPrompt, "llm.cost.prompt"},
		{LLMCostPromptDetails, "llm.cost.prompt_details"},
		{LLMCostPromptDetailsAudio, "llm.cost.prompt_details.audio"},
		{LLMCostPromptDetailsCacheInput, "llm.cost.prompt_details.cache_input"},
		{LLMCostPromptDetailsCacheRead, "llm.cost.prompt_details.cache_read"},
		{LLMCostPromptDetailsCacheWrite, "llm.cost.prompt_details.cache_write"},
		{LLMCostPromptDetailsInput, "llm.cost.prompt_details.input"},
		{LLMCostCompletion, "llm.cost.completion"},
		{LLMCostCompletionDetails, "llm.cost.completion_details"},
		{LLMCostCompletionDetailsAudio, "llm.cost.completion_details.audio"},
		{LLMCostCompletionDetailsOutput, "llm.cost.completion_details.output"},
		{LLMCostCompletionDetailsReasoning, "llm.cost.completion_details.reasoning"},
		{LLMCostTotal, "llm.cost.total"},

		{EmbeddingEmbeddings, "embedding.embeddings"},
		{EmbeddingInvocationParameters, "embedding.invocation_parameters"},
		{EmbeddingModelName, "embedding.model_name"},
		{EmbeddingText, "embedding.text"},
		{EmbeddingVector, "embedding.vector"},

		{ToolName, "tool.name"},
		{ToolDescription, "tool.description"},
		{ToolParameters, "tool.parameters"},
		{ToolID, "tool.id"},
		{ToolJSONSchema, "tool.json_schema"},

		{RetrievalDocuments, "retrieval.documents"},

		{RerankerInputDocuments, "reranker.input_documents"},
		{RerankerOutputDocuments, "reranker.output_documents"},
		{RerankerQuery, "reranker.query"},
		{RerankerModelName, "reranker.model_name"},
		{RerankerTopK, "reranker.top_k"},

		{MessageRole, "message.role"},
		{MessageContent, "message.content"},
		{MessageContents, "message.contents"},
		{MessageName, "message.name"},
		{MessageToolCalls, "message.tool_calls"},
		{MessageFunctionCallName, "message.function_call_name"},
		{MessageFunctionCallArgumentsJSON, "message.function_call_arguments_json"},
		{MessageToolCallID, "message.tool_call_id"},

		{MessageContentType, "message_content.type"},
		{MessageContentText, "message_content.text"},
		{MessageContentImage, "message_content.image"},
		{MessageContentSignature, "message_content.signature"},
		{MessageContentData, "message_content.data"},
		{MessageContentEncriptedContent, "message_content.encripted_content"},

		{ImageURL, "image.url"},

		{AudioURL, "audio.url"},
		{AudioMimeType, "audio.mime_type"},
		{AudioTranscript, "audio.transcript"},

		{DocumentID, "document.id"},
		{DocumentScore, "document.score"},
		{DocumentContent, "document.content"},
		{DocumentMetadata, "document.metadata"},

		{ToolCallID, "tool_call.id"},
		{ToolCallFunctionName, "tool_call.function.name"},
		{ToolCallFunctionArgumentsJSON, "tool_call.function.arguments"},
		{ToolCallSignature, "tool_call.signature"},

		{PromptText, "prompt.text"},
		{CompletionText, "completion.text"},

		{PromptVendor, "prompt.vendor"},
		{PromptID, "prompt.id"},
		{PromptURL, "prompt.url"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("attribute key drift: got %q want %q", c.got, c.want)
		}
	}
}

func TestEnumValues(t *testing.T) {
	cases := []struct {
		got, want string
	}{
		{SpanKindLLM, "LLM"},
		{SpanKindChain, "CHAIN"},
		{SpanKindTool, "TOOL"},
		{SpanKindRetriever, "RETRIEVER"},
		{SpanKindEmbedding, "EMBEDDING"},
		{SpanKindAgent, "AGENT"},
		{SpanKindReranker, "RERANKER"},
		{SpanKindGuardrail, "GUARDRAIL"},
		{SpanKindEvaluator, "EVALUATOR"},
		{SpanKindPrompt, "PROMPT"},
		{SpanKindUnknown, "UNKNOWN"},

		{MimeTypeText, "text/plain"},
		{MimeTypeJSON, "application/json"},

		{LLMSystemOpenAI, "openai"},
		{LLMSystemAnthropic, "anthropic"},
		{LLMSystemCohere, "cohere"},
		{LLMSystemMistralAI, "mistralai"},
		{LLMSystemVertexAI, "vertexai"},

		{LLMProviderOpenAI, "openai"},
		{LLMProviderAnthropic, "anthropic"},
		{LLMProviderCohere, "cohere"},
		{LLMProviderMistralAI, "mistralai"},
		{LLMProviderGoogle, "google"},
		{LLMProviderAzure, "azure"},
		{LLMProviderAWS, "aws"},
		{LLMProviderXAI, "xai"},
		{LLMProviderDeepSeek, "deepseek"},
		{LLMProviderGroq, "groq"},
		{LLMProviderFireworks, "fireworks"},
		{LLMProviderMoonshot, "moonshot"},
		{LLMProviderCerebras, "cerebras"},
		{LLMProviderPerplexity, "perplexity"},
		{LLMProviderTogether, "together"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("enum drift: got %q want %q", c.got, c.want)
		}
	}
}

func TestIndexers(t *testing.T) {
	cases := []struct {
		name      string
		got, want string
	}{
		{"LLMInputMessageRoleKey", LLMInputMessageRoleKey(0), "llm.input_messages.0.message.role"},
		{"LLMInputMessageContentKey", LLMInputMessageContentKey(1), "llm.input_messages.1.message.content"},
		{"LLMInputMessageNameKey", LLMInputMessageNameKey(2), "llm.input_messages.2.message.name"},
		{"LLMInputMessageToolCallIDKey", LLMInputMessageToolCallIDKey(3), "llm.input_messages.3.message.tool_call_id"},
		{"LLMOutputMessageRoleKey", LLMOutputMessageRoleKey(0), "llm.output_messages.0.message.role"},
		{"LLMOutputMessageContentKey", LLMOutputMessageContentKey(1), "llm.output_messages.1.message.content"},
		{"LLMOutputMessageToolCallKey/name", LLMOutputMessageToolCallKey(0, 1, ToolCallFunctionName), "llm.output_messages.0.message.tool_calls.1.tool_call.function.name"},
		{"LLMInputMessageToolCallKey/args", LLMInputMessageToolCallKey(2, 0, ToolCallFunctionArgumentsJSON), "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments"},
		{"LLMPromptKey", LLMPromptKey(0), "llm.prompts.0.prompt.text"},
		{"LLMChoiceKey", LLMChoiceKey(7), "llm.choices.7.completion.text"},
		{"LLMToolKey", LLMToolKey(0), "llm.tools.0.tool.json_schema"},
		{"RetrievalDocumentKey/content", RetrievalDocumentKey(4, DocumentContent), "retrieval.documents.4.document.content"},
		{"EmbeddingKey/text", EmbeddingKey(0, EmbeddingText), "embedding.embeddings.0.embedding.text"},
		{"EmbeddingKey/vector", EmbeddingKey(0, EmbeddingVector), "embedding.embeddings.0.embedding.vector"},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s: got %q want %q", c.name, c.got, c.want)
		}
	}
}
