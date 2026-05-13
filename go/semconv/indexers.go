package semconv

import "strconv"

// Indexed-attribute helpers. OTel attributes are a flat key/value map, so
// arrays (messages, documents, tool calls, embeddings) are represented by
// indexed string keys, e.g. "llm.input_messages.0.message.role".
//
// These helpers build the keys so callers don't hand-format them.

// LLMInputMessageRoleKey returns the attribute key for the role of the
// i-th input message, e.g. "llm.input_messages.0.message.role".
func LLMInputMessageRoleKey(i int) string {
	return llmMessageKey(LLMInputMessages, i, MessageRole)
}

// LLMInputMessageContentKey returns the attribute key for the content of
// the i-th input message, e.g. "llm.input_messages.0.message.content".
func LLMInputMessageContentKey(i int) string {
	return llmMessageKey(LLMInputMessages, i, MessageContent)
}

// LLMInputMessageNameKey returns the attribute key for the name of the
// i-th input message, e.g. "llm.input_messages.0.message.name".
func LLMInputMessageNameKey(i int) string {
	return llmMessageKey(LLMInputMessages, i, MessageName)
}

// LLMInputMessageToolCallIDKey returns the attribute key for the
// tool_call_id of the i-th input message.
func LLMInputMessageToolCallIDKey(i int) string {
	return llmMessageKey(LLMInputMessages, i, MessageToolCallID)
}

// LLMOutputMessageRoleKey returns the attribute key for the role of the
// i-th output message, e.g. "llm.output_messages.0.message.role".
func LLMOutputMessageRoleKey(i int) string {
	return llmMessageKey(LLMOutputMessages, i, MessageRole)
}

// LLMOutputMessageContentKey returns the attribute key for the content of
// the i-th output message, e.g. "llm.output_messages.0.message.content".
func LLMOutputMessageContentKey(i int) string {
	return llmMessageKey(LLMOutputMessages, i, MessageContent)
}

// LLMOutputMessageToolCallKey returns the attribute key for the j-th tool
// call on the i-th output message, e.g.
// "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"
// when child == ToolCallFunctionName.
func LLMOutputMessageToolCallKey(i, j int, child string) string {
	return LLMOutputMessages + "." + strconv.Itoa(i) + "." + MessageToolCalls + "." + strconv.Itoa(j) + "." + child
}

// LLMInputMessageToolCallKey returns the attribute key for the j-th tool
// call on the i-th input message.
func LLMInputMessageToolCallKey(i, j int, child string) string {
	return LLMInputMessages + "." + strconv.Itoa(i) + "." + MessageToolCalls + "." + strconv.Itoa(j) + "." + child
}

// LLMPromptKey returns the attribute key for the text of the i-th prompt
// in a completions-API call, e.g. "llm.prompts.0.prompt.text".
func LLMPromptKey(i int) string {
	return LLMPrompts + "." + strconv.Itoa(i) + "." + PromptText
}

// LLMChoiceKey returns the attribute key for the text of the i-th choice
// in a completions-API response, e.g. "llm.choices.0.completion.text".
func LLMChoiceKey(i int) string {
	return LLMChoices + "." + strconv.Itoa(i) + "." + CompletionText
}

// LLMToolKey returns the attribute key for the JSON schema of the i-th
// tool advertised to the LLM, e.g. "llm.tools.0.tool.json_schema".
func LLMToolKey(i int) string {
	return LLMTools + "." + strconv.Itoa(i) + "." + ToolJSONSchema
}

// RetrievalDocumentKey returns the attribute key for the child of the
// i-th retrieved document, e.g.
// RetrievalDocumentKey(0, DocumentContent) == "retrieval.documents.0.document.content".
func RetrievalDocumentKey(i int, child string) string {
	return RetrievalDocuments + "." + strconv.Itoa(i) + "." + child
}

// EmbeddingKey returns the attribute key for the child of the i-th
// embedding, e.g.
// EmbeddingKey(0, EmbeddingText) == "embedding.embeddings.0.embedding.text".
func EmbeddingKey(i int, child string) string {
	return EmbeddingEmbeddings + "." + strconv.Itoa(i) + "." + child
}

// llmMessageKey is the internal builder for "{prefix}.{i}.{child}" keys
// where the child is one of the Message* constants.
func llmMessageKey(prefix string, i int, child string) string {
	return prefix + "." + strconv.Itoa(i) + "." + child
}
