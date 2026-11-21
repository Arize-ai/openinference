// Code generated from the semantic convention registry in semantic-conventions/model. DO NOT EDIT.
// Change the model or semantic-conventions/templates/registry/go, then run
// `make -C semantic-conventions generate`.

package semconv

import "strconv"

// Indexed-attribute helpers. OTel attributes are a flat key/value map, so
// lists of objects (messages, documents, tool calls, embeddings) are
// represented by indexed string keys, e.g. "llm.input_messages.0.message.role".
//
// These helpers build the keys so callers don't hand-format them.

// AnnotationKey returns the key for a field of the i-th item of annotations, e.g. "annotations.0.annotation.<field>" when child is one of the annotation.* constants.
func AnnotationKey(i int, child string) string {
	return indexedKey(Annotations, i, child)
}

// EmbeddingKey returns the key for a field of the i-th item of embedding.embeddings, e.g. "embedding.embeddings.0.embedding.<field>" when child is one of the embedding.* constants.
func EmbeddingKey(i int, child string) string {
	return indexedKey(EmbeddingEmbeddings, i, child)
}

// EvaluationKey returns the key for a field of the i-th item of evaluations, e.g. "evaluations.0.evaluation.<field>" when child is one of the evaluation.* constants.
func EvaluationKey(i int, child string) string {
	return indexedKey(Evaluations, i, child)
}

// InputDocumentKey returns the key for a field of the i-th item of input.documents, e.g. "input.documents.0.document.<field>" when child is one of the document.* constants.
func InputDocumentKey(i int, child string) string {
	return indexedKey(InputDocuments, i, child)
}

// InputImageKey returns the key for a field of the i-th item of input.images, e.g. "input.images.0.image.<field>" when child is one of the image.* constants.
func InputImageKey(i int, child string) string {
	return indexedKey(InputImages, i, child)
}

// LLMInputMessageKey returns the key for a field of the i-th item of llm.input_messages, e.g. "llm.input_messages.0.message.<field>" when child is one of the message.* constants.
func LLMInputMessageKey(i int, child string) string {
	return indexedKey(LLMInputMessages, i, child)
}

// LLMOutputMessageKey returns the key for a field of the i-th item of llm.output_messages, e.g. "llm.output_messages.0.message.<field>" when child is one of the message.* constants.
func LLMOutputMessageKey(i int, child string) string {
	return indexedKey(LLMOutputMessages, i, child)
}

// OutputDocumentKey returns the key for a field of the i-th item of output.documents, e.g. "output.documents.0.document.<field>" when child is one of the document.* constants.
func OutputDocumentKey(i int, child string) string {
	return indexedKey(OutputDocuments, i, child)
}

// OutputImageKey returns the key for a field of the i-th item of output.images, e.g. "output.images.0.image.<field>" when child is one of the image.* constants.
func OutputImageKey(i int, child string) string {
	return indexedKey(OutputImages, i, child)
}

// RerankerInputDocumentKey returns the key for a field of the i-th item of reranker.input_documents, e.g. "reranker.input_documents.0.document.<field>" when child is one of the document.* constants.
func RerankerInputDocumentKey(i int, child string) string {
	return indexedKey(RerankerInputDocuments, i, child)
}

// RerankerOutputDocumentKey returns the key for a field of the i-th item of reranker.output_documents, e.g. "reranker.output_documents.0.document.<field>" when child is one of the document.* constants.
func RerankerOutputDocumentKey(i int, child string) string {
	return indexedKey(RerankerOutputDocuments, i, child)
}

// RetrievalDocumentKey returns the key for a field of the i-th item of retrieval.documents, e.g. "retrieval.documents.0.document.<field>" when child is one of the document.* constants.
func RetrievalDocumentKey(i int, child string) string {
	return indexedKey(RetrievalDocuments, i, child)
}

// SessionAnnotationKey returns the key for a field of the i-th item of session.annotations, e.g. "session.annotations.0.annotation.<field>" when child is one of the annotation.* constants.
func SessionAnnotationKey(i int, child string) string {
	return indexedKey(SessionAnnotations, i, child)
}

// SessionEvaluationKey returns the key for a field of the i-th item of session.evaluations, e.g. "session.evaluations.0.evaluation.<field>" when child is one of the evaluation.* constants.
func SessionEvaluationKey(i int, child string) string {
	return indexedKey(SessionEvaluations, i, child)
}

// TraceAnnotationKey returns the key for a field of the i-th item of trace.annotations, e.g. "trace.annotations.0.annotation.<field>" when child is one of the annotation.* constants.
func TraceAnnotationKey(i int, child string) string {
	return indexedKey(TraceAnnotations, i, child)
}

// TraceEvaluationKey returns the key for a field of the i-th item of trace.evaluations, e.g. "trace.evaluations.0.evaluation.<field>" when child is one of the evaluation.* constants.
func TraceEvaluationKey(i int, child string) string {
	return indexedKey(TraceEvaluations, i, child)
}

// LLMInputMessageRoleKey returns the key for message.role of the i-th item of llm.input_messages, e.g. "llm.input_messages.0.message.role".
func LLMInputMessageRoleKey(i int) string {
	return indexedKey(LLMInputMessages, i, MessageRole)
}

// LLMInputMessageContentKey returns the key for message.content of the i-th item of llm.input_messages, e.g. "llm.input_messages.0.message.content".
func LLMInputMessageContentKey(i int) string {
	return indexedKey(LLMInputMessages, i, MessageContent)
}

// LLMInputMessageNameKey returns the key for message.name of the i-th item of llm.input_messages, e.g. "llm.input_messages.0.message.name".
func LLMInputMessageNameKey(i int) string {
	return indexedKey(LLMInputMessages, i, MessageName)
}

// LLMInputMessageToolCallIDKey returns the key for message.tool_call_id of the i-th item of llm.input_messages, e.g. "llm.input_messages.0.message.tool_call_id".
func LLMInputMessageToolCallIDKey(i int) string {
	return indexedKey(LLMInputMessages, i, MessageToolCallID)
}

// LLMOutputMessageRoleKey returns the key for message.role of the i-th item of llm.output_messages, e.g. "llm.output_messages.0.message.role".
func LLMOutputMessageRoleKey(i int) string {
	return indexedKey(LLMOutputMessages, i, MessageRole)
}

// LLMOutputMessageContentKey returns the key for message.content of the i-th item of llm.output_messages, e.g. "llm.output_messages.0.message.content".
func LLMOutputMessageContentKey(i int) string {
	return indexedKey(LLMOutputMessages, i, MessageContent)
}

// LLMPromptKey returns the key for prompt.text of the i-th item of llm.prompts, e.g. "llm.prompts.0.prompt.text".
func LLMPromptKey(i int) string {
	return indexedKey(LLMPrompts, i, PromptText)
}

// LLMChoiceKey returns the key for completion.text of the i-th item of llm.choices, e.g. "llm.choices.0.completion.text".
func LLMChoiceKey(i int) string {
	return indexedKey(LLMChoices, i, CompletionText)
}

// LLMToolKey returns the key for tool.json_schema of the i-th item of llm.tools, e.g. "llm.tools.0.tool.json_schema".
func LLMToolKey(i int) string {
	return indexedKey(LLMTools, i, ToolJSONSchema)
}

// LLMOutputMessageToolCallKey returns the key for a field of the j-th item of message.tool_calls on the i-th item of llm.output_messages, e.g. "llm.output_messages.0.message.tool_calls.0.<field>".
func LLMOutputMessageToolCallKey(i, j int, child string) string {
	return indexedKey(indexedKey(LLMOutputMessages, i, MessageToolCalls), j, child)
}

// LLMInputMessageToolCallKey returns the key for a field of the j-th item of message.tool_calls on the i-th item of llm.input_messages, e.g. "llm.input_messages.0.message.tool_calls.0.<field>".
func LLMInputMessageToolCallKey(i, j int, child string) string {
	return indexedKey(indexedKey(LLMInputMessages, i, MessageToolCalls), j, child)
}

// indexedKey builds "{prefix}.{i}.{child}".
func indexedKey(prefix string, i int, child string) string {
	return prefix + "." + strconv.Itoa(i) + "." + child
}
