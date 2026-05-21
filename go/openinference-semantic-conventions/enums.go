package semconv

// Values for the OpenInferenceSpanKind attribute. Pick the value that best
// describes the span: LLM for raw provider API calls, CHAIN for orchestration
// boundaries, TOOL for function/tool execution, RETRIEVER for vector-store
// lookups, EMBEDDING for embedding API calls, AGENT for an autonomous
// sub-agent run nested inside a larger chain, RERANKER for rerank API calls,
// GUARDRAIL for guardrail/policy checks, EVALUATOR for online eval calls,
// PROMPT for a prompt-registry lookup.
const (
	SpanKindLLM       = "LLM"
	SpanKindChain     = "CHAIN"
	SpanKindTool      = "TOOL"
	SpanKindRetriever = "RETRIEVER"
	SpanKindEmbedding = "EMBEDDING"
	SpanKindAgent     = "AGENT"
	SpanKindReranker  = "RERANKER"
	SpanKindGuardrail = "GUARDRAIL"
	SpanKindEvaluator = "EVALUATOR"
	SpanKindPrompt    = "PROMPT"
	SpanKindAudio     = "AUDIO"
	SpanKindUser      = "USER"
	SpanKindUnknown   = "UNKNOWN"
)

// Values for the InputMimeType / OutputMimeType attributes.
const (
	MimeTypeText = "text/plain"
	MimeTypeJSON = "application/json"
)

// Values for the LLMSystem attribute (the AI product as identified by the
// client or server).
const (
	LLMSystemOpenAI    = "openai"
	LLMSystemAnthropic = "anthropic"
	LLMSystemCohere    = "cohere"
	LLMSystemMistralAI = "mistralai"
	LLMSystemVertexAI  = "vertexai"
)

// Values for the LLMProvider attribute (the company providing the model —
// often the same as the system, but distinct for providers that resell
// models from elsewhere, e.g. Azure → OpenAI).
const (
	LLMProviderOpenAI     = "openai"
	LLMProviderAnthropic  = "anthropic"
	LLMProviderCohere     = "cohere"
	LLMProviderMistralAI  = "mistralai"
	LLMProviderGoogle     = "google"
	LLMProviderAzure      = "azure"
	LLMProviderAWS        = "aws"
	LLMProviderXAI        = "xai"
	LLMProviderDeepSeek   = "deepseek"
	LLMProviderGroq       = "groq"
	LLMProviderFireworks  = "fireworks"
	LLMProviderMoonshot   = "moonshot"
	LLMProviderCerebras   = "cerebras"
	LLMProviderPerplexity = "perplexity"
	LLMProviderTogether   = "together"
)
