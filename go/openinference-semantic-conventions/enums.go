package semconv

// Values for the OpenInferenceSpanKind attribute. Pick the value that best
// describes the span: LLM for raw provider API calls, CHAIN for orchestration
// boundaries, TOOL for function/tool execution, RETRIEVER for vector-store
// lookups, EMBEDDING for embedding API calls, AGENT for an autonomous
// sub-agent run nested inside a larger chain, RERANKER for rerank API calls,
// GUARDRAIL for guardrail/policy checks, EVALUATOR for online eval calls,
// PROMPT for a prompt-registry lookup, DECISION for a decision-model call that
// scores or selects among candidate options.
const (
	SpanKindLLM       = "LLM"
	SpanKindChain     = "CHAIN"
	SpanKindDecision  = "DECISION"
	SpanKindTool      = "TOOL"
	SpanKindRetriever = "RETRIEVER"
	SpanKindEmbedding = "EMBEDDING"
	SpanKindAgent     = "AGENT"
	SpanKindReranker  = "RERANKER"
	SpanKindGuardrail = "GUARDRAIL"
	SpanKindEvaluator = "EVALUATOR"
	SpanKindPrompt    = "PROMPT"
	SpanKindUnknown   = "UNKNOWN"
)

// Values for AnnotationAnnotatorKind and EvaluationAnnotatorKind.
const (
	AnnotatorKindHuman = "HUMAN"
	AnnotatorKindLLM   = "LLM"
	AnnotatorKindCode  = "CODE"
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
	LLMSystemTypeSafe  = "typesafe"
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
	LLMProviderOllama     = "ollama"
	LLMProviderMeta       = "meta"
	LLMProviderZAI        = "zai"
	LLMProviderMiniMax    = "minimax"
	LLMProviderOracle     = "oracle"
	LLMProviderTypeSafe   = "typesafe"
)

// Values for the DecisionSystem attribute: the decision API ecosystem a
// DECISION span conforms to. Each constant aliases the LLMSystem* constant
// for the same vendor, so the same string names the same vendor on LLM and
// DECISION spans. The list is the subset of vendors currently known to offer
// a decision API.
const (
	// DecisionSystemTypeSafe is the TypeSafe AI System One / Jev API,
	// including Jev-compatible servers.
	DecisionSystemTypeSafe = LLMSystemTypeSafe
	// DecisionSystemOpenAI is the OpenAI Decisions API.
	DecisionSystemOpenAI = LLMSystemOpenAI
)

// Values for the DecisionProvider attribute: who hosts the decision model
// that answered. Each constant aliases the LLMProvider* constant for the
// same vendor.
const (
	DecisionProviderTypeSafe = LLMProviderTypeSafe
	DecisionProviderOpenAI   = LLMProviderOpenAI
)
