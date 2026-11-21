// Code generated from the semantic convention registry in semantic-conventions/model. DO NOT EDIT.
// Change the model or semantic-conventions/templates/registry/go, then run
// `make -C semantic-conventions generate`.

package semconv

// Well-known values for the openinference.span.kind attribute.
const (
	// SpanKindLLM: A call to a large language model.
	SpanKindLLM = "LLM"

	// SpanKindChain: A link between application steps, or the start of a request.
	SpanKindChain = "CHAIN"

	// SpanKindTool: Execution of a tool or function.
	SpanKindTool = "TOOL"

	// SpanKindRetriever: A query to a vector store, search engine, or database.
	SpanKindRetriever = "RETRIEVER"

	// SpanKindReranker: Reordering of candidate documents by relevance.
	SpanKindReranker = "RERANKER"

	// SpanKindEmbedding: Generation of vector embeddings.
	SpanKindEmbedding = "EMBEDDING"

	// SpanKindAgent: A reasoning step that acts on tools using an LLM's guidance.
	SpanKindAgent = "AGENT"

	// SpanKindGuardrail: An input or output moderation check.
	SpanKindGuardrail = "GUARDRAIL"

	// SpanKindEvaluator: An evaluation of a model's output.
	SpanKindEvaluator = "EVALUATOR"

	// SpanKindPrompt: The rendering of a prompt template.
	SpanKindPrompt = "PROMPT"

	// SpanKindDecision: A call to a decision model.
	SpanKindDecision = "DECISION"

	// SpanKindUnknown: Fallback used by SDKs when the kind of an operation cannot be determined.
	SpanKindUnknown = "UNKNOWN"
)

// Well-known values for the annotation.annotator_kind attribute.
const (
	// AnnotatorKindHuman: A human reviewer.
	AnnotatorKindHuman = "HUMAN"

	// AnnotatorKindLLM: An LLM judge.
	AnnotatorKindLLM = "LLM"

	// AnnotatorKindCode: A code-based evaluator.
	AnnotatorKindCode = "CODE"
)

// Well-known values for the input.mime_type attribute.
const (
	// MimeTypeText: Plain text. The default when unspecified.
	MimeTypeText = "text/plain"

	// MimeTypeJSON: A string holding serialized JSON.
	MimeTypeJSON = "application/json"
)

// Well-known values for the llm.system attribute.
const (
	// LLMSystemOpenAI: OpenAI
	LLMSystemOpenAI = "openai"

	// LLMSystemAnthropic: Anthropic
	LLMSystemAnthropic = "anthropic"

	// LLMSystemMistralAI: Mistral AI
	LLMSystemMistralAI = "mistralai"

	// LLMSystemCohere: Cohere
	LLMSystemCohere = "cohere"

	// LLMSystemVertexAI: Vertex AI
	LLMSystemVertexAI = "vertexai"

	// LLMSystemDeepSeek: DeepSeek
	LLMSystemDeepSeek = "deepseek"

	// LLMSystemAmazon: Amazon Bedrock native
	LLMSystemAmazon = "amazon"

	// LLMSystemMeta: Meta (Llama)
	LLMSystemMeta = "meta"

	// LLMSystemAI21: AI21 Labs
	LLMSystemAI21 = "ai21"

	// LLMSystemTypeSafe: TypeSafe AI
	LLMSystemTypeSafe = "typesafe"

	// LLMSystemXAI: xAI (Grok)
	LLMSystemXAI = "xai"
)

// Well-known values for the llm.provider attribute.
const (
	// LLMProviderOpenAI: OpenAI
	LLMProviderOpenAI = "openai"

	// LLMProviderAnthropic: Anthropic
	LLMProviderAnthropic = "anthropic"

	// LLMProviderMistralAI: Mistral AI
	LLMProviderMistralAI = "mistralai"

	// LLMProviderCohere: Cohere
	LLMProviderCohere = "cohere"

	// LLMProviderGoogle: Google (Vertex)
	LLMProviderGoogle = "google"

	// LLMProviderAWS: AWS Bedrock
	LLMProviderAWS = "aws"

	// LLMProviderAzure: Azure
	LLMProviderAzure = "azure"

	// LLMProviderXAI: xAI
	LLMProviderXAI = "xai"

	// LLMProviderDeepSeek: DeepSeek
	LLMProviderDeepSeek = "deepseek"

	// LLMProviderGroq: Groq
	LLMProviderGroq = "groq"

	// LLMProviderFireworks: Fireworks AI
	LLMProviderFireworks = "fireworks"

	// LLMProviderMoonshot: Moonshot AI
	LLMProviderMoonshot = "moonshot"

	// LLMProviderCerebras: Cerebras
	LLMProviderCerebras = "cerebras"

	// LLMProviderPerplexity: Perplexity
	LLMProviderPerplexity = "perplexity"

	// LLMProviderTogether: Together AI
	LLMProviderTogether = "together"

	// LLMProviderOllama: Ollama
	LLMProviderOllama = "ollama"

	// LLMProviderMeta: Meta AI
	LLMProviderMeta = "meta"

	// LLMProviderZAI: Z.ai (GLM)
	LLMProviderZAI = "zai"

	// LLMProviderMiniMax: MiniMax
	LLMProviderMiniMax = "minimax"

	// LLMProviderOracle: Oracle (OCI Generative AI)
	LLMProviderOracle = "oracle"

	// LLMProviderIBM: IBM (watsonx.ai)
	LLMProviderIBM = "ibm"

	// LLMProviderTypeSafe: TypeSafe AI
	LLMProviderTypeSafe = "typesafe"

	// LLMProviderDoubleword: Doubleword
	LLMProviderDoubleword = "doubleword"
)

// Well-known values for the decision.system attribute.
const (
	// DecisionSystemTypeSafe: [TypeSafe AI System One / Jev API](https://typesafe.ai/blog/introducing-system-one-models-and-jev), including Jev-compatible servers such as vLLM's `/v1/systemone`
	DecisionSystemTypeSafe = "typesafe"

	// DecisionSystemOpenAI: [OpenAI Decisions API](https://openai.com/index/devday-2026-recap/)
	DecisionSystemOpenAI = "openai"
)

// Well-known values for the decision.provider attribute.
const (
	// DecisionProviderTypeSafe: TypeSafe AI
	DecisionProviderTypeSafe = "typesafe"

	// DecisionProviderOpenAI: OpenAI
	DecisionProviderOpenAI = "openai"
)
