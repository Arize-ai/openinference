// Generated from the semantic convention registry in semantic-conventions/model.
// Do not edit by hand: change the model or semantic-conventions/templates/registry/java,
// then run `make -C semantic-conventions generate`.
package com.arize.semconv.trace;

import lombok.experimental.UtilityClass;

public class SemanticConventions {
    @UtilityClass
    public static class SemanticAttributePrefixes {
        public static final String AGENT = "agent";
        public static final String ANNOTATION = "annotation";
        public static final String ANNOTATIONS = "annotations";
        public static final String AUDIO = "audio";
        public static final String COMPLETION = "completion";
        public static final String DECISION = "decision";
        public static final String DOCUMENT = "document";
        public static final String EMBEDDING = "embedding";
        public static final String EVALUATION = "evaluation";
        public static final String EVALUATIONS = "evaluations";
        public static final String GRAPH = "graph";
        public static final String IMAGE = "image";
        public static final String INPUT = "input";
        public static final String LLM = "llm";
        public static final String MESSAGE = "message";
        public static final String MESSAGE_CONTENT = "message_content";
        public static final String MESSAGES = "messages";
        public static final String METADATA = "metadata";
        public static final String OPENINFERENCE = "openinference";
        public static final String OUTPUT = "output";
        public static final String PROMPT = "prompt";
        public static final String RERANKER = "reranker";
        public static final String RETRIEVAL = "retrieval";
        public static final String SESSION = "session";
        public static final String TAG = "tag";
        public static final String TOOL = "tool";
        public static final String TOOL_CALL = "tool_call";
        public static final String TRACE = "trace";
        public static final String USER = "user";
        public static final String VIDEO = "video";
    }

    @UtilityClass
    public static class LLMAttributePostfixes {
        public static final String CHOICES = "choices";
        public static final String COST = "cost";
        public static final String FINISH_REASON = "finish_reason";
        public static final String FUNCTION_CALL = "function_call";
        public static final String INPUT_MESSAGES = "input_messages";
        public static final String INVOCATION_PARAMETERS = "invocation_parameters";
        public static final String MODEL_NAME = "model_name";
        public static final String OUTPUT_MESSAGES = "output_messages";
        public static final String PROMPT_TEMPLATE = "prompt_template";
        public static final String PROMPTS = "prompts";
        public static final String PROVIDER = "provider";
        public static final String REQUEST = "request";
        public static final String RESPONSE = "response";
        public static final String SYSTEM = "system";
        public static final String TOKEN_COUNT = "token_count";
        public static final String TOOLS = "tools";
    }

    @UtilityClass
    public static class DecisionAttributePostfixes {
        public static final String MODEL_NAME = "model_name";
        public static final String PROVIDER = "provider";
        public static final String REQUEST = "request";
        public static final String RESPONSE = "response";
        public static final String SYSTEM = "system";
        public static final String TOKEN_COUNT = "token_count";
    }

    @UtilityClass
    public static class LLMPromptTemplateAttributePostfixes {
        public static final String TEMPLATE = "template";
        public static final String VARIABLES = "variables";
        public static final String VERSION = "version";
    }

    @UtilityClass
    public static class RetrievalAttributePostfixes {
        public static final String DOCUMENTS = "documents";
    }

    @UtilityClass
    public static class RerankerAttributePostfixes {
        public static final String INPUT_DOCUMENTS = "input_documents";
        public static final String MODEL_NAME = "model_name";
        public static final String OUTPUT_DOCUMENTS = "output_documents";
        public static final String QUERY = "query";
        public static final String TOP_K = "top_k";
    }

    @UtilityClass
    public static class EmbeddingAttributePostfixes {
        public static final String EMBEDDINGS = "embeddings";
        public static final String INVOCATION_PARAMETERS = "invocation_parameters";
        public static final String MODEL_NAME = "model_name";
        public static final String TEXT = "text";
        public static final String VECTOR = "vector";
    }

    @UtilityClass
    public static class ToolAttributePostfixes {
        public static final String DESCRIPTION = "description";
        public static final String ID = "id";
        public static final String JSON_SCHEMA = "json_schema";
        public static final String NAME = "name";
        public static final String PARAMETERS = "parameters";
    }

    @UtilityClass
    public static class MessageAttributePostfixes {
        public static final String CONTENT = "content";
        public static final String CONTENTS = "contents";
        public static final String FUNCTION_CALL_ARGUMENTS_JSON = "function_call_arguments_json";
        public static final String FUNCTION_CALL_NAME = "function_call_name";
        public static final String NAME = "name";
        public static final String ROLE = "role";
        public static final String TOOL_CALL_ID = "tool_call_id";
        public static final String TOOL_CALLS = "tool_calls";
    }

    @UtilityClass
    public static class MessageContentsAttributePostfixes {
        public static final String AUDIO = "audio";
        public static final String DATA = "data";
        public static final String DOCUMENT = "document";
        public static final String ENCRYPTED_CONTENT = "encrypted_content";
        public static final String ID = "id";
        public static final String IMAGE = "image";
        public static final String SIGNATURE = "signature";
        public static final String TEXT = "text";
        public static final String TYPE = "type";
        public static final String VIDEO = "video";
    }

    @UtilityClass
    public static class ImageAttributesPostfixes {
        public static final String URL = "url";
    }

    @UtilityClass
    public static class VideoAttributesPostfixes {
        public static final String URL = "url";
    }

    @UtilityClass
    public static class ToolCallAttributePostfixes {
        public static final String FUNCTION_ARGUMENTS = "function.arguments";
        public static final String FUNCTION_NAME = "function.name";
        public static final String ID = "id";
        public static final String REASONING_SIGNATURE = "reasoning_signature";
    }

    @UtilityClass
    public static class DocumentAttributePostfixes {
        public static final String CONTENT = "content";
        public static final String ID = "id";
        public static final String METADATA = "metadata";
        public static final String MIME_TYPE = "mime_type";
        public static final String NAME = "name";
        public static final String SCORE = "score";
        public static final String URL = "url";
    }

    @UtilityClass
    public static class TagAttributePostfixes {
        public static final String TAGS = "tags";
    }

    @UtilityClass
    public static class SessionAttributePostfixes {
        public static final String ANNOTATIONS = "annotations";
        public static final String EVALUATIONS = "evaluations";
        public static final String ID = "id";
    }

    @UtilityClass
    public static class UserAttributePostfixes {
        public static final String ID = "id";
    }

    @UtilityClass
    public static class AudioAttributesPostfixes {
        public static final String MIME_TYPE = "mime_type";
        public static final String TRANSCRIPT = "transcript";
        public static final String URL = "url";
    }

    @UtilityClass
    public static class PromptAttributePostfixes {
        public static final String ID = "id";
        public static final String TEXT = "text";
        public static final String URL = "url";
        public static final String VENDOR = "vendor";
    }

    @UtilityClass
    public static class AgentPostfixes {
        public static final String NAME = "name";
    }

    @UtilityClass
    public static class GraphPostfixes {
        public static final String NODE_ID = "node.id";
        public static final String NODE_NAME = "node.name";
        public static final String NODE_PARENT_ID = "node.parent_id";
    }

    @UtilityClass
    public static class FeedbackAttributePostfixes {
        public static final String ANNOTATOR_KIND = "annotator_kind";
        public static final String EXPLANATION = "explanation";
        public static final String IDENTIFIER = "identifier";
        public static final String LABEL = "label";
        public static final String METADATA = "metadata";
        public static final String NAME = "name";
        public static final String SCORE = "score";
    }

    /**
     * The name of the agent that this span represents.
     */
    public static final String AGENT_NAME = "agent.name";

    /**
     * Judge kind: `HUMAN`, `LLM`, `CODE`, or custom. See [Annotator Kinds](https://github.com/Arize-ai/openinference/blob/main/spec/feedback/annotations.md#annotator-kinds)
     */
    public static final String ANNOTATION_ANNOTATOR_KIND = "annotation.annotator_kind";

    /**
     * Reason or evidence for the result
     */
    public static final String ANNOTATION_EXPLANATION = "annotation.explanation";

    /**
     * Stable result ID for the same name and target
     */
    public static final String ANNOTATION_IDENTIFIER = "annotation.identifier";

    /**
     * Categorical result
     */
    public static final String ANNOTATION_LABEL = "annotation.label";

    /**
     * Additional result or judge data
     */
    public static final String ANNOTATION_METADATA = "annotation.metadata";

    /**
     * Criterion or metric name
     */
    public static final String ANNOTATION_NAME = "annotation.name";

    /**
     * Numeric result.
     */
    public static final String ANNOTATION_SCORE = "annotation.score";

    /**
     * Span-scoped feedback using annotation terminology, flattened as `annotations.0.annotation.*`
     */
    public static final String ANNOTATIONS = "annotations";

    /**
     * The MIME type of the audio file.
     */
    public static final String AUDIO_MIME_TYPE = "audio.mime_type";

    /**
     * The transcript of the audio file (e.g. a Whisper transcription)
     */
    public static final String AUDIO_TRANSCRIPT = "audio.transcript";

    /**
     * The URL of an audio file (e.g. cloud storage), or a base64 data URI
     */
    public static final String AUDIO_URL = "audio.url";

    /**
     * The text of a choice returned from a completions API. Used under `llm.choices`.
     */
    public static final String COMPLETION_TEXT = "completion.text";

    /**
     * The name of the decision model being utilized. Mirrors `llm.model_name` for `DECISION` spans.
     */
    public static final String DECISION_MODEL_NAME = "decision.model_name";

    /**
     * The hosting provider of the decision model: who runs the hardware that answered, e.g. `typesafe` when calling TypeSafe directly, `openai` for the OpenAI Decisions API, or the cloud or self-hosting provider for a model served elsewhere. Distinct from `decision.system`, which names the API shape. The well-known values alias the matching `llm.provider` values.
     */
    public static final String DECISION_PROVIDER = "decision.provider";

    /**
     * The decision model requested by the caller, as sent in the request. May differ from `decision.response.model_name` when the provider resolves an alias or routes the request.
     */
    public static final String DECISION_REQUEST_MODEL_NAME = "decision.request.model_name";

    /**
     * The decision model that actually produced the response, as reported by the provider. May differ from `decision.request.model_name`.
     */
    public static final String DECISION_RESPONSE_MODEL_NAME = "decision.response.model_name";

    /**
     * The decision API ecosystem the call conforms to, i.e. which request/response shape the client speaks, as identified by the client or server instrumentation: `typesafe` for the TypeSafe System One / Jev API, `openai` for the OpenAI Decisions API. A self-hosted vLLM server answering the Jev-compatible `/v1/systemone` shape is still `typesafe`. Distinct from `decision.provider`, which says who hosts the model. The well-known values alias the matching `llm.system` values.
     */
    public static final String DECISION_SYSTEM = "decision.system";

    /**
     * The number of input tokens consumed by a decision model call: the state, questions, and candidate options. Maps to `usage.input_tokens` in TypeSafe System One responses.
     */
    public static final String DECISION_TOKEN_COUNT_INPUT = "decision.token_count.input";

    /**
     * The number of output tokens produced by a decision model call. Decision models emit typed answers rather than text, so this is typically small. Maps to `usage.output_tokens` in TypeSafe System One responses.
     */
    public static final String DECISION_TOKEN_COUNT_OUTPUT = "decision.token_count.output";

    /**
     * The text content of a document: a retrieved passage, or an inline text document passed to a model (e.g. an Anthropic `text` source)
     */
    public static final String DOCUMENT_CONTENT = "document.content";

    /**
     * Unique identifier for a document: a retrieved document or chunk in a retrieval system, or the provider file id (e.g. an OpenAI or Anthropic Files API id) for a document passed to a model.
     */
    public static final String DOCUMENT_ID = "document.id";

    /**
     * Metadata associated with a document
     */
    public static final String DOCUMENT_METADATA = "document.metadata";

    /**
     * The MIME type of a file document such as a PDF
     */
    public static final String DOCUMENT_MIME_TYPE = "document.mime_type";

    /**
     * The name supplied with a document, copied verbatim: a file name (OpenAI `filename`) or a title (Anthropic `title`, Bedrock `name`, Gemini `display_name`); not a unique identifier
     */
    public static final String DOCUMENT_NAME = "document.name";

    /**
     * Score representing the relevance of a document
     */
    public static final String DOCUMENT_SCORE = "document.score";

    /**
     * The URL, object-store URI, or base64 data URI of a file document such as a PDF
     */
    public static final String DOCUMENT_URL = "document.url";

    /**
     * List of embedding objects including text and vector data
     */
    public static final String EMBEDDING_EMBEDDINGS = "embedding.embeddings";

    /**
     * Parameters used during the invocation of an embedding model or API (excluding input)
     */
    public static final String EMBEDDING_INVOCATION_PARAMETERS = "embedding.invocation_parameters";

    /**
     * Name of the embedding model used
     */
    public static final String EMBEDDING_MODEL_NAME = "embedding.model_name";

    /**
     * The text represented in the embedding
     */
    public static final String EMBEDDING_TEXT = "embedding.text";

    /**
     * The embedding vector consisting of a list of floats
     */
    public static final String EMBEDDING_VECTOR = "embedding.vector";

    /**
     * Judge kind: `HUMAN`, `LLM`, `CODE`, or custom. See [Annotator Kinds](https://github.com/Arize-ai/openinference/blob/main/spec/feedback/annotations.md#annotator-kinds)
     */
    public static final String EVALUATION_ANNOTATOR_KIND = "evaluation.annotator_kind";

    /**
     * Reason or evidence for the result
     */
    public static final String EVALUATION_EXPLANATION = "evaluation.explanation";

    /**
     * Stable result ID for the same name and target
     */
    public static final String EVALUATION_IDENTIFIER = "evaluation.identifier";

    /**
     * Categorical result
     */
    public static final String EVALUATION_LABEL = "evaluation.label";

    /**
     * Additional result or judge data
     */
    public static final String EVALUATION_METADATA = "evaluation.metadata";

    /**
     * Criterion or metric name
     */
    public static final String EVALUATION_NAME = "evaluation.name";

    /**
     * Numeric result.
     */
    public static final String EVALUATION_SCORE = "evaluation.score";

    /**
     * Span-scoped feedback using evaluation or eval terminology, flattened as `evaluations.0.evaluation.*`
     */
    public static final String EVALUATIONS = "evaluations";

    /**
     * The id of the node in the execution graph. This along with `graph.node.parent_id` are used to visualize the execution graph.
     */
    public static final String GRAPH_NODE_ID = "graph.node.id";

    /**
     * The name of the node in the execution graph. Use this to present a human readable name for the node. Optional
     */
    public static final String GRAPH_NODE_NAME = "graph.node.name";

    /**
     * This references the id of the parent node. Leaving this unset or set as empty string implies that the current span is the root node.
     */
    public static final String GRAPH_NODE_PARENT_ID = "graph.node.parent_id";

    /**
     * The link to the image or its base64 encoding
     */
    public static final String IMAGE_URL = "image.url";

    /**
     * File document (e.g. PDF) inputs to an operation, independent of span kind, e.g. `input.documents.0.document.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-documents)
     */
    public static final String INPUT_DOCUMENTS = "input.documents";

    /**
     * Image inputs to an operation, independent of span kind, e.g. `input.images.0.image.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-images)
     */
    public static final String INPUT_IMAGES = "input.images";

    /**
     * MIME type representing the format of `input.value`
     */
    public static final String INPUT_MIME_TYPE = "input.mime_type";

    /**
     * The input value to an operation
     */
    public static final String INPUT_VALUE = "input.value";

    /**
     * Text choices returned from a completions API, e.g. `llm.choices.0.completion.text`
     */
    public static final String LLM_CHOICES = "llm.choices";

    /**
     * Total cost of all output tokens generated by the LLM in USD
     */
    public static final String LLM_COST_COMPLETION = "llm.cost.completion";

    /**
     * Cost of audio tokens in the completion in USD
     */
    public static final String LLM_COST_COMPLETION_DETAILS_AUDIO = "llm.cost.completion_details.audio";

    /**
     * Total cost of output tokens in USD
     */
    public static final String LLM_COST_OUTPUT = "llm.cost.completion_details.output";

    /**
     * Cost of reasoning steps in the completion in USD
     */
    public static final String LLM_COST_COMPLETION_DETAILS_REASONING = "llm.cost.completion_details.reasoning";

    /**
     * Total cost of all input tokens sent to the LLM in USD
     */
    public static final String LLM_COST_PROMPT = "llm.cost.prompt";

    /**
     * Cost of audio tokens in the prompt in USD
     */
    public static final String LLM_COST_PROMPT_DETAILS_AUDIO = "llm.cost.prompt_details.audio";

    /**
     * Cost of input tokens in the prompt that were cached in USD
     */
    public static final String LLM_COST_PROMPT_DETAILS_CACHE_INPUT = "llm.cost.prompt_details.cache_input";

    /**
     * Cost of prompt tokens read from cache in USD
     */
    public static final String LLM_COST_PROMPT_DETAILS_CACHE_READ = "llm.cost.prompt_details.cache_read";

    /**
     * Cost of prompt tokens written to cache in USD
     */
    public static final String LLM_COST_PROMPT_DETAILS_CACHE_WRITE = "llm.cost.prompt_details.cache_write";

    /**
     * Total cost of input tokens in USD
     */
    public static final String LLM_COST_INPUT = "llm.cost.prompt_details.input";

    /**
     * Total cost of the LLM call in USD (prompt + completion)
     */
    public static final String LLM_COST_TOTAL = "llm.cost.total";

    /**
     * The reason the model stopped generating tokens.
     */
    public static final String LLM_FINISH_REASON = "llm.finish_reason";

    /**
     * Object recording details of a function call in models or APIs
     *
     * @deprecated Use `message.tool_calls` instead.
     */
    @Deprecated
    public static final String LLM_FUNCTION_CALL = "llm.function_call";

    /**
     * List of messages sent to the LLM in a chat API request, e.g. `llm.input_messages.0.message.role`. See [Messages](https://github.com/Arize-ai/openinference/blob/main/spec/content/messages.md)
     */
    public static final String LLM_INPUT_MESSAGES = "llm.input_messages";

    /**
     * Parameters used during the invocation of an LLM or API
     */
    public static final String LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters";

    /**
     * The name of the language model being utilized
     */
    public static final String LLM_MODEL_NAME = "llm.model_name";

    /**
     * List of messages received from the LLM in a chat API response, e.g. `llm.output_messages.0.message.role`. See [Messages](https://github.com/Arize-ai/openinference/blob/main/spec/content/messages.md)
     */
    public static final String LLM_OUTPUT_MESSAGES = "llm.output_messages";

    /**
     * Template used to generate prompts (e.g., using `{variable}` placeholder syntax)
     */
    public static final String PROMPT_TEMPLATE_TEMPLATE = "llm.prompt_template.template";

    /**
     * JSON of key value pairs applied to the prompt template
     */
    public static final String PROMPT_TEMPLATE_VARIABLES = "llm.prompt_template.variables";

    /**
     * The version of the prompt template
     */
    public static final String PROMPT_TEMPLATE_VERSION = "llm.prompt_template.version";

    /**
     * Prompts provided to a completions API, e.g. `llm.prompts.0.prompt.text`
     */
    public static final String LLM_PROMPTS = "llm.prompts";

    /**
     * The hosting provider of the LLM, e.g. `azure`. See [well-known values](https://github.com/Arize-ai/openinference/blob/main/spec/attributes/semantic-conventions.md#llm-provider-values).
     */
    public static final String LLM_PROVIDER = "llm.provider";

    /**
     * The model requested by the caller, as sent in the request. May differ from `llm.response.model_name` when the provider routes the request to a different model.
     */
    public static final String LLM_REQUEST_MODEL_NAME = "llm.request.model_name";

    /**
     * The model that actually generated the response, as reported by the provider. May differ from `llm.request.model_name` when the provider routes the request to a different model.
     */
    public static final String LLM_RESPONSE_MODEL_NAME = "llm.response.model_name";

    /**
     * The AI product as identified by the client or server instrumentation. See [well-known values](https://github.com/Arize-ai/openinference/blob/main/spec/attributes/semantic-conventions.md#llm-system-values).
     */
    public static final String LLM_SYSTEM = "llm.system";

    /**
     * The number of tokens in the completion
     */
    public static final String LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion";

    /**
     * The number of audio tokens generated by the model
     */
    public static final String LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO = "llm.token_count.completion_details.audio";

    /**
     * The number of tokens used for model reasoning
     */
    public static final String LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING =
            "llm.token_count.completion_details.reasoning";

    /**
     * The number of tokens in the prompt
     */
    public static final String LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt";

    /**
     * The number of audio input tokens presented in the prompt
     */
    public static final String LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO = "llm.token_count.prompt_details.audio";

    /**
     * The number of input tokens in the prompt that were cached
     */
    public static final String LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_INPUT =
            "llm.token_count.prompt_details.cache_input";

    /**
     * The number of prompt tokens successfully retrieved from cache (cache hits). Maps to `cached_tokens` in OpenAI responses
     */
    public static final String LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = "llm.token_count.prompt_details.cache_read";

    /**
     * The number of prompt tokens not found in cache that were written to cache (cache misses). Maps to `cache_write_tokens` in OpenAI responses
     */
    public static final String LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE =
            "llm.token_count.prompt_details.cache_write";

    /**
     * Total number of tokens, including prompt and completion
     */
    public static final String LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total";

    /**
     * List of tools that are advertised to the LLM to be able to call, e.g. `llm.tools.0.tool.name`, `llm.tools.0.tool.description`, `llm.tools.0.tool.json_schema`. See [Tool Calling](https://github.com/Arize-ai/openinference/blob/main/spec/content/tool-calling.md)
     */
    public static final String LLM_TOOLS = "llm.tools";

    /**
     * The content of a message in a chat
     */
    public static final String MESSAGE_CONTENT = "message.content";

    /**
     * The message contents to the LLM, an array of `message_content` objects. Use this form when a message has multiple content items or when item ordering matters.
     */
    public static final String MESSAGE_CONTENTS = "message.contents";

    /**
     * The arguments to the function call in JSON
     *
     * @deprecated Use `message.tool_calls` instead.
     */
    @Deprecated
    public static final String MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = "message.function_call_arguments_json";

    /**
     * Function call function name
     *
     * @deprecated Use `message.tool_calls` instead.
     */
    @Deprecated
    public static final String MESSAGE_FUNCTION_CALL_NAME = "message.function_call_name";

    /**
     * The name of the function or tool that produced a tool/function role message. Used when `message.role` is `"tool"` or `"function"` to identify the origin.
     */
    public static final String MESSAGE_NAME = "message.name";

    /**
     * Role of the entity in a message (e.g., user, system)
     */
    public static final String MESSAGE_ROLE = "message.role";

    /**
     * Tool call result identifier corresponding to `tool_call.id`
     */
    public static final String MESSAGE_TOOL_CALL_ID = "message.tool_call_id";

    /**
     * List of tool calls (e.g. function calls) generated by the LLM, e.g. `llm.output_messages.0.message.tool_calls.0.tool_call.id`
     */
    public static final String MESSAGE_TOOL_CALLS = "message.tool_calls";

    /**
     * The audio content of the message, if the type is `"audio"`. Nested leaves are `audio.url` and `audio.transcript`.
     */
    public static final String MESSAGE_CONTENT_AUDIO = "message_content.audio";

    /**
     * Opaque vendor-issued data captured verbatim. Maps to Anthropic `redacted_thinking.data`.
     */
    public static final String MESSAGE_CONTENT_DATA = "message_content.data";

    /**
     * The document content of the message (e.g. a PDF, an uploaded file, or inline text), if the type is `"document"`. Nested leaves are `document.url`, `document.content`, `document.mime_type`, `document.id`, and `document.name`.
     */
    public static final String MESSAGE_CONTENT_DOCUMENT = "message_content.document";

    /**
     * OpenAI `encrypted_content` captured verbatim.
     */
    public static final String MESSAGE_CONTENT_ENCRYPTED_CONTENT = "message_content.encrypted_content";

    /**
     * Provider-assigned identifier for this message content item. For OpenAI Responses reasoning items, this maps to `ResponseReasoningItem.id` and should be preserved for stateless replay.
     */
    public static final String MESSAGE_CONTENT_ID = "message_content.id";

    /**
     * The image content of the message, if the type is `"image"`.
     */
    public static final String MESSAGE_CONTENT_IMAGE = "message_content.image";

    /**
     * Opaque vendor-issued signature captured verbatim. Maps to provider `signature` fields, and to Gemini `thoughtSignature` fields when the signature is attached to a non-tool content part such as text.
     */
    public static final String MESSAGE_CONTENT_SIGNATURE = "message_content.signature";

    /**
     * The text content of the message. Used when `message_content.type` is `"text"` or `"reasoning"` (visible reasoning or summary text). For OpenAI `summary_text` arrays, concatenate the array items into this single text value for now.
     */
    public static final String MESSAGE_CONTENT_TEXT = "message_content.text";

    /**
     * The type of the content. Use `"reasoning"` for reasoning/thinking content, including Anthropic `redacted_thinking` blocks. Use `"tool_use"` when a provider returns a tool call as an ordered content part.
     */
    public static final String MESSAGE_CONTENT_TYPE = "message_content.type";

    /**
     * The video content of the message, if the type is `"video"`. The nested leaf is `video.url`. Consumers infer MIME type from the URL path extension or a data URI prefix.
     */
    public static final String MESSAGE_CONTENT_VIDEO = "message_content.video";

    /**
     * Metadata associated with a span
     */
    public static final String METADATA = "metadata";

    /**
     * The role an operation plays in an AI application.
     */
    public static final String OPENINFERENCE_SPAN_KIND = "openinference.span.kind";

    /**
     * File document (e.g. PDF) outputs of an operation, independent of span kind, e.g. `output.documents.0.document.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-documents)
     */
    public static final String OUTPUT_DOCUMENTS = "output.documents";

    /**
     * Image outputs of an operation, independent of span kind, e.g. `output.images.0.image.url`. See [Multimodal Content](https://github.com/Arize-ai/openinference/blob/main/spec/content/multimodal.md#span-kind-independent-images)
     */
    public static final String OUTPUT_IMAGES = "output.images";

    /**
     * MIME type representing the format of `output.value`
     */
    public static final String OUTPUT_MIME_TYPE = "output.mime_type";

    /**
     * The output value of an operation
     */
    public static final String OUTPUT_VALUE = "output.value";

    /**
     * A vendor-specific id used to identify the prompt.
     */
    public static final String PROMPT_ID = "prompt.id";

    /**
     * The text of a prompt in a completions API request. Used under `llm.prompts`.
     */
    public static final String PROMPT_TEXT = "prompt.text";

    /**
     * A vendor-specific URL used to locate the prompt via the web
     */
    public static final String PROMPT_URL = "prompt.url";

    /**
     * The vendor or origin of the prompt, e.g. `langsmith`, `portkey`, `arize-phoenix`
     */
    public static final String PROMPT_VENDOR = "prompt.vendor";

    /**
     * List of documents as input to the reranker
     */
    public static final String RERANKER_INPUT_DOCUMENTS = "reranker.input_documents";

    /**
     * Model name of the reranker
     */
    public static final String RERANKER_MODEL_NAME = "reranker.model_name";

    /**
     * List of documents outputted by the reranker
     */
    public static final String RERANKER_OUTPUT_DOCUMENTS = "reranker.output_documents";

    /**
     * Query parameter of the reranker
     */
    public static final String RERANKER_QUERY = "reranker.query";

    /**
     * Top K parameter of the reranker
     */
    public static final String RERANKER_TOP_K = "reranker.top_k";

    /**
     * List of retrieved documents
     */
    public static final String RETRIEVAL_DOCUMENTS = "retrieval.documents";

    /**
     * Session-scoped feedback using annotation terminology, flattened as `session.annotations.0.annotation.*`; requires `session.id`
     */
    public static final String SESSION_ANNOTATIONS = "session.annotations";

    /**
     * Session-scoped feedback using evaluation terminology, flattened as `session.evaluations.0.evaluation.*`; requires `session.id`
     */
    public static final String SESSION_EVALUATIONS = "session.evaluations";

    /**
     * Unique identifier for a session, such as a multi-turn conversation.
     */
    public static final String SESSION_ID = "session.id";

    /**
     * List of tags to give the span a category
     */
    public static final String TAG_TAGS = "tag.tags";

    /**
     * Description of the tool's purpose and functionality. On a `TOOL` span, the tool being invoked; under `llm.tools.&lt;index&gt;`, an advertised tool definition
     */
    public static final String TOOL_DESCRIPTION = "tool.description";

    /**
     * The identifier for the result of the tool call (corresponding to `tool_call.id`)
     */
    public static final String TOOL_ID = "tool.id";

    /**
     * The JSON schema of a tool input
     */
    public static final String TOOL_JSON_SCHEMA = "tool.json_schema";

    /**
     * The name of the tool. On a `TOOL` span, the tool being invoked; under `llm.tools.&lt;index&gt;`, the name of an advertised tool definition
     */
    public static final String TOOL_NAME = "tool.name";

    /**
     * The parameters definition for invoking the tool
     */
    public static final String TOOL_PARAMETERS = "tool.parameters";

    /**
     * The arguments for the function being invoked by a tool call
     */
    public static final String TOOL_CALL_FUNCTION_ARGUMENTS_JSON = "tool_call.function.arguments";

    /**
     * The name of the function being invoked by a tool call
     */
    public static final String TOOL_CALL_FUNCTION_NAME = "tool_call.function.name";

    /**
     * The id of a tool call (useful when there is more than one call at the same time)
     */
    public static final String TOOL_CALL_ID = "tool_call.id";

    /**
     * Opaque vendor-issued reasoning echo token attached to a tool call. Used by Gemini, which puts `thoughtSignature` on `functionCall` parts rather than on the reasoning summary.
     */
    public static final String TOOL_CALL_REASONING_SIGNATURE = "tool_call.reasoning_signature";

    /**
     * Trace-scoped feedback using annotation terminology, flattened as `trace.annotations.0.annotation.*`
     */
    public static final String TRACE_ANNOTATIONS = "trace.annotations";

    /**
     * Trace-scoped feedback using evaluation terminology, flattened as `trace.evaluations.0.evaluation.*`
     */
    public static final String TRACE_EVALUATIONS = "trace.evaluations";

    /**
     * Unique identifier for a user.
     */
    public static final String USER_ID = "user.id";

    /**
     * The URL, object-store URI, or base64 data URI of a video. Consumers infer MIME type from the path extension (`.mp4` to `video/mp4`, `.webm` to `video/webm`, `.ogv` to `video/ogg`, `.avi` to `video/x-msvideo`) or from a data URI prefix.
     */
    public static final String VIDEO_URL = "video.url";

    /**
     * Key prefix for additional prompt token count details. Each detail is a separate attribute with this prefix, e.g. llm.token_count.prompt_details.audio. All values are in tokens.
     */
    public static final String LLM_TOKEN_COUNT_PROMPT_DETAILS = "llm.token_count.prompt_details";

    /**
     * Key prefix for additional completion token count details. Each detail is a separate attribute with this prefix, e.g. llm.token_count.completion_details.reasoning. All values are in tokens.
     */
    public static final String LLM_TOKEN_COUNT_COMPLETION_DETAILS = "llm.token_count.completion_details";

    /**
     * Key prefix for cost information. All values are in USD.
     */
    public static final String LLM_COST = "llm.cost";

    /**
     * Well-known values for {@code openinference.span.kind}.
     */
    public enum OpenInferenceSpanKind {
        LLM("LLM"),
        CHAIN("CHAIN"),
        TOOL("TOOL"),
        RETRIEVER("RETRIEVER"),
        RERANKER("RERANKER"),
        EMBEDDING("EMBEDDING"),
        AGENT("AGENT"),
        GUARDRAIL("GUARDRAIL"),
        EVALUATOR("EVALUATOR"),
        PROMPT("PROMPT"),
        DECISION("DECISION"),
        UNKNOWN("UNKNOWN");

        private final String value;

        OpenInferenceSpanKind(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        @Override
        public String toString() {
            return value;
        }
    }

    /**
     * Well-known values for {@code annotation.annotator_kind}.
     */
    public enum AnnotatorKind {
        HUMAN("HUMAN"),
        LLM("LLM"),
        CODE("CODE");

        private final String value;

        AnnotatorKind(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        @Override
        public String toString() {
            return value;
        }
    }

    /**
     * Well-known values for {@code input.mime_type}.
     */
    public enum MimeType {
        TEXT("text/plain"),
        JSON("application/json"),
        AUDIO_WAV("audio/wav");

        private final String value;

        MimeType(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        @Override
        public String toString() {
            return value;
        }
    }

    /**
     * Well-known values for {@code llm.system}.
     */
    public enum LLMSystem {
        OPENAI("openai"),
        ANTHROPIC("anthropic"),
        MISTRALAI("mistralai"),
        COHERE("cohere"),
        VERTEXAI("vertexai"),
        DEEPSEEK("deepseek"),
        AMAZON("amazon"),
        META("meta"),
        AI21("ai21"),
        TYPESAFE("typesafe"),
        XAI("xai");

        private final String value;

        LLMSystem(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        @Override
        public String toString() {
            return value;
        }
    }

    /**
     * Well-known values for {@code llm.provider}.
     */
    public enum LLMProvider {
        OPENAI("openai"),
        ANTHROPIC("anthropic"),
        MISTRALAI("mistralai"),
        COHERE("cohere"),
        GOOGLE("google"),
        AWS("aws"),
        AZURE("azure"),
        XAI("xai"),
        DEEPSEEK("deepseek"),
        GROQ("groq"),
        FIREWORKS("fireworks"),
        MOONSHOT("moonshot"),
        CEREBRAS("cerebras"),
        PERPLEXITY("perplexity"),
        TOGETHER("together"),
        OLLAMA("ollama"),
        META("meta"),
        ZAI("zai"),
        MINIMAX("minimax"),
        ORACLE("oracle"),
        IBM("ibm"),
        TYPESAFE("typesafe"),
        DOUBLEWORD("doubleword");

        private final String value;

        LLMProvider(String value) {
            this.value = value;
        }

        public String getValue() {
            return value;
        }

        @Override
        public String toString() {
            return value;
        }
    }

    /**
     * Well-known values for {@code decision.system}. Each constant aliases the {@link LLMSystem} constant for the same vendor.
     */
    public enum DecisionSystem {
        TYPESAFE(LLMSystem.TYPESAFE),
        OPENAI(LLMSystem.OPENAI);

        private final LLMSystem llmSystem;

        DecisionSystem(LLMSystem llmSystem) {
            this.llmSystem = llmSystem;
        }

        /** The {@link LLMSystem} constant this value aliases. */
        public LLMSystem getLLMSystem() {
            return llmSystem;
        }

        public String getValue() {
            return llmSystem.getValue();
        }

        @Override
        public String toString() {
            return llmSystem.getValue();
        }
    }

    /**
     * Well-known values for {@code decision.provider}. Each constant aliases the {@link LLMProvider} constant for the same vendor.
     */
    public enum DecisionProvider {
        TYPESAFE(LLMProvider.TYPESAFE),
        OPENAI(LLMProvider.OPENAI);

        private final LLMProvider llmProvider;

        DecisionProvider(LLMProvider llmProvider) {
            this.llmProvider = llmProvider;
        }

        /** The {@link LLMProvider} constant this value aliases. */
        public LLMProvider getLLMProvider() {
            return llmProvider;
        }

        public String getValue() {
            return llmProvider.getValue();
        }

        @Override
        public String toString() {
            return llmProvider.getValue();
        }
    }
}
