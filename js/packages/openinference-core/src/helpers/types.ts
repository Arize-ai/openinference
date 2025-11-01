/**
 * Type definitions for OpenInference tracing functionality.
 *
 * This module provides TypeScript type definitions for the core tracing concepts
 * used throughout the OpenInference library, including span input/output structures,
 * processing functions, and configuration options.
 */

import {
  MimeType,
  OpenInferenceSpanKind,
} from "@arizeai/openinference-semantic-conventions";

import { Attributes, SpanKind, Tracer } from "@opentelemetry/api";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnyFn = (...args: any[]) => any;

/**
 * Represents the input data structure for OpenInference spans.
 *
 * This type defines how function arguments or input data should be structured
 * when creating span attributes. It provides both the serialized content and
 * metadata about the content format for proper interpretation and display.
 *
 * @example
 * ```typescript
 * const textInput: SpanInput = {
 *   value: "Hello, world!",
 *   mimeType: MimeType.TEXT
 * };
 *
 * const jsonInput: SpanInput = {
 *   value: '{"query": "search term", "limit": 10}',
 *   mimeType: MimeType.JSON
 * };
 * ```
 */
export type SpanInput = {
  /**
   * The textual representation of the input data.
   *
   * This should be a string representation of the input, either as plain text
   * or as a serialized format (like JSON) depending on the mimeType.
   */
  value: string;
  /**
   * The MIME type indicating the format of the input value.
   *
   * Specifies how the value should be interpreted (e.g., text/plain, application/json).
   * This helps tools and UIs properly display and process the input data.
   */
  mimeType: MimeType;
};

/**
 * Represents the output data structure for OpenInference spans.
 *
 * This type defines how function return values or output data should be structured
 * when creating span attributes. It provides both the serialized content and
 * metadata about the content format for proper interpretation and display.
 *
 * @example
 * ```typescript
 * const textOutput: SpanOutput = {
 *   value: "Operation completed successfully",
 *   mimeType: MimeType.TEXT
 * };
 *
 * const jsonOutput: SpanOutput = {
 *   value: '{"status": "success", "result": {"id": 123}}',
 *   mimeType: MimeType.JSON
 * };
 * ```
 */
export type SpanOutput = {
  /**
   * The textual representation of the output data.
   *
   * This should be a string representation of the output, either as plain text
   * or as a serialized format (like JSON) depending on the mimeType.
   */
  value: string;
  /**
   * The MIME type indicating the format of the output value.
   *
   * Specifies how the value should be interpreted (e.g., text/plain, application/json).
   * This helps tools and UIs properly display and process the output data.
   */
  mimeType: MimeType;
};

/**
 * Function type for processing function arguments into OpenTelemetry attributes.
 *
 * This function type defines the signature for custom input processors that convert
 * function arguments into OpenTelemetry attributes for span enrichment. The function
 * receives the original function arguments and should return an attributes object
 * that will be added to the span.
 *
 * @param args - The function arguments to process into attributes
 * @returns OpenTelemetry attributes object containing the processed input data
 *
 * @example
 * ```typescript
 * const customInputProcessor: InputToAttributesFn = (...args) => {
 *   if (args.length === 1 && typeof args[0] === 'object') {
 *     const input = args[0] as Record<string, unknown>;
 *     return {
 *       'input.type': typeof input,
 *       'input.value': JSON.stringify(input)
 *     };
 *   }
 *   return { 'input.args_count': args.length };
 * };
 * ```
 */
export type InputToAttributesFn<Fn extends AnyFn = AnyFn> = (
  ...args: Parameters<Fn>
) => Attributes;

/**
 * Function type for processing function results into OpenTelemetry attributes.
 *
 * This function type defines the signature for custom output processors that convert
 * function return values into OpenTelemetry attributes for span enrichment. The function
 * receives the original function's return value and should return an attributes object
 * that will be added to the span.
 *
 * @param result - The function's return value to process into attributes
 * @returns OpenTelemetry attributes object containing the processed output data
 *
 * @example
 * ```typescript
 * const customOutputProcessor: OutputToAttributesFn = (result) => {
 *   if (result && typeof result === 'object') {
 *     return {
 *       'output.value': JSON.stringify(result)
 *     };
 *   }
 *   return {
 *     'output.value': String(result)
 *   };
 * };
 * ```
 */
export type OutputToAttributesFn<Fn extends AnyFn = AnyFn> = (
  result: Awaited<ReturnType<Fn>>,
) => Attributes;

/**
 * Configuration options for span tracing in OpenInference.
 *
 * This interface defines all the available options for customizing how functions
 * are traced, including span naming, tracer selection, span classification, and
 * custom input/output processing. These options are used by tracing decorators
 * and wrapper functions to control the tracing behavior.
 *
 * @example
 * ```typescript
 * // Basic configuration with custom name
 * const basicOptions: SpanTraceOptions = {
 *   name: "my-custom-operation"
 * };
 *
 * // Advanced configuration with custom processing and base attributes
 * const advancedOptions: SpanTraceOptions = {
 *   name: "llm-call",
 *   kind: "llm",
 *   openTelemetrySpanKind: SpanKind.CLIENT,
 *   attributes: {
 *     'service.name': 'ai-assistant',
 *     'llm.model': 'gpt-4',
 *     'environment': 'production'
 *   },
 *   processInput: (...args) => ({ "llm.prompt": args[0] }),
 *   processOutput: (result) => ({ "llm.response": result })
 * };
 *
 * // Agent-specific configuration with context attributes
 * const agentOptions: SpanTraceOptions = {
 *   kind: OpenInferenceSpanKind.AGENT,
 *   attributes: {
 *     'agent.type': 'decision-maker',
 *     'agent.version': '2.1.0'
 *   },
 *   processInput: (...args) => ({
 *     "agent.input": JSON.stringify(args[0]),
 *     "agent.context": args[1]?.context || "none"
 *   })
 * };
 * ```
 */
export interface SpanTraceOptions<Fn extends AnyFn = AnyFn> {
  /**
   * Custom name for the span.
   *
   * If not provided, the name of the decorated function or wrapped function
   * will be used as the span name. This is useful for providing more descriptive
   * or standardized names for operations.
   *
   * @example "user-authentication", "data-processing-pipeline", "llm-inference"
   */
  name?: string;

  /**
   * Custom OpenTelemetry tracer instance to use for this span.
   *
   * If not provided, the global tracer will be used. This allows for using
   * different tracers for different parts of the application or for testing
   * purposes with mock tracers.
   *
   * @example
   * ```typescript
   * const customTracer = trace.getTracer('my-service', '1.0.0');
   * const options: SpanTraceOptions = { tracer: customTracer };
   * ```
   */
  tracer?: Tracer;

  /**
   * The OpenTelemetry span kind to classify the span's role in a trace.
   *
   * This determines how the span is categorized in the OpenTelemetry ecosystem
   * and affects how tracing tools display and analyze the span.
   *
   * @default SpanKind.INTERNAL
   *
   * @example
   * - `SpanKind.CLIENT` for outbound requests
   * - `SpanKind.SERVER` for inbound request handling
   * - `SpanKind.INTERNAL` for internal operations
   */
  openTelemetrySpanKind?: SpanKind;

  /**
   * The OpenInference span kind for semantic categorization in LLM applications.
   *
   * This provides domain-specific classification for AI/ML operations, helping
   * to organize and understand the different types of operations in an LLM workflow.
   *
   * @default OpenInferenceSpanKind.CHAIN
   *
   * @example
   * - `LLM` for language model inference
   * - `CHAIN` for workflow sequences
   * - `AGENT` for autonomous decision-making
   * - `TOOL` for external tool usage
   */
  kind?: OpenInferenceSpanKind | `${OpenInferenceSpanKind}`;

  /**
   * Custom function to process input arguments into span attributes.
   *
   * This allows for custom serialization and attribute extraction from function
   * arguments. If not provided, the default input processor will be used, which
   * safely JSON-stringifies the arguments.
   *
   * @param args - The function arguments to process
   * @returns OpenTelemetry attributes object
   *
   * @example
   * ```typescript
   * processInput: (...args) => ({
   *   'input.value': JSON.stringify(args),
   *   'input.mimeType': MimeType.JSON
   * })
   * ```
   */
  processInput?: InputToAttributesFn<Fn>;

  /**
   * Custom function to process output values into span attributes.
   *
   * This allows for custom serialization and attribute extraction from function
   * return values. If not provided, the default output processor will be used,
   * which safely JSON-stringifies the result.
   *
   * @param result - The function's return value to process
   * @returns OpenTelemetry attributes object
   *
   * @example
   * ```typescript
   * processOutput: (result) => ({
   *   'output.value': JSON.stringify(result),
   *   'output.mimeType': MimeType.JSON
   * })
   * ```
   */
  processOutput?: OutputToAttributesFn<Fn>;

  /**
   * Base attributes to be added to every span created with these options.
   *
   * These attributes will be merged with any attributes generated by input/output
   * processors and OpenInference semantic attributes. Base attributes are useful
   * for adding consistent metadata like service information, version numbers,
   * environment details, or any other static attributes that should be present
   * on all spans.
   *
   * @example
   * ```typescript
   * // Custom business context
   * attributes: {
   *   'metadata': JSON.stringify({ tenant: 'tenant-123', feature: 'new-algorithm-enabled', request: 'mobile-app' })
   * }
   * ```
   */
  attributes?: Attributes;
}

/**
 * Image reference for message content.
 *
 * Represents an image that can be included in message content,
 * typically containing a URL or other reference to the image data.
 */
export interface Image {
  url?: string;
}

/**
 * Text-based message content.
 *
 * Represents textual content within a message, used for standard
 * text-based communication in LLM interactions.
 */
export interface TextMessageContent {
  type: "text";
  text: string;
}

/**
 * Image-based message content.
 *
 * Represents image content within a message, used for multimodal
 * LLM interactions that can process visual information.
 */
export interface ImageMessageContent {
  type: "image";
  image?: Image;
}

/**
 * Union type for different types of message content.
 *
 * Supports both text and image content types for multimodal
 * LLM applications that can handle various input formats.
 */
export type MessageContent = TextMessageContent | ImageMessageContent;

/**
 * Function call details for tool invocations.
 *
 * Contains the function name and arguments for tool calls
 * made by LLMs during execution.
 */
export interface ToolCallFunction {
  name?: string;
  arguments?: string | Record<string, unknown>;
}

/**
 * Tool call information.
 *
 * Represents a complete tool call made by an LLM, including
 * the call ID and function details.
 */
export interface ToolCall {
  id?: string;
  function?: ToolCallFunction;
}

/**
 * Message structure for LLM conversations.
 *
 * Represents a single message in an LLM conversation, supporting
 * various content types, roles, and tool interactions.
 */
export interface Message {
  role?: string;
  content?: string;
  contents?: MessageContent[];
  toolCallId?: string;
  toolCalls?: ToolCall[];
}

/**
 * Detailed prompt token usage information.
 *
 * Provides granular information about token usage in prompts,
 * including cache read/write operations and audio processing.
 */
export interface PromptDetails {
  audio?: number;
  cacheRead?: number;
  cacheWrite?: number;
}

/**
 * Token count information for LLM operations.
 *
 * Tracks token usage across different parts of LLM interactions,
 * including prompt, completion, and total token counts.
 */
export interface TokenCount {
  prompt?: number;
  completion?: number;
  total?: number;
  promptDetails?: PromptDetails;
}

/**
 * Tool definition for LLM function calling.
 *
 * Defines a tool that can be called by an LLM, including
 * its JSON schema for parameter validation.
 */
export interface Tool {
  jsonSchema: string | Record<string, unknown>;
}

/**
 * Embedding representation with associated text.
 *
 * Contains both the original text and its vector representation
 * for embedding-based operations and similarity searches.
 */
export interface Embedding {
  text?: string;
  vector?: number[];
}

/**
 * Document structure for retrieval and search operations.
 *
 * Represents a document in a knowledge base or search system,
 * including content, metadata, and relevance scoring.
 */
export interface Document {
  content?: string;
  id?: string | number;
  metadata?: string | Record<string, unknown>;
  score?: number;
}
