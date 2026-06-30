import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

import type { AnyFn, SpanTraceOptions } from "./types";
import { withSpan } from "./withSpan";

/**
 * Wraps a function with tracing capabilities, specifically marking it as a CHAIN span.
 *
 * This is a convenience function that wraps `withSpan` with the OpenInference span kind
 * pre-configured to CHAIN. Chain spans represent a sequence of operations or a workflow
 * in an LLM application, such as a series of processing steps or a pipeline.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @template Fn - The function type being wrapped, preserving original signature
 * @param fn - The function to wrap with CHAIN span tracing
 * @param options - Configuration options for tracing behavior (excluding kind)
 * @param options.tracer - Custom OpenTelemetry tracer instance (otherwise the current global tracer
 * provider is resolved when the wrapper is invoked)
 * @param options.name - Custom span name (defaults to function name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.processInput - Custom function to process input arguments into attributes
 * @param options.processOutput - Custom function to process output values into attributes
 * @param options.attributes - Base attributes to be added to every span created
 *
 * @returns A wrapped function with identical signature that creates CHAIN spans during execution
 *
 * @example
 * ```typescript
 * // Trace a data processing pipeline
 * const processData = (data: any[]) => {
 *   return data.map(item => transform(item)).filter(item => validate(item));
 * };
 * const tracedProcess = traceChain(processData, { name: "data-pipeline" });
 *
 * // Trace a multi-step workflow
 * const executeWorkflow = async (input: WorkflowInput) => {
 *   const step1 = await preprocessInput(input);
 *   const step2 = await processStep(step1);
 *   return await finalizeOutput(step2);
 * };
 * const tracedWorkflow = traceChain(executeWorkflow);
 * ```
 */
export function traceChain<Fn extends AnyFn>(fn: Fn, options?: Omit<SpanTraceOptions, "kind">): Fn {
  return withSpan<Fn>(fn, {
    ...options,
    kind: OpenInferenceSpanKind.CHAIN,
  });
}

/**
 * Wraps a function with tracing capabilities, specifically marking it as an AGENT span.
 *
 * This is a convenience function that wraps `withSpan` with the OpenInference span kind
 * pre-configured to AGENT. Agent spans represent autonomous decision-making entities
 * in an LLM application, such as AI agents, chat bots, or intelligent assistants that
 * can reason, plan, and execute actions.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @template Fn - The function type being wrapped, preserving original signature
 * @param fn - The function to wrap with AGENT span tracing
 * @param options - Configuration options for tracing behavior (excluding kind)
 * @param options.tracer - Custom OpenTelemetry tracer instance (otherwise the current global tracer
 * provider is resolved when the wrapper is invoked)
 * @param options.name - Custom span name (defaults to function name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.processInput - Custom function to process input arguments into attributes
 * @param options.processOutput - Custom function to process output values into attributes
 * @param options.attributes - Base attributes to be added to every span created
 *
 * @returns A wrapped function with identical signature that creates AGENT spans during execution
 *
 * @example
 * ```typescript
 * // Trace an AI agent's decision-making process
 * const makeDecision = async (context: AgentContext) => {
 *   const analysis = await analyzeContext(context);
 *   const plan = await createPlan(analysis);
 *   return await executePlan(plan);
 * };
 * const tracedAgent = traceAgent(makeDecision, { name: "decision-agent" });
 *
 * // Trace a chatbot response generation
 * const generateResponse = (userMessage: string, history: Message[]) => {
 *   const intent = classifyIntent(userMessage);
 *   const context = buildContext(history);
 *   return generateReply(intent, context);
 * };
 * const tracedChatbot = traceAgent(generateResponse);
 * ```
 */
export function traceAgent<Fn extends AnyFn>(fn: Fn, options?: Omit<SpanTraceOptions, "kind">): Fn {
  return withSpan<Fn>(fn, { ...options, kind: OpenInferenceSpanKind.AGENT });
}

/**
 * Wraps a function with tracing capabilities, specifically marking it as a TOOL span.
 *
 * This is a convenience function that wraps `withSpan` with the OpenInference span kind
 * pre-configured to TOOL. Tool spans represent external tools, utilities, or services
 * that an LLM application can invoke, such as APIs, databases, calculators, web scrapers,
 * or any external function that provides specific capabilities.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @template Fn - The function type being wrapped, preserving original signature
 * @param fn - The function to wrap with TOOL span tracing
 * @param options - Configuration options for tracing behavior (excluding kind)
 * @param options.tracer - Custom OpenTelemetry tracer instance (otherwise the current global tracer
 * provider is resolved when the wrapper is invoked)
 * @param options.name - Custom span name (defaults to function name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.processInput - Custom function to process input arguments into attributes
 * @param options.processOutput - Custom function to process output values into attributes
 * @param options.attributes - Base attributes to be added to every span created
 *
 * @returns A wrapped function with identical signature that creates TOOL spans during execution
 *
 * @example
 * ```typescript
 * // Trace an API call tool
 * const fetchWeather = async (city: string) => {
 *   const response = await fetch(`/api/weather?city=${city}`);
 *   return response.json();
 * };
 * const tracedWeatherTool = traceTool(fetchWeather, { name: "weather-api" });
 *
 * // Trace a calculator tool
 * const calculate = (expression: string) => {
 *   return eval(expression); // Note: eval is dangerous, use proper parser
 * };
 * const tracedCalculator = traceTool(calculate, { name: "calculator" });
 *
 * // Trace a database query tool
 * const queryDatabase = async (query: string, params: any[]) => {
 *   return await db.query(query, params);
 * };
 * const tracedDbTool = traceTool(queryDatabase);
 * ```
 */
export function traceTool<Fn extends AnyFn>(fn: Fn, options?: Omit<SpanTraceOptions, "kind">): Fn {
  return withSpan<Fn>(fn, { ...options, kind: OpenInferenceSpanKind.TOOL });
}

/**
 * Wraps a function with tracing capabilities, specifically marking it as an LLM span.
 *
 * This is a convenience function that wraps `withSpan` with the OpenInference span kind
 * pre-configured to LLM. LLM spans represent invocations of a large language model, such
 * as chat completions, text completions, or other inference calls to a foundation model.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @template Fn - The function type being wrapped, preserving original signature
 * @param fn - The function to wrap with LLM span tracing
 * @param options - Configuration options for tracing behavior (excluding kind)
 * @param options.tracer - Custom OpenTelemetry tracer instance (otherwise the current global tracer
 * provider is resolved when the wrapper is invoked)
 * @param options.name - Custom span name (defaults to function name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.processInput - Custom function to process input arguments into attributes
 * @param options.processOutput - Custom function to process output values into attributes
 * @param options.attributes - Base attributes to be added to every span created
 *
 * @returns A wrapped function with identical signature that creates LLM spans during execution
 *
 * @example
 * ```typescript
 * // Trace a chat completion call
 * const chatCompletion = async (messages: Message[]) => {
 *   return await client.chat.completions.create({ model: "gpt-4", messages });
 * };
 * const tracedCompletion = traceLLM(chatCompletion, { name: "chat-completion" });
 * ```
 */
export function traceLLM<Fn extends AnyFn>(fn: Fn, options?: Omit<SpanTraceOptions, "kind">): Fn {
  return withSpan<Fn>(fn, { ...options, kind: OpenInferenceSpanKind.LLM });
}

/**
 * Wraps a function with tracing capabilities, specifically marking it as a RETRIEVER span.
 *
 * This is a convenience function that wraps `withSpan` with the OpenInference span kind
 * pre-configured to RETRIEVER. Retriever spans represent operations that fetch relevant
 * documents or context from a knowledge base, vector store, or search index, typically as
 * part of a retrieval-augmented generation (RAG) workflow.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @template Fn - The function type being wrapped, preserving original signature
 * @param fn - The function to wrap with RETRIEVER span tracing
 * @param options - Configuration options for tracing behavior (excluding kind)
 * @param options.tracer - Custom OpenTelemetry tracer instance (otherwise the current global tracer
 * provider is resolved when the wrapper is invoked)
 * @param options.name - Custom span name (defaults to function name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.processInput - Custom function to process input arguments into attributes
 * @param options.processOutput - Custom function to process output values into attributes
 * @param options.attributes - Base attributes to be added to every span created
 *
 * @returns A wrapped function with identical signature that creates RETRIEVER spans during execution
 *
 * @example
 * ```typescript
 * // Trace a vector store similarity search
 * const retrieveDocuments = async (query: string) => {
 *   return await vectorStore.similaritySearch(query, 5);
 * };
 * const tracedRetriever = traceRetriever(retrieveDocuments, { name: "vector-search" });
 * ```
 */
export function traceRetriever<Fn extends AnyFn>(
  fn: Fn,
  options?: Omit<SpanTraceOptions, "kind">,
): Fn {
  return withSpan<Fn>(fn, {
    ...options,
    kind: OpenInferenceSpanKind.RETRIEVER,
  });
}

/**
 * Wraps a function with tracing capabilities, specifically marking it as a RERANKER span.
 *
 * This is a convenience function that wraps `withSpan` with the OpenInference span kind
 * pre-configured to RERANKER. Reranker spans represent operations that reorder or score a
 * set of candidate documents by relevance to a query, commonly used to refine the results
 * returned by a retriever before passing them to a language model.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @template Fn - The function type being wrapped, preserving original signature
 * @param fn - The function to wrap with RERANKER span tracing
 * @param options - Configuration options for tracing behavior (excluding kind)
 * @param options.tracer - Custom OpenTelemetry tracer instance (otherwise the current global tracer
 * provider is resolved when the wrapper is invoked)
 * @param options.name - Custom span name (defaults to function name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.processInput - Custom function to process input arguments into attributes
 * @param options.processOutput - Custom function to process output values into attributes
 * @param options.attributes - Base attributes to be added to every span created
 *
 * @returns A wrapped function with identical signature that creates RERANKER spans during execution
 *
 * @example
 * ```typescript
 * // Trace a document reranking operation
 * const rerankDocuments = async (query: string, documents: Document[]) => {
 *   return await reranker.rerank(query, documents);
 * };
 * const tracedReranker = traceReranker(rerankDocuments, { name: "rerank" });
 * ```
 */
export function traceReranker<Fn extends AnyFn>(
  fn: Fn,
  options?: Omit<SpanTraceOptions, "kind">,
): Fn {
  return withSpan<Fn>(fn, { ...options, kind: OpenInferenceSpanKind.RERANKER });
}

/**
 * Wraps a function with tracing capabilities, specifically marking it as an EMBEDDING span.
 *
 * This is a convenience function that wraps `withSpan` with the OpenInference span kind
 * pre-configured to EMBEDDING. Embedding spans represent operations that convert text or
 * other data into vector representations, such as generating embeddings for documents or
 * queries used in semantic search and retrieval.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @template Fn - The function type being wrapped, preserving original signature
 * @param fn - The function to wrap with EMBEDDING span tracing
 * @param options - Configuration options for tracing behavior (excluding kind)
 * @param options.tracer - Custom OpenTelemetry tracer instance (otherwise the current global tracer
 * provider is resolved when the wrapper is invoked)
 * @param options.name - Custom span name (defaults to function name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.processInput - Custom function to process input arguments into attributes
 * @param options.processOutput - Custom function to process output values into attributes
 * @param options.attributes - Base attributes to be added to every span created
 *
 * @returns A wrapped function with identical signature that creates EMBEDDING spans during execution
 *
 * @example
 * ```typescript
 * // Trace an embedding generation call
 * const embedText = async (text: string) => {
 *   return await client.embeddings.create({ model: "text-embedding-3-small", input: text });
 * };
 * const tracedEmbedding = traceEmbedding(embedText, { name: "embed-text" });
 * ```
 */
export function traceEmbedding<Fn extends AnyFn>(
  fn: Fn,
  options?: Omit<SpanTraceOptions, "kind">,
): Fn {
  return withSpan<Fn>(fn, {
    ...options,
    kind: OpenInferenceSpanKind.EMBEDDING,
  });
}

/**
 * Wraps a function with tracing capabilities, specifically marking it as a GUARDRAIL span.
 *
 * This is a convenience function that wraps `withSpan` with the OpenInference span kind
 * pre-configured to GUARDRAIL. Guardrail spans represent safety, validation, or policy
 * checks applied to inputs or outputs of an LLM application, such as content moderation,
 * PII detection, or compliance enforcement.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @template Fn - The function type being wrapped, preserving original signature
 * @param fn - The function to wrap with GUARDRAIL span tracing
 * @param options - Configuration options for tracing behavior (excluding kind)
 * @param options.tracer - Custom OpenTelemetry tracer instance (otherwise the current global tracer
 * provider is resolved when the wrapper is invoked)
 * @param options.name - Custom span name (defaults to function name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.processInput - Custom function to process input arguments into attributes
 * @param options.processOutput - Custom function to process output values into attributes
 * @param options.attributes - Base attributes to be added to every span created
 *
 * @returns A wrapped function with identical signature that creates GUARDRAIL spans during execution
 *
 * @example
 * ```typescript
 * // Trace a content moderation check
 * const moderateContent = async (text: string) => {
 *   return await moderationClient.check(text);
 * };
 * const tracedGuardrail = traceGuardrail(moderateContent, { name: "content-moderation" });
 * ```
 */
export function traceGuardrail<Fn extends AnyFn>(
  fn: Fn,
  options?: Omit<SpanTraceOptions, "kind">,
): Fn {
  return withSpan<Fn>(fn, {
    ...options,
    kind: OpenInferenceSpanKind.GUARDRAIL,
  });
}

/**
 * Wraps a function with tracing capabilities, specifically marking it as an EVALUATOR span.
 *
 * This is a convenience function that wraps `withSpan` with the OpenInference span kind
 * pre-configured to EVALUATOR. Evaluator spans represent operations that assess or score
 * the quality of an LLM application's output, such as relevance scoring, correctness
 * checks, or LLM-as-a-judge evaluations.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @template Fn - The function type being wrapped, preserving original signature
 * @param fn - The function to wrap with EVALUATOR span tracing
 * @param options - Configuration options for tracing behavior (excluding kind)
 * @param options.tracer - Custom OpenTelemetry tracer instance (otherwise the current global tracer
 * provider is resolved when the wrapper is invoked)
 * @param options.name - Custom span name (defaults to function name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.processInput - Custom function to process input arguments into attributes
 * @param options.processOutput - Custom function to process output values into attributes
 * @param options.attributes - Base attributes to be added to every span created
 *
 * @returns A wrapped function with identical signature that creates EVALUATOR spans during execution
 *
 * @example
 * ```typescript
 * // Trace an LLM-as-a-judge evaluation
 * const evaluateAnswer = async (question: string, answer: string) => {
 *   return await judge.score({ question, answer });
 * };
 * const tracedEvaluator = traceEvaluator(evaluateAnswer, { name: "answer-evaluation" });
 * ```
 */
export function traceEvaluator<Fn extends AnyFn>(
  fn: Fn,
  options?: Omit<SpanTraceOptions, "kind">,
): Fn {
  return withSpan<Fn>(fn, {
    ...options,
    kind: OpenInferenceSpanKind.EVALUATOR,
  });
}

/**
 * Wraps a function with tracing capabilities, specifically marking it as a PROMPT span.
 *
 * This is a convenience function that wraps `withSpan` with the OpenInference span kind
 * pre-configured to PROMPT. Prompt spans represent operations that construct, render, or
 * template a prompt before it is sent to a language model, such as formatting a prompt
 * template with variables or assembling a set of messages.
 *
 * @experimental This API is experimental and may change in future versions
 *
 * @template Fn - The function type being wrapped, preserving original signature
 * @param fn - The function to wrap with PROMPT span tracing
 * @param options - Configuration options for tracing behavior (excluding kind)
 * @param options.tracer - Custom OpenTelemetry tracer instance (otherwise the current global tracer
 * provider is resolved when the wrapper is invoked)
 * @param options.name - Custom span name (defaults to function name)
 * @param options.openTelemetrySpanKind - OpenTelemetry span kind (defaults to INTERNAL)
 * @param options.processInput - Custom function to process input arguments into attributes
 * @param options.processOutput - Custom function to process output values into attributes
 * @param options.attributes - Base attributes to be added to every span created
 *
 * @returns A wrapped function with identical signature that creates PROMPT spans during execution
 *
 * @example
 * ```typescript
 * // Trace a prompt template rendering
 * const renderPrompt = (variables: Record<string, string>) => {
 *   return promptTemplate.format(variables);
 * };
 * const tracedPrompt = tracePrompt(renderPrompt, { name: "render-prompt" });
 * ```
 */
export function tracePrompt<Fn extends AnyFn>(
  fn: Fn,
  options?: Omit<SpanTraceOptions, "kind">,
): Fn {
  return withSpan<Fn>(fn, { ...options, kind: OpenInferenceSpanKind.PROMPT });
}
