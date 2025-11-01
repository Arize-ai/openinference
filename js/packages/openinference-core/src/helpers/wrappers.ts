import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

import { AnyFn, SpanTraceOptions } from "./types";
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
 * @param options.tracer - Custom OpenTelemetry tracer instance (defaults to global tracer)
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
export function traceChain<Fn extends AnyFn>(
  fn: Fn,
  options?: Omit<SpanTraceOptions, "kind">,
): Fn {
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
 * @param options.tracer - Custom OpenTelemetry tracer instance (defaults to global tracer)
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
export function traceAgent<Fn extends AnyFn>(
  fn: Fn,
  options?: Omit<SpanTraceOptions, "kind">,
): Fn {
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
 * @param options.tracer - Custom OpenTelemetry tracer instance (defaults to global tracer)
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
 * const tracedCalculator = withToolSpan(calculate, { name: "calculator" });
 *
 * // Trace a database query tool
 * const queryDatabase = async (query: string, params: any[]) => {
 *   return await db.query(query, params);
 * };
 * const tracedDbTool = traceTool(queryDatabase);
 * ```
 */
export function traceTool<Fn extends AnyFn>(
  fn: Fn,
  options?: Omit<SpanTraceOptions, "kind">,
): Fn {
  return withSpan<Fn>(fn, { ...options, kind: OpenInferenceSpanKind.TOOL });
}
