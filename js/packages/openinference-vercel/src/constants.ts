import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

/**
 * A map of Vercel AI SDK function names to OpenInference span kinds.
 * @see https://sdk.vercel.ai/docs/ai-sdk-core/telemetry#collected-data
 * These are set on Vercel spans as under the operation.name attribute.
 * They are preceded by "ai.<wrapper-name>" and may be followed by a user provided functionID.
 * Top level operation names are typically AGENT span kinds, and the inner function names are typically LLM span kinds.
 * @example ai.<wrapper-name>.<ai-sdk-function-name> <user provided functionId>
 * @example ai.generateText.doGenerate my-chat-call
 */
export const VercelSDKFunctionNameToSpanKindMap = new Map([
  ["ai.generateText", OpenInferenceSpanKind.AGENT],
  ["ai.generateText.doGenerate", OpenInferenceSpanKind.LLM],
  ["ai.generateObject", OpenInferenceSpanKind.AGENT],
  ["ai.generateObject.doGenerate", OpenInferenceSpanKind.LLM],
  ["ai.streamText", OpenInferenceSpanKind.AGENT],
  ["ai.streamText.doStream", OpenInferenceSpanKind.LLM],
  ["ai.streamObject", OpenInferenceSpanKind.AGENT],
  ["ai.streamObject.doStream", OpenInferenceSpanKind.LLM],
  ["ai.embed", OpenInferenceSpanKind.CHAIN],
  ["ai.embed.doEmbed", OpenInferenceSpanKind.EMBEDDING],
  ["ai.embedMany", OpenInferenceSpanKind.CHAIN],
  ["ai.embedMany.doEmbed", OpenInferenceSpanKind.EMBEDDING],
  ["ai.toolCall", OpenInferenceSpanKind.TOOL],
]);

/**
 * A map of Vercel eve operation names to OpenInference span kinds.
 * eve sets these on its control-flow spans under the operation.name attribute. They carry
 * gen_ai.* context attributes (e.g. gen_ai.conversation.id) but no gen_ai.operation.name, so
 * without an explicit mapping they fall through to the GenAI converter's LLM default.
 * The model call and tool execution beneath them are the chat (LLM) and execute_tool (TOOL) spans.
 * @see https://eve.dev/docs/observability/otel#trace-topology
 */
export const EveOperationNameToSpanKindMap = new Map([
  ["agent.step", OpenInferenceSpanKind.CHAIN],
  ["agent.action", OpenInferenceSpanKind.CHAIN],
]);

export const GenAIOperationNameToSpanKindMap = new Map([
  ["invoke_agent", OpenInferenceSpanKind.AGENT],
  ["agent_step", OpenInferenceSpanKind.CHAIN],
  ["chat", OpenInferenceSpanKind.LLM],
  ["execute_tool", OpenInferenceSpanKind.TOOL],
  ["embeddings", OpenInferenceSpanKind.EMBEDDING],
  ["rerank", OpenInferenceSpanKind.RERANKER],
]);
