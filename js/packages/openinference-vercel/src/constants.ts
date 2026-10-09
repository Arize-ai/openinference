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
  ["ai.decide", OpenInferenceSpanKind.CHAIN],
  ["ai.decide.doDecide", OpenInferenceSpanKind.DECISION],
]);

/**
 * A map of Vercel eve control-flow span names to OpenInference span kinds.
 * eve 0.75 and earlier set these under the operation.name attribute. eve 0.76 and later set
 * operation.name and gen_ai.operation.name to the generic "workflow" instead, and keep the span
 * name under the resource.name attribute, so resource.name is also matched against this map
 * when gen_ai.operation.name is "workflow".
 * Neither "workflow" nor a missing gen_ai.operation.name is a kind the GenAI converter
 * recognizes, so without an explicit mapping these spans fall through to its LLM default.
 * The model call and tool execution beneath them are the chat (LLM) and execute_tool (TOOL) spans.
 * Not applied to spans carrying an agent identity (see {@link GenAIAgentIdentityAttributes}):
 * an agent.action span for a subagent or remote-agent call is an AGENT span.
 * @see https://eve.dev/docs/observability/otel#trace-topology
 */
export const EveOperationNameToSpanKindMap = new Map([
  ["agent.step", OpenInferenceSpanKind.CHAIN],
  ["agent.action", OpenInferenceSpanKind.CHAIN],
  ["agent.approval", OpenInferenceSpanKind.CHAIN],
]);

/**
 * gen_ai.agent.* attributes that identify a span as an agent invocation. The GenAI converter
 * classifies a span carrying any of them as AGENT.
 */
export const GenAIAgentIdentityAttributes = [
  "gen_ai.agent.id",
  "gen_ai.agent.name",
  "gen_ai.agent.description",
] as const;

export const GenAIOperationNameToSpanKindMap = new Map([
  ["invoke_agent", OpenInferenceSpanKind.AGENT],
  ["agent_step", OpenInferenceSpanKind.CHAIN],
  ["chat", OpenInferenceSpanKind.LLM],
  ["execute_tool", OpenInferenceSpanKind.TOOL],
  ["embeddings", OpenInferenceSpanKind.EMBEDDING],
  ["rerank", OpenInferenceSpanKind.RERANKER],
  ["decide", OpenInferenceSpanKind.DECISION],
]);
