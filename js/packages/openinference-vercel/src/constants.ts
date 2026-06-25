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
 * The name prefix shared by all Vercel AI SDK spans — top-level operations
 * (`ai.streamText`, `ai.generateText`, ...), their inner model calls
 * (`ai.streamText.doStream`), tool calls (`ai.toolCall`), and framework
 * wrappers built on the AI SDK (e.g. Eve's `ai.eve.turn`).
 *
 * `agentTraceMode` uses this prefix to find the outermost AI SDK span in a
 * trace (the first one to start) and promote it to the trace root, without
 * needing a per-framework list of wrapper span names.
 */
export const AI_SDK_SPAN_NAME_PREFIX = "ai.";
