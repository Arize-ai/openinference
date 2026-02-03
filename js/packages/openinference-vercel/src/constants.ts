import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

/**
 * A map of Vercel AI SDK function names to OpenInference span kinds.
 * @see https://sdk.vercel.ai/docs/ai-sdk-core/telemetry#collected-data
 * These are set on Vercel spans as under the operation.name attribute.
 * They are preceded by "ai.<wrapper-name>" and may be followed by a user provided functionID.
 * @example ai.<wrapper-name>.<ai-sdk-function-name> <user provided functionId>
 * @example ai.generateText.doGenerate my-chat-call
 */
export const VercelSDKFunctionNameToSpanKindMap = new Map([
  ["ai.generateText", OpenInferenceSpanKind.CHAIN],
  ["ai.generateText.doGenerate", OpenInferenceSpanKind.LLM],
  ["ai.generateObject", OpenInferenceSpanKind.CHAIN],
  ["ai.generateObject.doGenerate", OpenInferenceSpanKind.LLM],
  ["ai.streamText", OpenInferenceSpanKind.CHAIN],
  ["ai.streamText.doStream", OpenInferenceSpanKind.LLM],
  ["ai.streamObject", OpenInferenceSpanKind.CHAIN],
  ["ai.streamObject.doStream", OpenInferenceSpanKind.LLM],
  ["ai.embed", OpenInferenceSpanKind.CHAIN],
  ["ai.embed.doEmbed", OpenInferenceSpanKind.EMBEDDING],
  ["ai.embedMany", OpenInferenceSpanKind.CHAIN],
  ["ai.embedMany.doEmbed", OpenInferenceSpanKind.EMBEDDING],
  ["ai.toolCall", OpenInferenceSpanKind.TOOL],
]);
