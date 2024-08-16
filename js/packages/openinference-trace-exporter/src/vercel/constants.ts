import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  VercelSemanticConvention,
  VercelSemanticConventions,
} from "./VercelSemanticConventions";
import { OpenInferenceSemanticConvention } from "../types";

/**
 * A map of Vercel SDK function names to OpenInference span kinds.
 * @see https://sdk.vercel.ai/docs/ai-sdk-core/telemetry#collected-data
 * These are set on Vercel spans as under the operation.name attribute.
 * They are preceded by "ai.<wrapper-call-name>" and may be followed by a user provided functionID.
 * @example ai.<wrapper-call-name>.<vercel-sdk-function-name> <user provided functionId>
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

export const VercelSemConvToOISemConvMap: Record<
  VercelSemanticConvention,
  OpenInferenceSemanticConvention
> = {
  [VercelSemanticConventions.MODEL_ID]: SemanticConventions.LLM_MODEL_NAME,
  [VercelSemanticConventions.SETTINGS]:
    SemanticConventions.LLM_INVOCATION_PARAMETERS,
  [VercelSemanticConventions.METADATA]: SemanticConventions.METADATA,
  [VercelSemanticConventions.TOKEN_COUNT_COMPLETION]:
    SemanticConventions.LLM_TOKEN_COUNT_COMPLETION,
  [VercelSemanticConventions.TOKEN_COUNT_PROMPT]:
    SemanticConventions.LLM_TOKEN_COUNT_PROMPT,
  [VercelSemanticConventions.RESULT_TEXT]: SemanticConventions.OUTPUT_VALUE,
  [VercelSemanticConventions.RESULT_TOOL_CALLS]:
    SemanticConventions.MESSAGE_TOOL_CALLS,
  [VercelSemanticConventions.RESULT_OBJECT]: SemanticConventions.OUTPUT_VALUE,
  [VercelSemanticConventions.PROMPT]: SemanticConventions.INPUT_VALUE,
  [VercelSemanticConventions.PROMPT_MESSAGES]:
    SemanticConventions.LLM_INPUT_MESSAGES,
  [VercelSemanticConventions.EMBEDDING_TEXT]:
    SemanticConventions.EMBEDDING_TEXT,
  [VercelSemanticConventions.EMBEDDING_VECTOR]:
    SemanticConventions.EMBEDDING_VECTOR,
  [VercelSemanticConventions.EMBEDDING_TEXTS]:
    SemanticConventions.EMBEDDING_TEXT,
  [VercelSemanticConventions.EMBEDDING_VECTORS]:
    SemanticConventions.EMBEDDING_VECTOR,
  [VercelSemanticConventions.TOOL_CALL_NAME]: SemanticConventions.TOOL_NAME,
  [VercelSemanticConventions.TOOL_CALL_ARGS]:
    SemanticConventions.TOOL_PARAMETERS,
} as const;
