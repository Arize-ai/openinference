import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  VercelSemanticConvention,
  VercelSemanticConventions,
} from "./VercelSemanticConventions";

/**
 * A map of Vercel SDK function names to OpenInference span kinds.
 * @see https://sdk.vercel.ai/docs/ai-sdk-core/telemetry#collected-data
 * These are set on Vercel spans as under the operation.name attribute.
 * They are preceded by "ai.<wrapper-call-name>" and may be followed by a user provided functionID.
 * @example ai.<wrapper-call-name>.<vercel-sdk-function-name> <user provided functionId>
 * @example ai.generateText.doGenerate my-chat-call
 */
export const VercelSDKFunctionNameToSpanKindMap = new Map([
  ["doGenerate", OpenInferenceSpanKind.LLM],
  ["doStream", OpenInferenceSpanKind.LLM],
  ["doEmbed", OpenInferenceSpanKind.EMBEDDING],
  ["toolCall", OpenInferenceSpanKind.TOOL],
]);

type OISemanticConvention = SemanticConventions;

export const VercelSemConvToOISemConvMap: Record<
  VercelSemanticConvention,
  OISemanticConvention
> = {
  [VercelSemanticConventions.MODEL_ID]: SemanticConventions.LLM_MODEL_NAME,
  [VercelSemanticConventions.SETTINGS]:
    SemanticConventions.LLM_INVOCATION_PARAMETERS,
  [VercelSemanticConventions.METADATA]: SemanticConventions.METADATA,
  [VercelSemanticConventions.TOKEN_COUNT_COMPLETION]:
    SemanticConventions.LLM_TOKEN_COUNT_COMPLETION,
  [VercelSemanticConventions.TOKEN_COUNT_PROMPT]:
    SemanticConventions.LLM_TOKEN_COUNT_PROMPT,
} as const;
