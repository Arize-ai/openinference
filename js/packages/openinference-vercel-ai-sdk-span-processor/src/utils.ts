import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { Attributes } from "@opentelemetry/api";
import {
  VercelSDKFunctionNameToSpanKindMap,
  VercelSemConvToOISemConvMap,
} from "./constants";

export const hasAIAttributes = (attributes: Attributes) => {
  return Object.keys(attributes).some((key) => key.startsWith("ai."));
};

/**
 *
 * @param operationName - the operation name of the span
 * Operation names are set on Vercel spans as under the operation.name attribute with the
 * @example ai.generateText.doGenerate <functionId>
 * @returns the Vercel function name from the operation name or undefined if not found
 */
const getVercelFunctionNameFromOperationName = (
  operationName: string,
): string | undefined => {
  const maybeSDKPrefix = operationName.split(" ")[0];
  if (maybeSDKPrefix == null) {
    return;
  }
  return maybeSDKPrefix.split(".")[2];
};

export const getOISpanKindFromAttributes = (
  attributes: Attributes,
): OpenInferenceSpanKind | null => {
  const maybeOperationName = attributes["operation.name"];
  if (maybeOperationName == null || typeof maybeOperationName !== "string") {
    return null;
  }
  const maybeFunctionName =
    getVercelFunctionNameFromOperationName(maybeOperationName);
  if (maybeFunctionName == null) {
    return null;
  }
  return VercelSDKFunctionNameToSpanKindMap.get(maybeFunctionName) ?? null;
};

export const getOIModelNameAttribute = (
  attributes: Attributes,
): Attributes | null => {
  if (VERCEL_AI_MODEL_ID in attributes) {
    return {
      [SemanticConventions.LLM_MODEL_NAME]: attributes[VERCEL_AI_MODEL_ID],
    };
  }
  return null;
};

export const getOIInvocationParamAttributes = (attributes: Attributes) => {
  const settingAttributes = Object.keys(attributes)
    .filter((key) => key.startsWith(VERCEL_AI_SETTINGS))
    .reduce((acc, key) => {
      const keyParts = key.split(".");
      const paramKey = keyParts[keyParts.length - 1];
      acc[paramKey] = attributes[key];
      return acc;
    }, {} as Attributes);

  return {
    [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
      JSON.stringify(settingAttributes),
  };
};

export const getOIAttributes = (attributes: Attributes) => {
  Object.entries(VercelSemConvToOISemConvMap).forEach(([key, value]) => {
    if (key in attributes) {
      attributes[value] = attributes[key];
    }
  });
};
