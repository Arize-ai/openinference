import type { Attributes } from "@opentelemetry/api";
import type { RequestOptions, SystemOneRequest } from "@typesafe-ai/sdk";

import {
  getDecisionAttributes,
  getInputAttributes,
  getLLMAttributes,
  getOutputAttributes,
  isObjectWithStringKeys,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import { MimeType } from "@arizeai/openinference-semantic-conventions";

/**
 * Builds OpenInference attributes for a `systemOne` request.
 *
 * Serializes `request` onto `input.value`, records the resolved model as
 * `decision.request.model_name`, and records invocation parameters from
 * `options` (timeout and retry only; headers and abort signals are omitted).
 *
 * @param args - Attribute source values for the outbound call.
 * @param args.request - The `systemOne` request body.
 * @param args.options - Optional per-call request options.
 * @param args.defaultModel - Client default model when `request.model` is unset.
 * @returns OpenInference input, decision request, and invocation parameter attributes.
 */
function getRequestAttributes({
  request,
  options,
  defaultModel,
}: {
  request: SystemOneRequest;
  options?: RequestOptions;
  defaultModel: string;
}): Attributes {
  const model = request.model ?? defaultModel;
  const retry = options?.retry;
  return {
    ...getInputAttributes({
      value: safelyJSONStringify(request) ?? "",
      mimeType: MimeType.JSON,
    }),
    ...getDecisionAttributes({ requestModelName: model }),
    ...getLLMAttributes({
      invocationParameters: {
        model,
        timeout: options?.timeout,
        retry: retry && {
          ...retry,
          httpStatuses: retry.httpStatuses && Array.from(retry.httpStatuses),
        },
      },
    }),
  };
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

/**
 * Builds OpenInference attributes for a parsed `systemOne` response.
 *
 * Serializes `result` onto `output.value`. When `result` is an object, also
 * records `decision.response.model_name` and `decision.token_count.*` from
 * `usage` when those fields are present.
 *
 * @param result - The parsed `systemOne` response body.
 * @returns OpenInference output and decision response attributes.
 */
function getResponseAttributes(result: unknown): Attributes {
  const output = getOutputAttributes({
    value: safelyJSONStringify(result) ?? "",
    mimeType: MimeType.JSON,
  });
  if (!isObjectWithStringKeys(result)) return output;

  const usage = isObjectWithStringKeys(result.usage) ? result.usage : undefined;

  return {
    ...output,
    ...getDecisionAttributes({
      responseModelName: asString(result.model),
      tokenCount: {
        input: asNumber(usage?.input_tokens),
        output: asNumber(usage?.output_tokens),
      },
    }),
  };
}

export { getRequestAttributes, getResponseAttributes };
