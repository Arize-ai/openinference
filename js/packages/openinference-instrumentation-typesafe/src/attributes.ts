import type { Attributes } from "@opentelemetry/api";
import type { RequestOptions, SystemOneRequest } from "@typesafe-ai/sdk";

import {
  getInputAttributes,
  getLLMAttributes,
  getOutputAttributes,
  isObjectWithStringKeys,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import { MimeType } from "@arizeai/openinference-semantic-conventions";

/**
 * Request attributes for `systemOne`.
 * `input.value` is the actual request body. Invocation params exclude headers/signals.
 * Input masking is handled by OITracer / TraceConfig.
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
    ...getLLMAttributes({
      requestModelName: model,
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

/**
 * Response attributes for a parsed `systemOne` body.
 * `output.value` is the actual response body. Sets `llm.response.model_name` only when
 * present so the request/client model fallback stays. Output masking is handled by OITracer.
 */
function getResponseAttributes(result: unknown): Attributes {
  const output = getOutputAttributes({
    value: safelyJSONStringify(result) ?? "",
    mimeType: MimeType.JSON,
  });
  if (!isObjectWithStringKeys(result)) return output;

  const usage = isObjectWithStringKeys(result.usage) ? result.usage : {};
  const inputTokens = usage.input_tokens;
  const outputTokens = usage.output_tokens;
  return {
    ...output,
    ...getLLMAttributes({
      ...(typeof result.model === "string" && { responseModelName: result.model }),
      tokenCount: {
        ...(typeof inputTokens === "number" && { prompt: inputTokens }),
        ...(typeof outputTokens === "number" && { completion: outputTokens }),
        ...(typeof inputTokens === "number" &&
          typeof outputTokens === "number" && {
            total: inputTokens + outputTokens,
          }),
      },
    }),
  };
}

export { getRequestAttributes, getResponseAttributes };
