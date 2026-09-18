import type { Attributes } from "@opentelemetry/api";
import type { RequestOptions, SystemOneRequest } from "@typesafe-ai/sdk";

import type { TraceConfig } from "@arizeai/openinference-core";
import {
  getInputAttributes,
  getOutputAttributes,
  isObjectWithStringKeys,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import { MimeType, SemanticConventions as SC } from "@arizeai/openinference-semantic-conventions";

export function getRequestAttributes({
  request,
  options,
  defaultModel,
}: {
  request: SystemOneRequest;
  options?: RequestOptions;
  defaultModel: string;
}): Attributes {
  const model = request.model ?? defaultModel;
  // Headers can carry credentials and signals are caller-owned objects. Only
  // serializable transport settings belong in invocation parameters.
  const retry = options?.retry;
  return {
    ...getInputAttributes({
      value: safelyJSONStringify({ ...request, model }) ?? "",
      mimeType: MimeType.JSON,
    }),
    [SC.LLM_MODEL_NAME]: model,
    [SC.LLM_INVOCATION_PARAMETERS]:
      safelyJSONStringify({
        model,
        timeout: options?.timeout,
        retry: retry && {
          ...retry,
          httpStatuses: retry.httpStatuses && Array.from(retry.httpStatuses),
        },
      }) ?? undefined,
  };
}

export function getResponseAttributes(result: unknown): Attributes {
  // The SDK returns parsed JSON without validation. Partial usage and malformed
  // success bodies must not produce invalid attributes or affect the caller.
  const response = isObjectWithStringKeys(result) ? result : {};
  const usage = isObjectWithStringKeys(response.usage) ? response.usage : {};
  const inputTokens = usage.input_tokens;
  const outputTokens = usage.output_tokens;
  return {
    ...getOutputAttributes({
      value: safelyJSONStringify(result) ?? "",
      mimeType: MimeType.JSON,
    }),
    // An absent model must not overwrite the request/client fallback.
    ...(typeof response.model === "string" && { [SC.LLM_MODEL_NAME]: response.model }),
    ...(typeof inputTokens === "number" && { [SC.LLM_TOKEN_COUNT_PROMPT]: inputTokens }),
    ...(typeof outputTokens === "number" && { [SC.LLM_TOKEN_COUNT_COMPLETION]: outputTokens }),
    ...(typeof inputTokens === "number" &&
      typeof outputTokens === "number" && {
        [SC.LLM_TOKEN_COUNT_TOTAL]: inputTokens + outputTokens,
      }),
  };
}

export function getMetadataAttributes({
  request,
  result,
  requestId,
  metadata,
  config,
}: {
  request: SystemOneRequest;
  result?: unknown;
  requestId?: string;
  metadata?: Record<string, unknown>;
  config: TraceConfig;
}): Attributes {
  const answers =
    isObjectWithStringKeys(result) && isObjectWithStringKeys(result.answers) ? result.answers : {};
  return {
    [SC.METADATA]:
      safelyJSONStringify({
        ...metadata,
        typesafe: {
          request_id: requestId,
          // Question names are input too. Do not reintroduce hidden data through
          // metadata, and do not copy instructions, labels, or rubrics here.
          questions: config.hideInputs
            ? undefined
            : Object.fromEntries(
                Object.entries(request.questions).map(([name, question]) => {
                  const answer = answers[name];
                  return [
                    name,
                    {
                      type: question.type,
                      ...(!config.hideOutputs &&
                        isObjectWithStringKeys(answer) &&
                        typeof answer.confidence === "number" && {
                          confidence: answer.confidence,
                        }),
                    },
                  ];
                }),
              ),
        },
      }) ?? undefined,
  };
}
