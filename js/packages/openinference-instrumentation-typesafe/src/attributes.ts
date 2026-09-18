import type { Attributes } from "@opentelemetry/api";
import type { RequestOptions, SystemOneRequest } from "@typesafe-ai/sdk";

import type { TraceConfig } from "@arizeai/openinference-core";
import {
  getInputAttributes,
  getOutputAttributes,
  isObjectWithStringKeys,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import { MimeType, SemanticConventions } from "@arizeai/openinference-semantic-conventions";

const {
  LLM_INVOCATION_PARAMETERS,
  LLM_MODEL_NAME,
  LLM_REQUEST_MODEL_NAME,
  LLM_RESPONSE_MODEL_NAME,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_TOTAL,
  METADATA,
} = SemanticConventions;

/**
 * Builds request-side OpenInference attributes for a `systemOne` call.
 *
 * Emits:
 * - `input.value` — full JSON request with the resolved model (`application/json`)
 * - `llm.model_name` / `llm.request.model_name` — request model, else the client's `defaultModel`
 * - `llm.invocation_parameters` — model plus explicit `timeout` / `retry` only
 *   (headers, credentials, and abort signals are never copied)
 *
 * Does **not** emit `llm.input_messages`. State, questions, instructions, and
 * criteria stay in `input.value` as structured JSON.
 *
 * @param request - The `systemOne` request body from the SDK
 * @param options - Optional per-call transport overrides from the SDK
 * @param defaultModel - Client default used when `request.model` is omitted
 */
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
    [LLM_MODEL_NAME]: model,
    [LLM_REQUEST_MODEL_NAME]: model,
    [LLM_INVOCATION_PARAMETERS]:
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

/**
 * Builds response-side OpenInference attributes from a parsed `systemOne` body.
 *
 * Emits:
 * - `output.value` — full JSON response (`application/json`)
 * - `llm.model_name` / `llm.response.model_name` — only when the response includes a string `model`
 *   (absent model must not overwrite the request/client fallback)
 * - `llm.token_count.prompt` / `.completion` when present; `.total` only when both exist
 *
 * Does **not** emit `llm.output_messages`. Answers, probabilities, and score
 * legends stay in `output.value`. Tolerates partial or malformed success bodies
 * without throwing — the SDK does not validate response JSON.
 *
 * @param result - Parsed JSON body (or any value); non-objects yield empty usage fields
 */
export function getResponseAttributes(result: unknown): Attributes {
  // The SDK returns parsed JSON without validation. Partial usage and malformed
  // success bodies must not produce invalid attributes or affect the caller.
  const response = isObjectWithStringKeys(result) ? result : {};
  const usage = isObjectWithStringKeys(response.usage) ? response.usage : {};
  const inputTokens = usage.input_tokens;
  const outputTokens = usage.output_tokens;
  const responseModel = typeof response.model === "string" ? response.model : undefined;
  return {
    ...getOutputAttributes({
      value: safelyJSONStringify(result) ?? "",
      mimeType: MimeType.JSON,
    }),
    // An absent model must not overwrite the request/client fallback.
    ...(responseModel && {
      [LLM_MODEL_NAME]: responseModel,
      [LLM_RESPONSE_MODEL_NAME]: responseModel,
    }),
    ...(typeof inputTokens === "number" && { [LLM_TOKEN_COUNT_PROMPT]: inputTokens }),
    ...(typeof outputTokens === "number" && { [LLM_TOKEN_COUNT_COMPLETION]: outputTokens }),
    ...(typeof inputTokens === "number" &&
      typeof outputTokens === "number" && {
        [LLM_TOKEN_COUNT_TOTAL]: inputTokens + outputTokens,
      }),
  };
}

/**
 * Builds the `metadata` JSON attribute, merging active OpenInference context
 * metadata with a reserved `typesafe` object.
 *
 * `metadata.typesafe` contains:
 * - `request_id` — when known (response header or API error)
 * - `questions` — per-question `{ type, confidence? }` only (never instructions,
 *   labels, or rubrics, so `hideInputs` cannot be undone via metadata)
 *
 * Masking:
 * - `hideInputs` → omits `questions` entirely
 * - `hideOutputs` → omits `confidence` on each question
 * - Preserves caller context keys; `typesafe` is reserved for this instrumentor
 *
 * Noul answers have no confidence and do not get a synthetic value.
 *
 * @returns Attributes with a single `metadata` key, or empty if stringify fails
 */
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
  /** Active OpenInference context metadata to merge (session workflow tags, etc.). */
  metadata?: Record<string, unknown>;
  config: TraceConfig;
}): Attributes {
  const answers =
    isObjectWithStringKeys(result) && isObjectWithStringKeys(result.answers) ? result.answers : {};
  return {
    [METADATA]:
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
