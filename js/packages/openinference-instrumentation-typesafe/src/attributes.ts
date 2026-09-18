import type { Attributes } from "@opentelemetry/api";
import type { RequestOptions, SystemOneRequest } from "@typesafe-ai/sdk";

import type { TraceConfig } from "@arizeai/openinference-core";
import {
  getInputAttributes,
  getLLMAttributes,
  getMetadataAttributes,
  getOutputAttributes,
  isObjectWithStringKeys,
  REDACTED_VALUE,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import { MimeType, SemanticConventions } from "@arizeai/openinference-semantic-conventions";

const { INPUT_VALUE, METADATA, OUTPUT_VALUE } = SemanticConventions;

export type TypeSafeMetadataArgs = {
  request: SystemOneRequest;
  result?: unknown;
  requestId?: string;
  metadata?: Record<string, unknown>;
  config: TraceConfig;
};

/**
 * Request attributes for `systemOne`.
 * Structured JSON only — no chat messages. Invocation params exclude headers/signals.
 * Skips serializing the full payload when `hideInputs` is set.
 */
function getRequestAttributes({
  request,
  options,
  defaultModel,
  config,
}: {
  request: SystemOneRequest;
  options?: RequestOptions;
  defaultModel: string;
  config: TraceConfig;
}): Attributes {
  const model = request.model ?? defaultModel;
  const retry = options?.retry;
  return {
    ...(config.hideInputs
      ? { [INPUT_VALUE]: REDACTED_VALUE }
      : getInputAttributes({
          value: safelyJSONStringify({ ...request, model }) ?? "",
          mimeType: MimeType.JSON,
        })),
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
 * Sets `llm.response.model_name` only when present so request/client fallback stays.
 * Skips serializing the full payload when `hideOutputs` is set; still extracts model/tokens.
 */
function getResponseAttributes(result: unknown, config: TraceConfig): Attributes {
  const output = config.hideOutputs
    ? { [OUTPUT_VALUE]: REDACTED_VALUE }
    : getOutputAttributes({
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

/**
 * Merges context metadata with reserved `typesafe` question type/confidence.
 * `hideInputs` drops questions; `hideOutputs` drops confidence.
 */
function getTypeSafeMetadataAttributes({
  request,
  result,
  requestId,
  metadata,
  config,
}: TypeSafeMetadataArgs): Attributes {
  const answers =
    isObjectWithStringKeys(result) && isObjectWithStringKeys(result.answers) ? result.answers : {};
  const typesafe = {
    request_id: requestId,
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
  };
  const serialized = getMetadataAttributes({ ...metadata, typesafe });
  // Core falls back to "{}"; omit empty metadata rather than emitting a useless object.
  return serialized[METADATA] === "{}" ? {} : serialized;
}

export { getRequestAttributes, getResponseAttributes, getTypeSafeMetadataAttributes };
