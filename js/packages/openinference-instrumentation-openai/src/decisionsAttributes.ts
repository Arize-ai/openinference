import type { Attributes } from "@opentelemetry/api";
import type { Decision, DecisionCreateParams } from "openai/resources/decisions";

import {
  getDecisionAttributes,
  getInputAttributes,
  getOutputAttributes,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import {
  DecisionProvider,
  DecisionSystem,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";

/**
 * Builds the OpenInference attributes for a Decisions API request.
 *
 * Decision spans identify the model under `decision.*` rather than `llm.*`,
 * and carry the raw request as JSON on `input.value` instead of input
 * messages: the interesting part of a decision request is the set of typed
 * questions and candidate answers, which has no message equivalent.
 *
 * @param body - The `decisions.create` request body.
 * @returns `decision.system`, `decision.provider`, `decision.request.model_name`,
 * `decision.model_name` and the input attributes.
 * @see https://github.com/Arize-ai/openinference/blob/main/spec/decision_spans.md
 */
export function getDecisionsRequestAttributes(body: DecisionCreateParams): Attributes {
  const input = safelyJSONStringify(body);
  return {
    ...getDecisionAttributes({
      system: DecisionSystem.OPENAI,
      provider: DecisionProvider.OPENAI,
      requestModelName: body.model,
    }),
    ...(input != null ? getInputAttributes({ value: input, mimeType: MimeType.JSON }) : {}),
  };
}

/**
 * Builds the OpenInference attributes for a Decisions API response.
 *
 * Records the raw response as JSON on `output.value`, the model that served
 * the decision as `decision.response.model_name` (and as `decision.model_name`,
 * falling back to the requested model when the response omits it), and the
 * reported usage as `decision.token_count.input` / `decision.token_count.output`.
 *
 * @param args - Attribute sources.
 * @param args.response - The parsed `decisions.create` response.
 * @param args.requestModelName - The model named in the request, used as the
 * `decision.model_name` fallback.
 * @returns The output and decision response attributes.
 */
export function getDecisionsResponseAttributes({
  response,
  requestModelName,
}: {
  response: Decision;
  requestModelName?: string;
}): Attributes {
  const output = safelyJSONStringify(response);
  return {
    ...(output != null ? getOutputAttributes({ value: output, mimeType: MimeType.JSON }) : {}),
    ...getDecisionAttributes({
      requestModelName,
      responseModelName: response.model,
      tokenCount: {
        input: response.usage?.input_tokens,
        output: response.usage?.output_tokens,
      },
    }),
  };
}
