import type {
  SDKAssistantMessage,
  SDKResultError,
  SDKResultMessage,
  SDKResultSuccess,
  SDKSystemMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type { Attributes, Context } from "@opentelemetry/api";

import {
  getAttributesFromContext,
  getInputAttributes,
  getOutputAttributes,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import { MimeType, SemanticConventions } from "@arizeai/openinference-semantic-conventions";

/**
 * Type guard: checks if a message is a system init message.
 */
export function isSystemInitMessage(msg: unknown): msg is SDKSystemMessage {
  return (
    msg != null &&
    typeof msg === "object" &&
    "type" in msg &&
    msg.type === "system" &&
    "subtype" in msg &&
    msg.subtype === "init" &&
    "session_id" in msg &&
    "model" in msg
  );
}

/**
 * Type guard: checks if a message is a result success message.
 */
export function isResultSuccessMessage(msg: unknown): msg is SDKResultSuccess {
  return (
    msg != null &&
    typeof msg === "object" &&
    "type" in msg &&
    msg.type === "result" &&
    "subtype" in msg &&
    msg.subtype === "success" &&
    "result" in msg &&
    "usage" in msg
  );
}

/**
 * Type guard: checks if a message is a result error message.
 */
export function isResultErrorMessage(msg: unknown): msg is SDKResultError {
  if (msg == null || typeof msg !== "object" || !("subtype" in msg)) return false;
  const subtype = msg.subtype;
  return (
    "type" in msg &&
    msg.type === "result" &&
    typeof subtype === "string" &&
    subtype.startsWith("error") &&
    "usage" in msg
  );
}

/**
 * Type guard: checks if a message is any kind of result message (success or error).
 */
export function isResultMessage(msg: unknown): msg is SDKResultMessage {
  return isResultSuccessMessage(msg) || isResultErrorMessage(msg);
}

/**
 * Type guard: checks if a message is an assistant message.
 */
export function isAssistantMessage(msg: unknown): msg is SDKAssistantMessage {
  return (
    msg != null &&
    typeof msg === "object" &&
    "type" in msg &&
    msg.type === "assistant" &&
    "message" in msg &&
    msg.message != null &&
    typeof msg.message === "object" &&
    "stop_reason" in msg.message
  );
}

/**
 * Extracts attributes from a system init message.
 */
export function extractInitAttributes(msg: SDKSystemMessage): {
  sessionId: string;
  model: string;
  tools: string[];
} {
  return {
    sessionId: msg.session_id,
    model: msg.model,
    tools: msg.tools,
  };
}

/**
 * Extracts the model's stop reason from a result message. The top-level
 * `stop_reason` field is available in Claude Agent SDK 0.2.31 and later.
 */
function extractStopReason(msg: SDKResultMessage): string | undefined {
  return msg.stop_reason ? String(msg.stop_reason) : undefined;
}

/**
 * Extracts the model's stop reason from an assistant message. Claude Agent SDK
 * versions 0.2.0 through 0.2.30 expose it only at `message.stop_reason`.
 */
export function extractAssistantStopReason(msg: SDKAssistantMessage): string | undefined {
  return msg.message.stop_reason ? String(msg.message.stop_reason) : undefined;
}

/**
 * Extracts span attributes from a result success message.
 */
export function extractResultSuccessAttributes(msg: SDKResultSuccess): Attributes {
  const stopReason = extractStopReason(msg);
  return {
    ...getOutputAttributes(msg.result),
    [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: msg.usage.input_tokens,
    [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: msg.usage.output_tokens,
    [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: msg.usage.input_tokens + msg.usage.output_tokens,
    [SemanticConventions.LLM_COST_TOTAL]: msg.total_cost_usd,
    [SemanticConventions.SESSION_ID]: msg.session_id,
    ...(stopReason != null ? { [SemanticConventions.LLM_FINISH_REASON]: stopReason } : {}),
  };
}

/**
 * Extracts span attributes from a result error message.
 */
export function extractResultErrorAttributes(msg: SDKResultError): Attributes {
  const errorMessages = msg.errors;
  const outputAttrs =
    errorMessages && errorMessages.length > 0
      ? getOutputAttributes({
          value: safelyJSONStringify(errorMessages) ?? "",
          mimeType: MimeType.JSON,
        })
      : {};
  const stopReason = extractStopReason(msg);
  return {
    ...outputAttrs,
    [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: msg.usage.input_tokens,
    [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: msg.usage.output_tokens,
    [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: msg.usage.input_tokens + msg.usage.output_tokens,
    [SemanticConventions.LLM_COST_TOTAL]: msg.total_cost_usd,
    [SemanticConventions.SESSION_ID]: msg.session_id,
    ...(stopReason != null ? { [SemanticConventions.LLM_FINISH_REASON]: stopReason } : {}),
  };
}

/**
 * Converts a prompt value into OpenTelemetry input attributes.
 * Strings produce text/plain attributes; objects are JSON-stringified.
 * Delegates to {@link getInputAttributes} from `@arizeai/openinference-core`.
 */
export function formatPromptAttributes(prompt: unknown): Attributes {
  if (typeof prompt === "string") {
    return getInputAttributes(prompt);
  }
  return getInputAttributes({ value: safelyJSONStringify(prompt) ?? "", mimeType: MimeType.JSON });
}

/**
 * Whether the caller already scoped this span to a session through
 * OpenInference context (`setSession`). OITracer applies that `session.id`
 * when the span starts, and a caller-supplied session id always wins over the
 * SDK's own `session_id`, so message processing must not overwrite it.
 */
export function hasContextSessionId(ctx: Context): boolean {
  return getAttributesFromContext(ctx)[SemanticConventions.SESSION_ID] !== undefined;
}

/**
 * Returns the attributes without `session.id`, for spans whose session id
 * came from context.
 */
export function withoutSessionId(attributes: Attributes): Attributes {
  const rest = { ...attributes };
  delete rest[SemanticConventions.SESSION_ID];
  return rest;
}
