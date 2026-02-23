import type {
  SDKResultError,
  SDKResultMessage,
  SDKResultSuccess,
  SDKSystemMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type { Attributes } from "@opentelemetry/api";

import {
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
    (msg as Record<string, unknown>).type === "system" &&
    "subtype" in msg &&
    (msg as Record<string, unknown>).subtype === "init" &&
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
    (msg as Record<string, unknown>).type === "result" &&
    "subtype" in msg &&
    (msg as Record<string, unknown>).subtype === "success" &&
    "result" in msg &&
    "usage" in msg
  );
}

/**
 * Type guard: checks if a message is a result error message.
 */
export function isResultErrorMessage(msg: unknown): msg is SDKResultError {
  return (
    msg != null &&
    typeof msg === "object" &&
    "type" in msg &&
    (msg as Record<string, unknown>).type === "result" &&
    "subtype" in msg &&
    typeof (msg as Record<string, unknown>).subtype === "string" &&
    ((msg as Record<string, unknown>).subtype as string).startsWith("error") &&
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
 * Extracts span attributes from a result success message.
 */
export function extractResultSuccessAttributes(msg: SDKResultSuccess): Attributes {
  return {
    ...getOutputAttributes(msg.result),
    [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: msg.usage.input_tokens,
    [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: msg.usage.output_tokens,
    [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: msg.usage.input_tokens + msg.usage.output_tokens,
    [SemanticConventions.LLM_COST_TOTAL]: msg.total_cost_usd,
    [SemanticConventions.SESSION_ID]: msg.session_id,
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
  return {
    ...outputAttrs,
    [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: msg.usage.input_tokens,
    [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: msg.usage.output_tokens,
    [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: msg.usage.input_tokens + msg.usage.output_tokens,
    [SemanticConventions.LLM_COST_TOTAL]: msg.total_cost_usd,
    [SemanticConventions.SESSION_ID]: msg.session_id,
  };
}

/**
 * Converts a prompt value into OpenTelemetry input attributes.
 * Strings produce text/plain attributes; objects are JSON-stringified.
 * Delegates to {@link getInputAttributes} from `@arizeai/openinference-core`.
 */
export function formatPromptAttributes(prompt: string | unknown): Attributes {
  if (typeof prompt === "string") {
    return getInputAttributes(prompt);
  }
  return getInputAttributes({ value: safelyJSONStringify(prompt) ?? "", mimeType: MimeType.JSON });
}
