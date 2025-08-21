import { Span, diag } from "@opentelemetry/api";
import {
  ConverseCommand,
  ConverseRequest,
  ToolConfiguration,
  SystemContentBlock,
  Message,
  ConverseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import {
  SemanticConventions,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
import {
  withSafety,
  isObjectWithStringKeys,
} from "@arizeai/openinference-core";
import {
  setSpanAttribute,
  aggregateMessages,
  processMessages,
  extractModelName,
} from "./attribute-helpers";

/**
 * Type guard to safely validate ConverseRequest structure
 * Ensures the input object has required modelId and messages properties for processing
 *
 * @param input The input object to validate
 * @returns {boolean} True if input is a valid ConverseRequest, false otherwise
 */
function isConverseRequest(input: unknown): input is ConverseRequest {
  if (!isObjectWithStringKeys(input)) {
    return false;
  }

  return (
    "modelId" in input &&
    typeof input.modelId === "string" &&
    "messages" in input &&
    Array.isArray(input.messages)
  );
}

/**
 * Extracts base request attributes for OpenInference semantic conventions
 * Sets fundamental LLM attributes including model, system, provider, and invocation parameters
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.input The validated ConverseRequest to extract attributes from
 */
function extractBaseRequestAttributes({
  span,
  input,
}: {
  span: Span;
  input: ConverseRequest;
}): void {
  const modelId = input.modelId || "unknown";

  setSpanAttribute(
    span,
    SemanticConventions.LLM_MODEL_NAME,
    extractModelName(modelId),
  );

  if (input.inferenceConfig && Object.keys(input.inferenceConfig).length > 0) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_INVOCATION_PARAMETERS,
      JSON.stringify(input.inferenceConfig),
    );
  }

  setSpanAttribute(
    span,
    SemanticConventions.INPUT_VALUE,
    JSON.stringify(input),
  );
  setSpanAttribute(span, SemanticConventions.INPUT_MIME_TYPE, MimeType.JSON);
}

/**
 * Extracts input messages attributes with system prompt aggregation
 * Processes system prompts and conversation messages into OpenInference message format
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.input The ConverseRequest containing messages and system prompts
 */
function extractInputMessagesAttributes({
  span,
  input,
}: {
  span: Span;
  input: ConverseRequest;
}): void {
  const systemPrompts: SystemContentBlock[] = input.system || [];
  const messages: Message[] = input.messages || [];

  const aggregatedMessages = aggregateMessages(systemPrompts, messages);

  processMessages({
    span,
    messages: aggregatedMessages,
    baseKey: SemanticConventions.LLM_INPUT_MESSAGES,
  });
}

/**
 * Extracts tool configuration attributes from Converse request
 * Processes tool definitions and sets them as JSON schema attributes
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.input The ConverseRequest containing tool configuration
 */
function extractInputToolAttributes({
  span,
  input,
}: {
  span: Span;
  input: ConverseRequest;
}): void {
  const toolConfig: ToolConfiguration | undefined = input.toolConfig;
  if (!toolConfig?.tools) return;

  toolConfig.tools.forEach((tool, index: number) => {
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`,
      JSON.stringify(tool),
    );
  });
}

/**
 * Extracts semantic convention attributes from Converse command and sets them on the span
 * Main entry point for processing Converse API requests with comprehensive error handling
 *
 * Processes:
 * - Base model and system attributes
 * - Input messages with system prompt aggregation
 * - Tool configuration definitions
 *
 * @param params Object containing extraction parameters
 * @param params.span The OpenTelemetry span to set attributes on
 * @param params.command The ConverseCommand to extract attributes from
 */
export const extractConverseRequestAttributes = withSafety({
  fn: ({
    span,
    command,
  }: {
    span: Span;
    command: ConverseCommand | ConverseStreamCommand;
  }): void => {
    const input = command.input;
    if (!input || !isConverseRequest(input)) {
      return;
    }

    extractBaseRequestAttributes({ span, input });
    extractInputMessagesAttributes({ span, input });
    extractInputToolAttributes({ span, input });
  },
  onError: (error) => {
    diag.warn("Error extracting Converse request attributes:", error);
  },
});
