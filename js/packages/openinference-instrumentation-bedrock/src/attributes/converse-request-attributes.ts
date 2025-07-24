import { Span, diag } from "@opentelemetry/api";
import { 
  ConverseCommand,
  ConverseRequest,
  ToolConfiguration,
  SystemContentBlock,
  Message,
} from "@aws-sdk/client-bedrock-runtime";
import {
  SemanticConventions,
  OpenInferenceSpanKind,
  MimeType,
  LLMProvider,
} from "@arizeai/openinference-semantic-conventions";
import { withSafety } from "@arizeai/openinference-core";
import {
  setSpanAttribute,
  aggregateMessages,
  processMessages,
} from "./attribute-helpers";

/**
 * Type guard to safely validate ConverseRequest structure
 */
function isConverseRequest(input: unknown): input is ConverseRequest {
  if (!input || typeof input !== "object" || input === null) {
    return false;
  }
  
  const obj = input as Record<string, unknown>;
  return (
    "modelId" in obj &&
    typeof obj.modelId === "string" &&
    "messages" in obj &&
    Array.isArray(obj.messages)
  );
}

/**
 * Extracts vendor-specific system name from model ID
 */
function getSystemFromModelId(modelId: string): string {
  if (modelId.includes("anthropic")) return "anthropic";
  if (modelId.includes("ai21")) return "ai21";
  if (modelId.includes("amazon")) return "amazon";
  if (modelId.includes("cohere")) return "cohere";
  if (modelId.includes("meta")) return "meta";
  if (modelId.includes("mistral")) return "mistral";
  return "bedrock";
}

/**
 * Extracts clean model name from full model ID
 */
function extractModelName(modelId: string): string {
  const parts = modelId.split(".");
  if (parts.length > 1) {
    const modelPart = parts[1];
    if (modelId.includes("anthropic")) {
      const versionIndex = modelPart.indexOf("-v");
      if (versionIndex > 0) {
        return modelPart.substring(0, versionIndex);
      }
    }
    return modelPart;
  }
  return modelId;
}

/**
 * Extracts base request attributes: model, system, provider, parameters
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
    SemanticConventions.OPENINFERENCE_SPAN_KIND,
    OpenInferenceSpanKind.LLM,
  );
  setSpanAttribute(span, SemanticConventions.LLM_PROVIDER, LLMProvider.AWS);
  setSpanAttribute(
    span,
    SemanticConventions.LLM_SYSTEM,
    getSystemFromModelId(modelId),
  );
  setSpanAttribute(
    span,
    SemanticConventions.LLM_MODEL_NAME,
    extractModelName(modelId),
  );

  // Use AWS SDK InferenceConfiguration directly - no conversion needed!
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
 * Extracts tool configuration attributes
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
    // Store the tool data as-is without normalization
    // This avoids the complexity of tool format standardization across providers
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`,
      JSON.stringify(tool),
    );
  });
}

/**
 * Extracts request attributes from Converse command and sets them on the span
 */
export const extractConverseRequestAttributes = withSafety({
  fn: ({ span, command }: { span: Span; command: ConverseCommand }): void => {
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
