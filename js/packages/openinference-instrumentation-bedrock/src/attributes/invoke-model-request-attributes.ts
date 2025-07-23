/**
 * Request attribute extraction for AWS Bedrock instrumentation
 *
 * Handles extraction of semantic convention attributes from InvokeModel requests including:
 * - Base model and system attributes
 * - Input message processing
 * - Tool definition processing
 * - Invocation parameters
 */

import {
  SemanticConventions,
  OpenInferenceSpanKind,
  MimeType,
  LLMProvider,
} from "@arizeai/openinference-semantic-conventions";
import { InvokeModelCommand, InferenceConfiguration } from "@aws-sdk/client-bedrock-runtime";
import { InvokeModelRequestBody, BedrockMessage } from "../types/bedrock-types";
import {
  extractTextFromContent,
  extractToolResultBlocks,
  formatImageUrl,
} from "../utils/content-processing";
import { setSpanAttribute } from "./attribute-helpers";
import { Span } from "@opentelemetry/api";

/**
 * Extracts base request attributes (model, system, provider, invocation parameters)
 */
export function extractBaseRequestAttributes({
  span,
  command,
}: {
  span: Span;
  command: InvokeModelCommand;
}): void {
  const modelId = command.input?.modelId || "unknown";
  const requestBody = parseRequestBody(command);

  setSpanAttribute(
    span,
    SemanticConventions.OPENINFERENCE_SPAN_KIND,
    OpenInferenceSpanKind.LLM,
  );
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
  setSpanAttribute(span, SemanticConventions.INPUT_MIME_TYPE, MimeType.JSON);
  setSpanAttribute(span, SemanticConventions.LLM_PROVIDER, LLMProvider.AWS);

  const invocationParams = extractInvocationParameters(requestBody);
  if (Object.keys(invocationParams).length > 0) {
    setSpanAttribute(
      span,
      SemanticConventions.LLM_INVOCATION_PARAMETERS,
      JSON.stringify(invocationParams),
    );
  }
}

/**
 * Extracts input messages attributes from request body
 */
export function extractInputMessagesAttributes({
  span,
  requestBody,
}: {
  span: Span;
  requestBody: InvokeModelRequestBody;
}): void {
  const inputValue = extractPrimaryInputValue(requestBody);
  setSpanAttribute(span, SemanticConventions.INPUT_VALUE, inputValue);

  if (requestBody.messages && Array.isArray(requestBody.messages)) {
    requestBody.messages.forEach((message, index) => {
      addMessageAttributes({ span, message, index });
    });
  }
}

/**
 * Extracts input tool attributes from request body
 */
export function extractInputToolAttributes({
  span,
  requestBody,
}: {
  span: Span;
  requestBody: InvokeModelRequestBody;
}): void {
  if (requestBody.tools && Array.isArray(requestBody.tools)) {
    requestBody.tools.forEach((tool, index) => {
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_TOOLS}.${index}.${SemanticConventions.TOOL_JSON_SCHEMA}`,
        JSON.stringify(tool),
      );
    });
  }
}

/**
 * Extracts semantic convention attributes from InvokeModel request command
 */
export function extractInvokeModelRequestAttributes({
  span,
  command,
}: {
  span: Span;
  command: InvokeModelCommand;
}): void {
  const requestBody = parseRequestBody(command);

  extractBaseRequestAttributes({ span, command });
  extractInputMessagesAttributes({ span, requestBody });
  extractInputToolAttributes({ span, requestBody });
}

// Helper functions

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
 * Safely parses the request body JSON
 */
function parseRequestBody(command: InvokeModelCommand): InvokeModelRequestBody {
  try {
    if (!command.input?.body) {
      return {} as InvokeModelRequestBody;
    }

    let bodyString: string;
    if (typeof command.input.body === "string") {
      bodyString = command.input.body;
    } else if (command.input.body instanceof Uint8Array) {
      bodyString = new TextDecoder().decode(command.input.body as Uint8Array);
    } else if (command.input.body instanceof ArrayBuffer) {
      bodyString = new TextDecoder().decode(new Uint8Array(command.input.body));
    } else if (command.input.body instanceof Buffer) {
      bodyString = command.input.body.toString('utf8');
    } else {
      // For other types, use a safer approach that doesn't assume the type
      bodyString = String(command.input.body);
    }

    return JSON.parse(bodyString);
  } catch (error) {
    return {} as InvokeModelRequestBody;
  }
}

/**
 * Interface for invocation parameters combining AWS SDK standard interface with vendor-specific parameters
 * Uses AWS SDK's InferenceConfiguration for standard parameters across all model vendors
 */
interface ExtractedInvocationParameters extends Partial<InferenceConfiguration> {
  top_k?: number;
  anthropic_version?: string;
}

/**
 * Extracts invocation parameters from request body using AWS SDK standards
 * Maps snake_case parameter names to camelCase AWS SDK convention where applicable
 */
function extractInvocationParameters(
  requestBody: InvokeModelRequestBody,
): ExtractedInvocationParameters {
  const invocationParams: ExtractedInvocationParameters = {};

  if (requestBody.max_tokens) {
    invocationParams.maxTokens = requestBody.max_tokens;
  }
  if (requestBody.temperature != null) {
    invocationParams.temperature = requestBody.temperature;
  }
  if (requestBody.top_p != null) {
    invocationParams.topP = requestBody.top_p;
  }
  if (requestBody.stop_sequences) {
    invocationParams.stopSequences = requestBody.stop_sequences;
  }

  if (requestBody.top_k != null) {
    invocationParams.top_k = requestBody.top_k;
  }
  if (requestBody.anthropic_version) {
    invocationParams.anthropic_version = requestBody.anthropic_version;
  }

  return invocationParams;
}

/**
 * Extracts the primary input value as full request body JSON
 */
function extractPrimaryInputValue(requestBody: InvokeModelRequestBody): string {
  return JSON.stringify(requestBody);
}

/**
 * Adds message attributes to the span
 */
function addMessageAttributes({
  span,
  message,
  index,
}: {
  span: Span;
  message: BedrockMessage;
  index: number;
}): void {
  if (!message.role) return;

  setSpanAttribute(
    span,
    `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_ROLE}`,
    message.role,
  );

  const isSimpleTextMessage =
    typeof message.content === "string" ||
    (Array.isArray(message.content) &&
      message.content.length === 1 &&
      message.content[0].type === "text");

  if (isSimpleTextMessage) {
    const messageContent = extractTextFromContent(message.content);
    if (messageContent) {
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_CONTENT}`,
        messageContent,
      );
    }
  } else {
    addMessageContentAttributes({ span, message, messageIndex: index });
  }

  if (Array.isArray(message.content)) {
    const toolResultBlocks = extractToolResultBlocks(message.content);
    toolResultBlocks.forEach((contentBlock, contentIndex: number) => {
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${contentIndex}.${SemanticConventions.TOOL_CALL_ID}`,
        contentBlock.tool_use_id,
      );
      setSpanAttribute(
        span,
        `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_TOOL_CALLS}.${contentIndex}.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`,
        JSON.stringify({ result: contentBlock.content }),
      );
    });
  }
}

/**
 * Adds detailed message content structure attributes
 * Message Content Structure Enhancement
 */
function addMessageContentAttributes({
  span,
  message,
  messageIndex,
}: {
  span: Span;
  message: BedrockMessage;
  messageIndex: number;
}): void {
  if (!message.content) return;

  if (typeof message.content === "string") {
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
      "text",
    );
    setSpanAttribute(
      span,
      `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TEXT}`,
      message.content,
    );
    return;
  }

  if (Array.isArray(message.content)) {
    message.content.forEach((content, contentIndex) => {
      const contentPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${messageIndex}.${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}`;

      if (content.type === "text") {
        setSpanAttribute(
          span,
          `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
          "text",
        );
        setSpanAttribute(
          span,
          `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TEXT}`,
          content.text,
        );
      } else if (content.type === "image") {
        setSpanAttribute(
          span,
          `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
          "image",
        );
        if (content.source) {
          const imageUrl = formatImageUrl(content.source);
          setSpanAttribute(
            span,
            `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`,
            imageUrl,
          );
        }
      } else if (content.type === "tool_use") {
        setSpanAttribute(
          span,
          `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
          "tool_use",
        );
        setSpanAttribute(
          span,
          `${contentPrefix}.message_content.tool_use.name`,
          content.name,
        );
        setSpanAttribute(
          span,
          `${contentPrefix}.message_content.tool_use.input`,
          JSON.stringify(content.input),
        );
        setSpanAttribute(
          span,
          `${contentPrefix}.message_content.tool_use.id`,
          content.id,
        );
      } else if (content.type === "tool_result") {
        setSpanAttribute(
          span,
          `${contentPrefix}.${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
          "tool_result",
        );
        setSpanAttribute(
          span,
          `${contentPrefix}.message_content.tool_result.tool_use_id`,
          content.tool_use_id,
        );
        setSpanAttribute(
          span,
          `${contentPrefix}.message_content.tool_result.content`,
          content.content,
        );
      }
    });
  }
}
