import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import type { UnknownRecord } from "./conversionUtils";
import {
  getIndexedBuckets,
  isRecord,
  parseJSONIfString,
  toNonEmptyString,
  toOptionalString,
} from "./conversionUtils";
import {
  GenAIFinishReason,
  GenAIMessagePartType,
  GenAIModality,
  GenAIRole,
} from "./genAIAttributes";

/** `llm.choices`, the legacy completions counterpart of `llm.prompts` */
export const LLM_CHOICES = "llm.choices";
const PROMPT_TEXT = "prompt.text";
const COMPLETION_TEXT = "completion.text";
const MESSAGE_FINISH_REASON = "message.finish_reason";
const MESSAGE_ATTRIBUTE_PREFIX = "message.";
const DATA_URL_PATTERN = /^data:([^;]+);base64,(.+)$/;
const TOOL_CALL_FINISH_REASON_ALIASES = ["tool_calls", "function_call"];
/**
 * `message_content.type` values that carry text. The OpenAI Responses API names
 * its text blocks `input_text` and `output_text`, and instrumentations record
 * the block type as-is.
 */
const TEXT_CONTENT_TYPES = ["text", "input_text", "output_text"];

export type GenAIMessagePart = UnknownRecord;

/**
 * A message in the shape of the OTel GenAI `gen_ai.input.messages` /
 * `gen_ai.output.messages` JSON schemas.
 */
export type GenAIMessage = {
  role: string;
  parts: GenAIMessagePart[];
  name?: string;
  finish_reason?: string;
};

function normalizeMessageRole(role: string | undefined): string | undefined {
  if (role === undefined) {
    return undefined;
  }
  const loweredRole = role.toLowerCase();
  if (loweredRole === "agent") {
    return GenAIRole.ASSISTANT;
  }
  if (loweredRole === "function") {
    return GenAIRole.TOOL;
  }
  return loweredRole;
}

function normalizeFinishReason(finishReason: string | undefined): string | undefined {
  if (finishReason === undefined) {
    return undefined;
  }
  const loweredFinishReason = finishReason.toLowerCase();
  if (TOOL_CALL_FINISH_REASON_ALIASES.includes(loweredFinishReason)) {
    return GenAIFinishReason.TOOL_CALL;
  }
  return loweredFinishReason;
}

function hasToolCallPart(parts: unknown): boolean {
  return (
    Array.isArray(parts) &&
    parts.some(
      (part) => isRecord(part) && toOptionalString(part.type) === GenAIMessagePartType.TOOL_CALL,
    )
  );
}

function getFinishReasonFromParts(parts: unknown): string {
  return hasToolCallPart(parts) ? GenAIFinishReason.TOOL_CALL : GenAIFinishReason.STOP;
}

function buildTextPart(content: string): GenAIMessagePart {
  return { type: GenAIMessagePartType.TEXT, content };
}

function buildImagePart(url: string | undefined): GenAIMessagePart | undefined {
  if (url === undefined) {
    return undefined;
  }
  const dataUrlMatch = DATA_URL_PATTERN.exec(url);
  if (dataUrlMatch) {
    return {
      type: GenAIMessagePartType.BLOB,
      modality: GenAIModality.IMAGE,
      mime_type: dataUrlMatch[1],
      content: dataUrlMatch[2],
    };
  }
  return { type: GenAIMessagePartType.URI, modality: GenAIModality.IMAGE, uri: url };
}

function getMessageContentParts(bucket: UnknownRecord): GenAIMessagePart[] {
  const imageUrlKey = `${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`;
  const parts: GenAIMessagePart[] = [];
  for (const content of getIndexedBuckets({
    attributes: bucket,
    prefix: SemanticConventions.MESSAGE_CONTENTS,
  })) {
    const contentType = toOptionalString(content[SemanticConventions.MESSAGE_CONTENT_TYPE]);
    if (contentType !== undefined && TEXT_CONTENT_TYPES.includes(contentType)) {
      const text = toOptionalString(content[SemanticConventions.MESSAGE_CONTENT_TEXT]);
      if (text !== undefined) {
        parts.push(buildTextPart(text));
      }
    } else if (contentType === "image") {
      const imagePart = buildImagePart(toOptionalString(content[imageUrlKey]));
      if (imagePart) {
        parts.push(imagePart);
      }
    }
  }
  return parts;
}

function getMessageToolCallParts(bucket: UnknownRecord): GenAIMessagePart[] {
  const parts: GenAIMessagePart[] = [];
  for (const toolCall of getIndexedBuckets({
    attributes: bucket,
    prefix: SemanticConventions.MESSAGE_TOOL_CALLS,
  })) {
    const functionName = toOptionalString(toolCall[SemanticConventions.TOOL_CALL_FUNCTION_NAME]);
    if (functionName === undefined) {
      continue;
    }
    const part: GenAIMessagePart = {
      type: GenAIMessagePartType.TOOL_CALL,
      name: functionName,
      arguments: parseJSONIfString(toolCall[SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON]),
    };
    const toolCallId = toNonEmptyString(toolCall[SemanticConventions.TOOL_CALL_ID]);
    if (toolCallId) {
      part.id = toolCallId;
    }
    parts.push(part);
  }
  return parts;
}

/**
 * Whether `message.content` repeats the single text entry of `message.contents`
 */
function isContentDuplicated({
  content,
  parts,
}: {
  content: string;
  parts: GenAIMessagePart[];
}): boolean {
  const textParts = parts.filter(
    (part) => toOptionalString(part.type) === GenAIMessagePartType.TEXT,
  );
  return textParts.length === 1 && textParts[0]?.content === content;
}

function getBucketToolResponse({
  content,
  contentParts,
}: {
  content: string | undefined;
  contentParts: GenAIMessagePart[];
}): unknown {
  if (content !== undefined) {
    return parseJSONIfString(content);
  }
  const [firstPart] = contentParts;
  if (contentParts.length === 1 && firstPart?.type === GenAIMessagePartType.TEXT) {
    return firstPart.content ?? "";
  }
  return contentParts;
}

function getBucketParts({ bucket, role }: { bucket: UnknownRecord; role: string }) {
  const contentParts = getMessageContentParts(bucket);
  const content = toOptionalString(bucket[SemanticConventions.MESSAGE_CONTENT]);
  const toolCallId = toNonEmptyString(bucket[SemanticConventions.MESSAGE_TOOL_CALL_ID]);
  if (role === GenAIRole.TOOL && toolCallId) {
    return [
      {
        type: GenAIMessagePartType.TOOL_CALL_RESPONSE,
        id: toolCallId,
        response: getBucketToolResponse({ content, contentParts }),
      },
    ];
  }
  const shouldPrependContent =
    content !== undefined && !isContentDuplicated({ content, parts: contentParts });
  return [
    ...(shouldPrependContent ? [buildTextPart(content)] : []),
    ...contentParts,
    ...getMessageToolCallParts(bucket),
  ];
}

/**
 * Builds a GenAI message from the flattened `message.*` attributes of a single
 * OpenInference message.
 *
 * @param params.bucket - The `message.*` attributes of one message
 * @param params.isOutput - Whether the message is an output (choice) message
 * @param params.fallbackFinishReason - The finish reason to use when the message has none
 */
function buildMessageFromBucket({
  bucket,
  isOutput,
  fallbackFinishReason,
}: {
  bucket: UnknownRecord;
  isOutput: boolean;
  fallbackFinishReason?: string;
}): GenAIMessage | undefined {
  const role = normalizeMessageRole(toOptionalString(bucket[SemanticConventions.MESSAGE_ROLE]));
  if (role === undefined) {
    return undefined;
  }
  const parts = getBucketParts({ bucket, role });
  if (parts.length === 0) {
    return undefined;
  }
  const message: GenAIMessage = { role, parts };
  const name =
    toNonEmptyString(bucket[SemanticConventions.MESSAGE_NAME]) ||
    toNonEmptyString(bucket[SemanticConventions.MESSAGE_FUNCTION_CALL_NAME]);
  if (name) {
    message.name = name;
  }
  if (isOutput) {
    message.finish_reason =
      normalizeFinishReason(toOptionalString(bucket[MESSAGE_FINISH_REASON])) ||
      fallbackFinishReason ||
      getFinishReasonFromParts(parts);
  }
  return message;
}

function convertMarshaledContents(contents: unknown[]): GenAIMessagePart[] {
  const parts: GenAIMessagePart[] = [];
  for (const content of contents) {
    if (!isRecord(content)) {
      continue;
    }
    const contentType = toOptionalString(content.type);
    if (contentType !== undefined && TEXT_CONTENT_TYPES.includes(contentType)) {
      const text = toNonEmptyString(content.text);
      if (text) {
        parts.push(buildTextPart(text));
      }
    } else if (contentType === "image" && isRecord(content.image)) {
      const imagePart = buildImagePart(
        toOptionalString(content.image[SemanticConventions.IMAGE_URL]),
      );
      if (imagePart) {
        parts.push(imagePart);
      }
    }
  }
  return parts;
}

function convertMarshaledToolCalls(toolCalls: unknown): GenAIMessagePart[] {
  if (!Array.isArray(toolCalls)) {
    return [];
  }
  const parts: GenAIMessagePart[] = [];
  for (const toolCall of toolCalls) {
    if (!isRecord(toolCall) || !isRecord(toolCall.function)) {
      continue;
    }
    const part: GenAIMessagePart = {
      type: GenAIMessagePartType.TOOL_CALL,
      name: toNonEmptyString(toolCall.function.name) ?? "",
      arguments: parseJSONIfString(toolCall.function.arguments),
    };
    const toolCallId = toNonEmptyString(toolCall.id);
    if (toolCallId) {
      part.id = toolCallId;
    }
    parts.push(part);
  }
  return parts;
}

function getMarshaledParts(message: UnknownRecord): GenAIMessagePart[] {
  const rawParts = message.parts;
  if (Array.isArray(rawParts) && rawParts.every(isRecord)) {
    return rawParts.map((part) => ({ ...part }));
  }
  const content = toNonEmptyString(message.content);
  if (content) {
    return [buildTextPart(content)];
  }
  if (Array.isArray(message.contents)) {
    return convertMarshaledContents(message.contents);
  }
  return [];
}

function getMarshaledToolResponse(message: UnknownRecord): unknown {
  const content = toNonEmptyString(message.content);
  if (content) {
    return parseJSONIfString(content);
  }
  if (Array.isArray(message.parts)) {
    return [...message.parts];
  }
  return "";
}

/**
 * Builds a GenAI message from a JSON-marshaled message, which is either a
 * record of flattened `message.*` attributes or an already structured message.
 *
 * @param params.message - The parsed message
 * @param params.isOutput - Whether the message is an output (choice) message
 */
function buildMessageFromMarshaled({
  message,
  isOutput,
}: {
  message: UnknownRecord;
  isOutput: boolean;
}): GenAIMessage | undefined {
  const explicitFinishReason = normalizeFinishReason(toNonEmptyString(message.finish_reason));
  const hasFlattenedMessageKeys = Object.keys(message).some((key) =>
    key.startsWith(MESSAGE_ATTRIBUTE_PREFIX),
  );
  if (hasFlattenedMessageKeys) {
    const inferredFinishReason = hasToolCallPart(message.parts)
      ? GenAIFinishReason.TOOL_CALL
      : undefined;
    return buildMessageFromBucket({
      bucket: message,
      isOutput,
      fallbackFinishReason: explicitFinishReason || inferredFinishReason,
    });
  }

  const role = normalizeMessageRole(toOptionalString(message.role));
  if (role === undefined) {
    return undefined;
  }
  const toolCallId = toNonEmptyString(message.tool_call_id);
  const isToolResponse = role === GenAIRole.TOOL && Boolean(toolCallId);
  const parts = isToolResponse
    ? [
        {
          type: GenAIMessagePartType.TOOL_CALL_RESPONSE,
          id: toolCallId,
          response: getMarshaledToolResponse(message),
        },
      ]
    : [...getMarshaledParts(message), ...convertMarshaledToolCalls(message.tool_calls)];
  if (parts.length === 0) {
    return undefined;
  }

  const genAIMessage: GenAIMessage = { role, parts };
  const name = toNonEmptyString(message.name);
  if (name) {
    genAIMessage.name = name;
  }
  if (isOutput) {
    genAIMessage.finish_reason = explicitFinishReason || getFinishReasonFromParts(parts);
  }
  return genAIMessage;
}

function parseMarshaledMessages({
  value,
  isOutput,
}: {
  value: unknown;
  isOutput: boolean;
}): GenAIMessage[] {
  const parsedValue = parseJSONIfString(value);
  if (!Array.isArray(parsedValue)) {
    return [];
  }
  return parsedValue
    .filter(isRecord)
    .map((message) => buildMessageFromMarshaled({ message, isOutput }))
    .filter((message) => message !== undefined);
}

/**
 * The finish reasons explicitly recorded on the span via `llm.finish_reason`
 */
export function getExplicitFinishReasons(attributes: UnknownRecord): string[] {
  const finishReason = attributes[SemanticConventions.LLM_FINISH_REASON];
  if (typeof finishReason === "string") {
    return [normalizeFinishReason(finishReason) || GenAIFinishReason.STOP];
  }
  if (Array.isArray(finishReason)) {
    return finishReason
      .map((rawFinishReason) => normalizeFinishReason(String(rawFinishReason)))
      .filter((normalizedFinishReason) => normalizedFinishReason !== undefined)
      .filter((normalizedFinishReason) => normalizedFinishReason !== "");
  }
  return [];
}

function getDefaultOutputFinishReason(attributes: UnknownRecord): string {
  const [explicitFinishReason] = getExplicitFinishReasons(attributes);
  if (explicitFinishReason) {
    return explicitFinishReason;
  }
  const outputMessagePrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.`;
  const hasOutputToolCalls = Object.keys(attributes).some(
    (key) =>
      key.startsWith(outputMessagePrefix) && key.includes(SemanticConventions.MESSAGE_TOOL_CALLS),
  );
  return hasOutputToolCalls ? GenAIFinishReason.TOOL_CALL : GenAIFinishReason.STOP;
}

/**
 * Converts the OpenInference messages under a prefix (`llm.input_messages` or
 * `llm.output_messages`) to GenAI messages. Flattened attributes take
 * precedence over a JSON-marshaled value at the prefix itself.
 */
function getMessages({
  attributes,
  prefix,
  isOutput,
}: {
  attributes: UnknownRecord;
  prefix: string;
  isOutput: boolean;
}): GenAIMessage[] {
  const buckets = getIndexedBuckets({ attributes, prefix });
  if (buckets.length === 0) {
    return parseMarshaledMessages({ value: attributes[prefix], isOutput });
  }
  const fallbackFinishReason = isOutput ? getDefaultOutputFinishReason(attributes) : undefined;
  return buckets
    .map((bucket) => buildMessageFromBucket({ bucket, isOutput, fallbackFinishReason }))
    .filter((message) => message !== undefined);
}

/**
 * Reads the texts of legacy completions attributes, e.g. `llm.prompts` or
 * `llm.choices`, in either list or flattened (`llm.prompts.0.prompt.text`) form.
 */
function getCompletionTexts({
  attributes,
  prefix,
  textKey,
}: {
  attributes: UnknownRecord;
  prefix: string;
  textKey: string;
}): string[] {
  const parsedTexts = parseJSONIfString(attributes[prefix]);
  if (Array.isArray(parsedTexts)) {
    return parsedTexts.map((text) => toOptionalString(text) ?? String(text));
  }
  return getIndexedBuckets({ attributes, prefix })
    .map((bucket) => toOptionalString(bucket[textKey]))
    .filter((text) => text !== undefined);
}

export function getInputMessages(attributes: UnknownRecord): GenAIMessage[] {
  const messages = getMessages({
    attributes,
    prefix: SemanticConventions.LLM_INPUT_MESSAGES,
    isOutput: false,
  });
  if (messages.length > 0) {
    return messages;
  }
  return getCompletionTexts({
    attributes,
    prefix: SemanticConventions.LLM_PROMPTS,
    textKey: PROMPT_TEXT,
  }).map((prompt) => ({ role: GenAIRole.USER, parts: [buildTextPart(prompt)] }));
}

export function getOutputMessages(attributes: UnknownRecord): GenAIMessage[] {
  const messages = getMessages({
    attributes,
    prefix: SemanticConventions.LLM_OUTPUT_MESSAGES,
    isOutput: true,
  });
  if (messages.length > 0) {
    return messages;
  }
  const choices = getCompletionTexts({ attributes, prefix: LLM_CHOICES, textKey: COMPLETION_TEXT });
  if (choices.length === 0) {
    return [];
  }
  const finishReason = getDefaultOutputFinishReason(attributes);
  return choices.map((choice) => ({
    role: GenAIRole.ASSISTANT,
    parts: [buildTextPart(choice)],
    finish_reason: finishReason,
  }));
}

/**
 * The value for `gen_ai.response.finish_reasons`: the explicit
 * `llm.finish_reason` when present, otherwise one finish reason per output
 * message.
 *
 * @param params.attributes - The OpenInference span attributes
 * @param params.outputMessages - The already converted output messages
 */
export function getResponseFinishReasons({
  attributes,
  outputMessages,
}: {
  attributes: UnknownRecord;
  outputMessages: GenAIMessage[];
}): string[] {
  const explicitFinishReasons = getExplicitFinishReasons(attributes);
  if (explicitFinishReasons.length > 0) {
    return explicitFinishReasons;
  }
  return outputMessages.map(
    (message) => normalizeFinishReason(message.finish_reason) || GenAIFinishReason.STOP,
  );
}
