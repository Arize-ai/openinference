import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import type { UnknownRecord } from "../../types";
import { isArray, isDefined, isRecord, isString, isUndefined } from "../../utils";
import { getIndexedBuckets } from "./flattenedAttributeUtils";
import {
  GenAIFinishReason,
  GenAIMessagePartType,
  GenAIModality,
  GenAIRole,
} from "./genAIAttributes";
import { parseJSONIfString, toNonEmptyString, toOptionalString } from "./serializationUtils";

/** `llm.choices`, the legacy completions counterpart of `llm.prompts` */
export const LLM_CHOICES = "llm.choices";
/** The text key of a flattened `llm.prompts.<index>` entry. */
const PROMPT_TEXT = "prompt.text";
/** The text key of a flattened `llm.choices.<index>` entry. */
const COMPLETION_TEXT = "completion.text";
/** The per-message finish reason key inside a flattened output message. */
const MESSAGE_FINISH_REASON = "message.finish_reason";
/** The prefix shared by every flattened message attribute. */
const MESSAGE_ATTRIBUTE_PREFIX = "message.";
/** Matches a base64 data URL, capturing its MIME type and its base64 content. */
const DATA_URL_PATTERN = /^data:([^;]+);base64,(.+)$/;
/**
 * Provider finish reasons, lower-cased, and the GenAI finish reason each one
 * means. Reasons that are not listed pass through unchanged.
 */
const FINISH_REASON_BY_PROVIDER_FINISH_REASON: Partial<Record<string, string>> = {
  // OpenAI
  tool_calls: GenAIFinishReason.TOOL_CALL,
  function_call: GenAIFinishReason.TOOL_CALL,
  // Anthropic and Bedrock
  tool_use: GenAIFinishReason.TOOL_CALL,
  end_turn: GenAIFinishReason.STOP,
  stop_sequence: GenAIFinishReason.STOP,
  max_tokens: GenAIFinishReason.LENGTH,
  refusal: GenAIFinishReason.CONTENT_FILTER,
  content_filtered: GenAIFinishReason.CONTENT_FILTER,
  guardrail_intervened: GenAIFinishReason.CONTENT_FILTER,
  // Gemini
  safety: GenAIFinishReason.CONTENT_FILTER,
};
/**
 * `message_content.type` values that carry text. The OpenAI Responses API names
 * its text blocks `input_text` and `output_text`, and instrumentations record
 * the block type as-is.
 */
const TEXT_CONTENT_TYPES = ["text", "input_text", "output_text"];

/** A part of a GenAI message, e.g. text, a tool call or a tool call response. */
export type GenAIMessagePart = UnknownRecord;

/**
 * A message in the shape of the OTel GenAI `gen_ai.input.messages` /
 * `gen_ai.output.messages` JSON schemas.
 * @see https://github.com/open-telemetry/semantic-conventions-genai
 */
export type GenAIMessage = {
  /** Who sent the message, e.g. `user`, `assistant` or `tool` */
  role: string;
  /** The content of the message, in order */
  parts: GenAIMessagePart[];
  /** The name of the participant, when the message records one */
  name?: string;
  /** Why the model stopped generating. Only set on output messages. */
  finish_reason?: string;
};

/**
 * Lower-cases a message role and maps legacy roles to GenAI roles:
 * `agent` becomes `assistant` and `function` becomes `tool`.
 * @param role - The OpenInference message role
 * @returns The GenAI role, or undefined when there is no role
 */
function normalizeMessageRole(role: string | undefined): string | undefined {
  if (isUndefined(role)) {
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

/**
 * Lower-cases a finish reason and maps provider-specific reasons to the GenAI
 * ones, e.g. `tool_use` to `tool_call` and `max_tokens` to `length`.
 * @param finishReason - The recorded finish reason
 * @returns The GenAI finish reason, or undefined when there is none
 */
function normalizeFinishReason(finishReason: string | undefined): string | undefined {
  if (isUndefined(finishReason)) {
    return undefined;
  }
  const loweredFinishReason = finishReason.toLowerCase();
  return FINISH_REASON_BY_PROVIDER_FINISH_REASON[loweredFinishReason] ?? loweredFinishReason;
}

/**
 * Whether any of the message parts is a tool call.
 * @param parts - The parts of a message
 */
function hasToolCallPart(parts: unknown): boolean {
  return (
    isArray(parts) &&
    parts.some(
      (part) => isRecord(part) && toOptionalString(part.type) === GenAIMessagePartType.TOOL_CALL,
    )
  );
}

/**
 * Infers a finish reason from message parts: `tool_call` when the message
 * calls a tool, otherwise `stop`.
 * @param parts - The parts of an output message
 */
function getFinishReasonFromParts(parts: unknown): string {
  return hasToolCallPart(parts) ? GenAIFinishReason.TOOL_CALL : GenAIFinishReason.STOP;
}

/**
 * Builds a text message part.
 * @param content - The text
 */
function buildTextPart(content: string): GenAIMessagePart {
  return { type: GenAIMessagePartType.TEXT, content };
}

/**
 * Builds an image message part: a `blob` part for a base64 data URL, otherwise a `uri` part.
 * @param url - The image URL
 * @returns The image part, or undefined when there is no URL
 */
function buildImagePart(url: string | undefined): GenAIMessagePart | undefined {
  if (isUndefined(url)) {
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

/**
 * Converts the flattened `message.contents.<index>.*` entries of a message to
 * text and image parts. Other content types are skipped.
 * @param bucket - The `message.*` attributes of one message
 */
function getMessageContentParts(bucket: UnknownRecord): GenAIMessagePart[] {
  const imageUrlKey = `${SemanticConventions.MESSAGE_CONTENT_IMAGE}.${SemanticConventions.IMAGE_URL}`;
  const parts: GenAIMessagePart[] = [];
  for (const content of getIndexedBuckets({
    attributes: bucket,
    prefix: SemanticConventions.MESSAGE_CONTENTS,
  })) {
    const contentType = toOptionalString(content[SemanticConventions.MESSAGE_CONTENT_TYPE]);
    if (isDefined(contentType) && TEXT_CONTENT_TYPES.includes(contentType)) {
      const text = toOptionalString(content[SemanticConventions.MESSAGE_CONTENT_TEXT]);
      if (isDefined(text)) {
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

/**
 * Converts the flattened `message.tool_calls.<index>.*` entries of a message to
 * tool call parts. Tool calls without a function name are skipped.
 * @param bucket - The `message.*` attributes of one message
 */
function getMessageToolCallParts(bucket: UnknownRecord): GenAIMessagePart[] {
  const parts: GenAIMessagePart[] = [];
  for (const toolCall of getIndexedBuckets({
    attributes: bucket,
    prefix: SemanticConventions.MESSAGE_TOOL_CALLS,
  })) {
    const functionName = toOptionalString(toolCall[SemanticConventions.TOOL_CALL_FUNCTION_NAME]);
    if (isUndefined(functionName)) {
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
 *
 * @param params.content - The `message.content` text
 * @param params.parts - The parts converted from `message.contents`
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

/**
 * Resolves the response of a tool message: `message.content` when present,
 * otherwise the text of a single text part, otherwise all content parts.
 *
 * @param params.content - The `message.content` of the tool message
 * @param params.contentParts - The parts converted from `message.contents`
 */
function getBucketToolResponse({
  content,
  contentParts,
}: {
  content: string | undefined;
  contentParts: GenAIMessagePart[];
}): unknown {
  if (isDefined(content)) {
    return parseJSONIfString(content);
  }
  const [firstPart] = contentParts;
  if (contentParts.length === 1 && firstPart?.type === GenAIMessagePartType.TEXT) {
    return firstPart.content ?? "";
  }
  return contentParts;
}

/**
 * Builds the parts of a flattened message. A tool message with a tool call id
 * becomes a single tool call response part; any other message becomes its
 * text, content and tool call parts, in that order.
 *
 * @param params.bucket - The `message.*` attributes of one message
 * @param params.role - The normalized GenAI role of the message
 */
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
    isDefined(content) && !isContentDuplicated({ content, parts: contentParts });
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
  if (isUndefined(role)) {
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

/**
 * Converts the `contents` of a JSON-marshaled message to text and image parts.
 * Entries that are not records, and other content types, are skipped.
 * @param contents - The parsed `contents` array
 */
function convertMarshaledContents(contents: unknown[]): GenAIMessagePart[] {
  const parts: GenAIMessagePart[] = [];
  for (const content of contents) {
    if (!isRecord(content)) {
      continue;
    }
    const contentType = toOptionalString(content.type);
    if (isDefined(contentType) && TEXT_CONTENT_TYPES.includes(contentType)) {
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

/**
 * Converts the OpenAI-style `tool_calls` of a JSON-marshaled message to tool call parts.
 * @param toolCalls - The parsed `tool_calls` value
 */
function convertMarshaledToolCalls(toolCalls: unknown): GenAIMessagePart[] {
  if (!isArray(toolCalls)) {
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

/**
 * Reads the parts of a JSON-marshaled message. Already structured `parts` are
 * copied as-is; otherwise `content` becomes a text part, then `contents` is converted.
 * @param message - The parsed message
 */
function getMarshaledParts(message: UnknownRecord): GenAIMessagePart[] {
  const rawParts = message.parts;
  if (isArray(rawParts) && rawParts.every(isRecord)) {
    return rawParts.map((part) => ({ ...part }));
  }
  const content = toNonEmptyString(message.content);
  if (content) {
    return [buildTextPart(content)];
  }
  if (isArray(message.contents)) {
    return convertMarshaledContents(message.contents);
  }
  return [];
}

/**
 * Resolves the response of a JSON-marshaled tool message: its `content` when
 * present, otherwise its `parts`.
 * @param message - The parsed tool message
 */
function getMarshaledToolResponse(message: UnknownRecord): unknown {
  const content = toNonEmptyString(message.content);
  if (content) {
    return parseJSONIfString(content);
  }
  if (isArray(message.parts)) {
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
  if (isUndefined(role)) {
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

/**
 * Converts a JSON-marshaled list of messages to GenAI messages. Entries that
 * are not records, or that convert to no message, are dropped.
 *
 * @param params.value - The list of messages, or a string holding it as JSON
 * @param params.isOutput - Whether the messages are output (choice) messages
 */
function parseMarshaledMessages({
  value,
  isOutput,
}: {
  value: unknown;
  isOutput: boolean;
}): GenAIMessage[] {
  const parsedValue = parseJSONIfString(value);
  if (!isArray(parsedValue)) {
    return [];
  }
  return parsedValue
    .filter(isRecord)
    .map((message) => buildMessageFromMarshaled({ message, isOutput }))
    .filter(isDefined);
}

/**
 * The finish reasons explicitly recorded on the span via `llm.finish_reason`
 * @param attributes - The OpenInference span attributes
 * @returns The normalized finish reasons, or an empty list when none are recorded
 */
export function getExplicitFinishReasons(attributes: UnknownRecord): string[] {
  const finishReason = attributes[SemanticConventions.LLM_FINISH_REASON];
  if (isString(finishReason)) {
    return [normalizeFinishReason(finishReason) || GenAIFinishReason.STOP];
  }
  if (isArray(finishReason)) {
    return finishReason
      .map((rawFinishReason) => normalizeFinishReason(String(rawFinishReason)))
      .filter(isDefined)
      .filter((normalizedFinishReason) => normalizedFinishReason !== "");
  }
  return [];
}

/**
 * The finish reason for output messages that record none: the explicit
 * `llm.finish_reason` when present, otherwise `tool_call` when any output
 * message calls a tool, otherwise `stop`.
 * @param attributes - The OpenInference span attributes
 */
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
 *
 * @param params.attributes - The OpenInference span attributes
 * @param params.prefix - The message attribute prefix
 * @param params.isOutput - Whether the messages are output (choice) messages
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
    .filter(isDefined);
}

/**
 * Reads the texts of legacy completions attributes, e.g. `llm.prompts` or
 * `llm.choices`, in either list or flattened (`llm.prompts.0.prompt.text`) form.
 *
 * @param params.attributes - The OpenInference span attributes
 * @param params.prefix - The completions attribute prefix
 * @param params.textKey - The text key of a flattened entry, e.g. `prompt.text`
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
  if (isArray(parsedTexts)) {
    return parsedTexts.map((text) => toOptionalString(text) ?? String(text));
  }
  return getIndexedBuckets({ attributes, prefix })
    .map((bucket) => toOptionalString(bucket[textKey]))
    .filter(isDefined);
}

/**
 * Converts the input of an LLM span to the value of `gen_ai.input.messages`.
 * `llm.input_messages` is preferred; legacy `llm.prompts` become user messages.
 * @param attributes - The OpenInference span attributes
 */
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

/**
 * Converts the output of an LLM span to the value of `gen_ai.output.messages`.
 * `llm.output_messages` is preferred; legacy `llm.choices` become assistant messages.
 * @param attributes - The OpenInference span attributes
 */
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
