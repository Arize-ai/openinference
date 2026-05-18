import type { AttributeValue, Attributes, SpanKind } from "@opentelemetry/api";
import {
  ATTR_GEN_AI_AGENT_DESCRIPTION,
  ATTR_GEN_AI_AGENT_ID,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OPERATION_NAME,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_REQUEST_FREQUENCY_PENALTY,
  ATTR_GEN_AI_REQUEST_MAX_TOKENS,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_REQUEST_PRESENCE_PENALTY,
  ATTR_GEN_AI_REQUEST_SEED,
  ATTR_GEN_AI_REQUEST_STOP_SEQUENCES,
  ATTR_GEN_AI_REQUEST_TEMPERATURE,
  ATTR_GEN_AI_REQUEST_TOP_K,
  ATTR_GEN_AI_REQUEST_TOP_P,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_SYSTEM,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_DESCRIPTION,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_TOOL_TYPE,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
} from "@opentelemetry/semantic-conventions/incubating";

import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import type { ChatMessage, GenericPart } from "./__generated__/opentelemetryInputMessages.js";
import type { OutputMessage } from "./__generated__/opentelemetryOutputMessages.js";
import {
  getMimeType,
  getNumber,
  getString,
  getStringArray,
  merge,
  safelyJSONStringify,
  safelyParseJSON,
  set,
  toStringContent,
} from "./utils.js";

export type GenAIInputMessage = ChatMessage;
export type GenAIInputMessagePart = ChatMessage["parts"][number];
export type GenAIOutputMessage = OutputMessage & {
  /** @deprecated use parts instead */
  text?: string;
};
export type GenAIOutputMessagePart = OutputMessage["parts"][number];

export type GenAISpanEvent = {
  name: string;
  attributes?: Attributes;
};

export type GenAISpanLike = {
  name?: string;
  kind?: SpanKind;
  attributes: Attributes;
  events?: GenAISpanEvent[];
};

export type MutableGenAISpanLike = GenAISpanLike & {
  attributes: Attributes;
};

export type FinishReasonStrategy = "first" | "join" | "preserve-array-as-json";

export type ProviderMapping = "strict" | "system-as-provider-fallback";

export type SpanKindResolver = (input: {
  name?: string;
  kind?: SpanKind;
  attributes: Attributes;
  events: GenAISpanEvent[];
  defaultKind?: OpenInferenceSpanKind;
}) => OpenInferenceSpanKind | undefined;

export type ConvertGenAISpanOptions = {
  spanKindResolver?: SpanKindResolver;
  finishReasonStrategy?: FinishReasonStrategy;
  providerMapping?: ProviderMapping;
};

const AGENT_KIND_PREFIXES = [
  ATTR_GEN_AI_AGENT_ID,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_AGENT_DESCRIPTION,
] as const;

const TOOL_EXECUTION_PREFIXES = [
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_TOOL_DESCRIPTION,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_TYPE,
] as const;

const INPUT_MESSAGE_EVENT_ROLES: Record<string, string> = {
  "gen_ai.system.message": "system",
  "gen_ai.user.message": "user",
  "gen_ai.assistant.message": "assistant",
  "gen_ai.tool.message": "tool",
};

const OUTPUT_MESSAGE_EVENT_NAMES = new Set(["gen_ai.choice"]);

// Shared part parsing
type AnyPart = GenAIInputMessagePart | GenAIOutputMessagePart;

/**
 * Type guard for a GenAI chat message
 * @param value - The value to check
 * @returns True if the value is a chat message, false otherwise
 */
const isGenAIChatMessage = (value: unknown): value is ChatMessage => {
  if (typeof value !== "object" || value === null) return false;
  if (!("role" in value) || !("parts" in value)) return false;
  if (typeof value.role !== "string" || !Array.isArray(value.parts)) return false;
  if (!value.parts || !Array.isArray(value.parts)) return false;
  return true;
};

const isGenAIMessageLike = (
  value: unknown,
): value is { role: string; parts?: AnyPart[]; content?: unknown } => {
  return (
    typeof value === "object" && value !== null && "role" in value && typeof value.role === "string"
  );
};

/**
 * Process genai message parts into openinference attributes
 *
 * @param params - The parameters to process the message parts
 */
const processMessageParts = ({
  attrs,
  msgPrefix,
  parts,
}: {
  /** The attributes to mutate with new message attributes */
  attrs: Attributes;
  /** The prefix to add to the attributes */
  msgPrefix: string;
  /** The parts to process */
  parts: AnyPart[] | undefined;
}): void => {
  if (!Array.isArray(parts) || parts.length === 0) return;

  // track the index of the content and tool calls outside the loop
  // this is just in-case we have to skip bad parts
  let contentIndex = 0;
  let toolIndex = 0;

  for (const part of parts) {
    if (!part || typeof part !== "object") continue;
    if (!part.type) continue;

    switch (part.type) {
      case "text": {
        const text = toStringContent(part.content);
        if (text !== undefined) {
          // MESSAGE_CONTENTS entries
          const contentPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.`;
          set(attrs, `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`, "text");
          set(attrs, `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`, text);
          contentIndex += 1;
        }
        continue;
      }
      case "tool_call": {
        const id = part.id ?? undefined;
        const name = part.name;
        const args = part.arguments ?? {};
        const toolPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolIndex}.`;
        set(attrs, `${toolPrefix}${SemanticConventions.TOOL_CALL_ID}`, id);
        set(attrs, toolPrefix + SemanticConventions.TOOL_CALL_FUNCTION_NAME, name);
        set(
          attrs,
          toolPrefix + SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
          safelyJSONStringify(args),
        );
        toolIndex += 1;
        continue;
      }
      case "tool_call_response": {
        const id = part.id ?? undefined;
        const response = toStringContent(part.response);

        set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`, id);
        const contentPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.`;
        set(attrs, `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`, "text");
        set(attrs, `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`, response);
        contentIndex += 1;
        continue;
      }
      default: {
        // Generic / unknown part type: capture as JSON text content
        const genericPart = part as GenericPart;
        const genericText = toStringContent(genericPart);
        const contentPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.`;
        set(attrs, `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`, genericPart.type);
        set(attrs, `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`, genericText);
        set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_CONTENT}`, genericText);
        contentIndex += 1;
      }
    }
  }
};

/**
 * Convert GenAI span attributes to OpenInference span attributes
 * @param spanAttributes - The span attributes containing GenAI span attributes to convert
 * @returns The converted OpenInference span attributes
 */
export const convertGenAISpanAttributesToOpenInferenceSpanAttributes = (
  spanAttributes: Attributes,
): Attributes => {
  return convertGenAISpanToOpenInference({ attributes: spanAttributes });
};

/**
 * Convert a GenAI span to OpenInference span attributes.
 * @param span - Span-like input containing GenAI attributes and optional events
 * @param options - Conversion options for span kind and provider handling
 * @returns The converted OpenInference attributes
 */
export const convertGenAISpanToOpenInference = (
  span: GenAISpanLike,
  options: ConvertGenAISpanOptions = {},
): Attributes => {
  const events = span.events ?? [];
  return merge(
    mapProviderAndSystem(span.attributes, options),
    mapModels(span.attributes),
    mapSpanKind(span, options),
    mapInvocationParameters(span.attributes),
    mapInputMessages(span.attributes),
    mapOutputMessages(span.attributes),
    mapGenAIMessageEvents(events, {
      inputStartIndex: getGenAIMessageCount(span.attributes[ATTR_GEN_AI_INPUT_MESSAGES]),
      outputStartIndex: getGenAIMessageCount(span.attributes[ATTR_GEN_AI_OUTPUT_MESSAGES]),
    }),
    mapFinishReason(span.attributes, options),
    mapTokenCounts(span.attributes),
    mapToolExecution(span.attributes),
    mapInputValue(span.attributes),
    mapOutputValue(span.attributes),
  );
};

export const addOpenInferenceAttributesToSpan = (
  span: MutableGenAISpanLike,
  options: ConvertGenAISpanOptions = {},
): void => {
  const attributes = convertGenAISpanToOpenInference(span, options);
  Object.entries(attributes).forEach(([key, value]) => {
    span.attributes[key] = value;
  });
};

/**
 * Map provider and system to openinference attributes
 * @remarks TODO: add some heuristics that can map incoming provider names to the correct OpenInference provider name
 * @param spanAttributes - The span attributes containing provider and system to map
 * @returns The mapped provider and system attributes
 */
export const mapProviderAndSystem = (
  spanAttributes: Attributes,
  options: Pick<ConvertGenAISpanOptions, "providerMapping"> = {},
): Attributes => {
  const attrs: Attributes = {};
  const system = getString(spanAttributes[ATTR_GEN_AI_SYSTEM]);
  const provider = getString(spanAttributes[ATTR_GEN_AI_PROVIDER_NAME]);
  set(attrs, SemanticConventions.LLM_SYSTEM, system);
  set(attrs, SemanticConventions.LLM_PROVIDER, provider);
  if (provider == null && options.providerMapping === "system-as-provider-fallback") {
    set(attrs, SemanticConventions.LLM_PROVIDER, system);
  }
  return attrs;
};

/**
 * Map model name to openinference attributes
 * @param spanAttributes - The span attributes containing model name to map
 * @returns The mapped model name attributes
 */
export const mapModels = (spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  const requestModel = getString(spanAttributes[ATTR_GEN_AI_REQUEST_MODEL]);
  const responseModel = getString(spanAttributes[ATTR_GEN_AI_RESPONSE_MODEL]);
  const modelName = responseModel ?? requestModel;
  set(attrs, SemanticConventions.LLM_MODEL_NAME, modelName);
  return attrs;
};

/**
 * Map span kind to openinference attributes
 * @param spanAttributes - The span attributes containing span kind to map
 * @returns The mapped span kind attributes
 */
export const inferOpenInferenceSpanKindFromGenAI = (
  spanAttributes: Attributes,
): OpenInferenceSpanKind | undefined => {
  const attrs: Attributes = {};
  const operationName = getString(spanAttributes[ATTR_GEN_AI_OPERATION_NAME]);

  if (operationName === "execute_tool") {
    return OpenInferenceSpanKind.TOOL;
  }
  if (operationName === "embeddings") {
    return OpenInferenceSpanKind.EMBEDDING;
  }
  if (operationName === "create_agent" || operationName === "invoke_agent") {
    return OpenInferenceSpanKind.AGENT;
  }
  if (AGENT_KIND_PREFIXES.some((prefix) => spanAttributes[prefix])) {
    return OpenInferenceSpanKind.AGENT;
  }
  if (TOOL_EXECUTION_PREFIXES.some((prefix) => spanAttributes[prefix])) {
    return OpenInferenceSpanKind.TOOL;
  }
  if (operationName === "chat" || operationName === "generate_content") {
    return OpenInferenceSpanKind.LLM;
  }
  if (operationName === "text_completion") {
    return OpenInferenceSpanKind.LLM;
  }

  if (Object.keys(spanAttributes).some((key) => key.startsWith("gen_ai."))) {
    return OpenInferenceSpanKind.LLM;
  }

  return undefined;
};

export const mapSpanKind = (
  span: GenAISpanLike | Attributes,
  options: Pick<ConvertGenAISpanOptions, "spanKindResolver"> = {},
): Attributes => {
  const attrs: Attributes = {};
  const spanInput = isGenAISpanLike(span) ? span : { attributes: span };
  const events = spanInput.events ?? [];
  const defaultKind = inferOpenInferenceSpanKindFromGenAI(spanInput.attributes);
  const spanKind =
    options.spanKindResolver?.({
      name: spanInput.name,
      kind: spanInput.kind,
      attributes: spanInput.attributes,
      events,
      defaultKind,
    }) ?? defaultKind;

  set(attrs, SemanticConventions.OPENINFERENCE_SPAN_KIND, spanKind);

  return attrs;
};

export const mapFinishReason = (
  spanAttributes: Attributes,
  options: Pick<ConvertGenAISpanOptions, "finishReasonStrategy"> = {},
): Attributes => {
  const attrs: Attributes = {};
  const finishReasons = getStringArray(spanAttributes[ATTR_GEN_AI_RESPONSE_FINISH_REASONS]);
  if (finishReasons == null || finishReasons.length === 0) {
    return attrs;
  }

  const strategy = options.finishReasonStrategy ?? "first";
  const finishReason =
    strategy === "join"
      ? finishReasons.join(",")
      : strategy === "preserve-array-as-json"
        ? safelyJSONStringify(finishReasons)
        : finishReasons[0];
  set(attrs, "llm.finish_reason", finishReason);
  return attrs;
};

const isGenAISpanLike = (value: GenAISpanLike | Attributes): value is GenAISpanLike => {
  return typeof value.attributes === "object" && value.attributes != null;
};

/**
 * Map invocation parameters to openinference attributes
 * @param spanAttributes - The span attributes containing invocation parameters to map
 * @returns The mapped invocation parameters attributes
 */
export const mapInvocationParameters = (spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  const requestModel = getString(spanAttributes[ATTR_GEN_AI_REQUEST_MODEL]);
  const maxTokens = getNumber(spanAttributes[ATTR_GEN_AI_REQUEST_MAX_TOKENS]);
  const temperature = getNumber(spanAttributes[ATTR_GEN_AI_REQUEST_TEMPERATURE]);
  const topP = getNumber(spanAttributes[ATTR_GEN_AI_REQUEST_TOP_P]);
  const topK = getNumber(spanAttributes[ATTR_GEN_AI_REQUEST_TOP_K]);
  const presencePenalty = getNumber(spanAttributes[ATTR_GEN_AI_REQUEST_PRESENCE_PENALTY]);
  const frequencyPenalty = getNumber(spanAttributes[ATTR_GEN_AI_REQUEST_FREQUENCY_PENALTY]);
  const seed = getNumber(spanAttributes[ATTR_GEN_AI_REQUEST_SEED]);
  const stopSequences = getStringArray(spanAttributes[ATTR_GEN_AI_REQUEST_STOP_SEQUENCES]);
  const invocationParameters: Record<string, unknown> = {};
  if (requestModel) invocationParameters.model = requestModel;
  if (typeof temperature === "number") invocationParameters.temperature = temperature;
  if (typeof topP === "number") invocationParameters.top_p = topP;
  if (typeof topK === "number") invocationParameters.top_k = topK;
  if (typeof presencePenalty === "number") invocationParameters.presence_penalty = presencePenalty;
  if (typeof frequencyPenalty === "number")
    invocationParameters.frequency_penalty = frequencyPenalty;
  if (typeof seed === "number") invocationParameters.seed = seed;
  if (stopSequences && stopSequences.length > 0)
    invocationParameters.stop_sequences = stopSequences;
  if (typeof maxTokens === "number") invocationParameters.max_completion_tokens = maxTokens;
  if (Object.keys(invocationParameters).length > 0) {
    set(
      attrs,
      SemanticConventions.LLM_INVOCATION_PARAMETERS,
      safelyJSONStringify(invocationParameters),
    );
  }
  return attrs;
};

/**
 * Map input value to openinference attributes
 * @param spanAttributes - The span attributes containing input value to map
 * @returns The mapped input value attributes
 */
export const mapInputValue = (spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  let input = getString(spanAttributes["input"]);
  if (!input) {
    // fallback to deprecated prompt attribute if input is not present
    input = getString(spanAttributes[ATTR_GEN_AI_PROMPT]);
  }
  // only set input value and mime type if input is present
  if (input) {
    set(attrs, SemanticConventions.INPUT_VALUE, input);
    set(attrs, SemanticConventions.INPUT_MIME_TYPE, getMimeType(input));
  }
  return attrs;
};

/**
 * Map output value to openinference attributes
 * @param spanAttributes - The span attributes containing output value to map
 * @returns The mapped output value attributes
 */
export const mapOutputValue = (spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  let output = getString(spanAttributes["output"]);
  if (!output) {
    // fallback to deprecated completion attribute if output is not present
    output = getString(spanAttributes[ATTR_GEN_AI_COMPLETION]);
  }
  // only set output value and mime type if output is present
  if (output) {
    set(attrs, SemanticConventions.OUTPUT_VALUE, output);
    set(attrs, SemanticConventions.OUTPUT_MIME_TYPE, getMimeType(output));
  }
  return attrs;
};

/**
 * Map input messages to openinference attributes
 * @param spanAttributes - The span attributes containing input messages to map
 * @returns The mapped input messages attributes
 */
export const mapInputMessages = (spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  const genAIInputMessages = safelyParseJSON(spanAttributes[ATTR_GEN_AI_INPUT_MESSAGES]);

  if (Array.isArray(genAIInputMessages)) {
    (genAIInputMessages as unknown[]).forEach((msg, msgIndex) => {
      if (!isGenAIMessageLike(msg)) return;
      const msgPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${msgIndex}.`;
      // set the message role
      set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_ROLE}`, msg.role);
      // process and set the rest of the message parts
      processMessageParts({ attrs, msgPrefix, parts: msg.parts });
      if (msg.content != null) {
        set(
          attrs,
          `${msgPrefix}${SemanticConventions.MESSAGE_CONTENT}`,
          toStringContent(msg.content),
        );
        processMessageParts({
          attrs,
          msgPrefix,
          parts: [{ type: "text", content: msg.content } as AnyPart],
        });
      }
    });
  }

  return attrs;
};

/**
 * Map output messages to openinference attributes
 * @param spanAttributes - The span attributes containing output messages to map
 * @returns The mapped output messages attributes
 */
export const mapOutputMessages = (spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  const genAIOutputMessages = safelyParseJSON(spanAttributes[ATTR_GEN_AI_OUTPUT_MESSAGES]);

  if (Array.isArray(genAIOutputMessages) && genAIOutputMessages.length > 0) {
    // recast as unknown[] for safety, as Array.isArray() retypes to any[]
    (genAIOutputMessages as unknown[]).forEach((msg, msgIndex) => {
      if (!isGenAIMessageLike(msg)) return;
      const msgPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${msgIndex}.`;
      // set the message role
      set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_ROLE}`, msg.role);
      // process and set the rest of the message parts
      processMessageParts({ attrs, msgPrefix, parts: msg.parts });
      if (msg.content != null) {
        set(
          attrs,
          `${msgPrefix}${SemanticConventions.MESSAGE_CONTENT}`,
          toStringContent(msg.content),
        );
        processMessageParts({
          attrs,
          msgPrefix,
          parts: [{ type: "text", content: msg.content } as AnyPart],
        });
      }
    });
  }

  return attrs;
};

const getGenAIMessageCount = (value: unknown): number => {
  const messages = safelyParseJSON(value);
  if (!Array.isArray(messages)) {
    return 0;
  }
  return messages.filter(isGenAIMessageLike).length;
};

/**
 * Map usage token counts to openinference attributes
 * @param spanAttributes - The span attributes containing usage token counts to map
 * @returns The mapped usage token counts attributes
 */
export const mapTokenCounts = (spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  const inputTokens = getNumber(spanAttributes[ATTR_GEN_AI_USAGE_INPUT_TOKENS]);
  const outputTokens = getNumber(spanAttributes[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS]);
  if (typeof inputTokens === "number") {
    set(attrs, SemanticConventions.LLM_TOKEN_COUNT_PROMPT, inputTokens);
  }
  if (typeof outputTokens === "number") {
    set(attrs, SemanticConventions.LLM_TOKEN_COUNT_COMPLETION, outputTokens);
  }
  if (typeof inputTokens === "number" && typeof outputTokens === "number") {
    set(attrs, SemanticConventions.LLM_TOKEN_COUNT_TOTAL, inputTokens + outputTokens);
  }
  return attrs;
};

/**
 * Map tool execution to openinference attributes
 * @see https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span
 * @param spanAttributes - The span attributes containing tool execution to map
 * @returns The mapped tool execution attributes
 */
export const mapToolExecution = (spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  const toolName = getString(spanAttributes[ATTR_GEN_AI_TOOL_NAME]);
  const toolDescription = getString(spanAttributes[ATTR_GEN_AI_TOOL_DESCRIPTION]);
  const toolCallId = getString(spanAttributes[ATTR_GEN_AI_TOOL_CALL_ID]);
  // parse supported tool details
  // note: while openinference can track parameters, gen_ai does not provide this information
  set(attrs, SemanticConventions.TOOL_NAME, toolName);
  set(attrs, SemanticConventions.TOOL_DESCRIPTION, toolDescription);
  set(attrs, SemanticConventions.TOOL_ID, toolCallId);

  return attrs;
};

const getEventAttribute = (
  attributes: Attributes | undefined,
  keys: string[],
): AttributeValue | undefined => {
  if (attributes == null) return undefined;
  for (const key of keys) {
    const value = attributes[key];
    if (value != null) return value;
  }
  return undefined;
};

const setMessageContent = (attrs: Attributes, msgPrefix: string, content: unknown) => {
  const text = toStringContent(content);
  set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_CONTENT}`, text);
  const contentPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_CONTENTS}.0.`;
  set(attrs, `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`, "text");
  set(attrs, `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`, text);
};

const setMessageToolCalls = (attrs: Attributes, msgPrefix: string, toolCalls: unknown) => {
  const calls = Array.isArray(toolCalls) ? toolCalls : toolCalls == null ? [] : [toolCalls];
  calls.forEach((call, toolIndex) => {
    if (typeof call !== "object" || call == null) return;
    const callRecord = call as Record<string, unknown>;
    const functionRecord =
      typeof callRecord.function === "object" && callRecord.function != null
        ? (callRecord.function as Record<string, unknown>)
        : undefined;
    const id = getString(callRecord.id) ?? getString(callRecord.tool_call_id);
    const name =
      getString(callRecord.name) ??
      getString(callRecord.tool_name) ??
      getString(functionRecord?.name);
    const args =
      callRecord.arguments ?? callRecord.args ?? functionRecord?.arguments ?? functionRecord?.args;
    const toolPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolIndex}.`;
    set(attrs, `${toolPrefix}${SemanticConventions.TOOL_CALL_ID}`, id);
    set(attrs, `${toolPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`, name);
    set(
      attrs,
      `${toolPrefix}${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`,
      typeof args === "string" ? args : safelyJSONStringify(args),
    );
  });
};

const setMessageFromEvent = (params: {
  attrs: Attributes;
  prefix: string;
  index: number;
  role: string;
  event: GenAISpanEvent;
}) => {
  const { attrs, prefix, index, role, event } = params;
  const msgPrefix = `${prefix}.${index}.`;
  const eventAttributes = event.attributes;
  set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_ROLE}`, role);
  set(
    attrs,
    `${msgPrefix}${SemanticConventions.MESSAGE_NAME}`,
    getString(getEventAttribute(eventAttributes, ["name", "gen_ai.message.name", "tool.name"])),
  );
  set(
    attrs,
    `${msgPrefix}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`,
    getString(getEventAttribute(eventAttributes, ["tool_call_id", "gen_ai.tool.call.id"])),
  );

  const content = getEventAttribute(eventAttributes, ["content", "message", "text"]);
  if (content != null) {
    setMessageContent(attrs, msgPrefix, content);
  }
  setMessageToolCalls(attrs, msgPrefix, getEventAttribute(eventAttributes, ["tool_calls"]));
};

export const mapGenAIMessageEvents = (
  events: GenAISpanEvent[],
  options: { inputStartIndex?: number; outputStartIndex?: number } = {},
): Attributes => {
  const attrs: Attributes = {};
  let inputMessageIndex = options.inputStartIndex ?? 0;
  let outputMessageIndex = options.outputStartIndex ?? 0;

  for (const event of events) {
    const inputRole = INPUT_MESSAGE_EVENT_ROLES[event.name];
    if (inputRole != null) {
      setMessageFromEvent({
        attrs,
        prefix: SemanticConventions.LLM_INPUT_MESSAGES,
        index: inputMessageIndex,
        role: inputRole,
        event,
      });
      inputMessageIndex += 1;
      continue;
    }

    if (OUTPUT_MESSAGE_EVENT_NAMES.has(event.name)) {
      setMessageFromEvent({
        attrs,
        prefix: SemanticConventions.LLM_OUTPUT_MESSAGES,
        index: outputMessageIndex,
        role: "assistant",
        event,
      });
      outputMessageIndex += 1;
    }
  }

  return attrs;
};
