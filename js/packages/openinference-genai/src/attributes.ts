import type { Attributes } from "@opentelemetry/api";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  ATTR_GEN_AI_PROVIDER_NAME,
  ATTR_GEN_AI_REQUEST_MODEL,
  ATTR_GEN_AI_RESPONSE_MODEL,
  ATTR_GEN_AI_REQUEST_MAX_TOKENS,
  ATTR_GEN_AI_REQUEST_TEMPERATURE,
  ATTR_GEN_AI_REQUEST_TOP_P,
  ATTR_GEN_AI_REQUEST_TOP_K,
  ATTR_GEN_AI_REQUEST_PRESENCE_PENALTY,
  ATTR_GEN_AI_REQUEST_FREQUENCY_PENALTY,
  ATTR_GEN_AI_REQUEST_STOP_SEQUENCES,
  ATTR_GEN_AI_REQUEST_SEED,
  ATTR_GEN_AI_INPUT_MESSAGES,
  ATTR_GEN_AI_OUTPUT_MESSAGES,
  ATTR_GEN_AI_RESPONSE_ID,
  ATTR_GEN_AI_RESPONSE_FINISH_REASONS,
  ATTR_GEN_AI_USAGE_INPUT_TOKENS,
  ATTR_GEN_AI_USAGE_OUTPUT_TOKENS,
  ATTR_GEN_AI_PROMPT,
  ATTR_GEN_AI_COMPLETION,
  ATTR_GEN_AI_AGENT_ID,
  ATTR_GEN_AI_AGENT_NAME,
  ATTR_GEN_AI_AGENT_DESCRIPTION,
  ATTR_GEN_AI_TOOL_NAME,
  ATTR_GEN_AI_TOOL_DESCRIPTION,
  ATTR_GEN_AI_TOOL_CALL_ID,
  ATTR_GEN_AI_TOOL_TYPE,
} from "@opentelemetry/semantic-conventions/incubating";

import { safelyJSONStringify } from "./utils.js";
import type { ChatMessage } from "./schemas/opentelemetryInputMessages.js";
import type { OutputMessage } from "./schemas/opentelemetryOutputMessages.js";

export type GenAIInputMessage = ChatMessage;
export type GenAIInputMessagePart = ChatMessage["parts"][number];
export type GenAIOutputMessage = OutputMessage & {
  /** @deprecated use parts instead */
  text?: string;
};
export type GenAIOutputMessagePart = OutputMessage["parts"][number];

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

const getNumber = (value: unknown): number | undefined => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  return undefined;
};

const getString = (value: unknown): string | undefined => {
  if (typeof value === "string" && value.length > 0) return value;
  return undefined;
};

const getStringArray = (value: unknown): string[] | undefined => {
  if (Array.isArray(value) && value.every((v) => typeof v === "string")) {
    return value as string[];
  }
  return undefined;
};

const parseJSON = <T = unknown>(value: unknown): T | undefined => {
  const s = getString(value);
  if (!s) return undefined;
  try {
    return JSON.parse(s) as T;
  } catch {
    return undefined;
  }
};

const getMimeType = (value: unknown): string | undefined => {
  if (parseJSON(value)) return MimeType.JSON;
  return MimeType.TEXT;
};

const set = (attrs: Attributes, key: string, value: unknown) => {
  if (value === undefined || value === null) return;
  attrs[key] = value as never;
};

// Shared part parsing
type AnyPart = GenAIInputMessagePart | GenAIOutputMessagePart;

const toStringContent = (value: unknown): string => {
  if (typeof value === "string") return value;
  const json = safelyJSONStringify(value);
  if (typeof json === "string") return json;
  return String(value);
};

interface ProcessedParts {
  textContents: string[];
  toolCallsForCanonical: unknown[];
  toolCallResponse?: { id?: string; texts: string[] };
}

// TODO: solve duplicate content issues
// I should not persist content and contents in the same message
const processMessageParts = (
  attrs: Attributes,
  msgPrefix: string,
  parts: AnyPart[] | undefined,
): ProcessedParts => {
  const result: ProcessedParts = {
    textContents: [],
    toolCallsForCanonical: [],
  };
  if (!Array.isArray(parts) || parts.length === 0) return result;

  let contentIndex = 0;
  let toolIndex = 0;

  // early return if there is only one text part
  // just use the content attribute instead of the contents array
  // if (parts.length === 1 && parts[0]?.type === "text") {
  //   const text = toStringContent(parts[0].content);
  //   if (text !== undefined) {
  //     set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_CONTENT}`, text);
  //     result.textContents.push(text);
  //     return result;
  //   }
  // }

  for (const part of parts) {
    if (!part || typeof part !== "object") continue;
    const p = part as Record<string, unknown>;
    const type = p["type"] as string | undefined;
    if (!type) continue;

    if (type === "text") {
      const text = toStringContent(p.content);
      if (text !== undefined) {
        // MESSAGE_CONTENTS entries
        const contentPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.`;
        set(
          attrs,
          `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
          "text",
        );
        set(
          attrs,
          `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`,
          text,
        );
        result.textContents.push(text);
        contentIndex += 1;
      }
      continue;
    }

    if (type === "tool_call") {
      const id = (p["id"] as string | null) ?? undefined;
      const name = p["name"] as string | undefined;
      const args = (p["arguments"] ?? {}) as Record<string, unknown>;
      const toolPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolIndex}.`;
      if (id)
        set(attrs, `${toolPrefix}${SemanticConventions.TOOL_CALL_ID}`, id);
      if (name)
        set(
          attrs,
          toolPrefix + SemanticConventions.TOOL_CALL_FUNCTION_NAME,
          name,
        );
      set(
        attrs,
        toolPrefix + SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
        safelyJSONStringify(args),
      );
      result.toolCallsForCanonical.push({
        id,
        type: "function",
        function: { name: name ?? "", arguments: safelyJSONStringify(args) },
      });
      toolIndex += 1;
      continue;
    }

    if (type === "tool_call_response") {
      const id = (p["id"] as string | null) ?? undefined;
      const response = toStringContent((p as { response?: unknown }).response);
      if (id)
        set(
          attrs,
          `${msgPrefix}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`,
          id,
        );
      const contentPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.`;
      set(
        attrs,
        `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
        "text",
      );
      set(
        attrs,
        `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`,
        response,
      );
      if (!result.toolCallResponse) {
        result.toolCallResponse = { id, texts: [response] };
      } else {
        result.toolCallResponse.texts.push(response);
      }
      result.textContents.push(response);
      contentIndex += 1;
      continue;
    }

    // Generic / unknown part type: capture as JSON text content
    const genericText = toStringContent(part);
    const contentPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_CONTENTS}.${contentIndex}.`;
    set(
      attrs,
      `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
      type,
    );
    set(
      attrs,
      `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`,
      genericText,
    );
    if (result.textContents.length === 0) {
      set(
        attrs,
        `${msgPrefix}${SemanticConventions.MESSAGE_CONTENT}`,
        genericText,
      );
    }
    result.textContents.push(genericText);
    contentIndex += 1;
  }

  return result;
};

export const convertGenAISpanAttributesToOpenInferenceSpanAttributes = (
  spanAttributes: Attributes,
): Attributes => {
  const merge = (...groups: Attributes[]): Attributes =>
    groups.reduce((acc, g) => Object.assign(acc, g), {} as Attributes);

  const mapped = merge(
    mapProviderAndSystem(spanAttributes),
    mapModels(spanAttributes),
    mapSpanKind(spanAttributes),
    mapInvocationParameters(spanAttributes),
    mapInputMessagesAndInputValue(spanAttributes),
    mapOutputMessagesAndOutputValue(spanAttributes),
    mapTokenCounts(spanAttributes),
    mapToolExecution(spanAttributes),
  );
  return mapped;
};

// Provider/system
export const mapProviderAndSystem = (
  spanAttributes: Attributes,
): Attributes => {
  const attrs: Attributes = {};
  const provider = getString(spanAttributes[ATTR_GEN_AI_PROVIDER_NAME]);
  if (provider) {
    set(attrs, SemanticConventions.LLM_SYSTEM, provider);
    set(attrs, SemanticConventions.LLM_PROVIDER, provider);
  }
  return attrs;
};

// Model name
export const mapModels = (spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  const requestModel = getString(spanAttributes[ATTR_GEN_AI_REQUEST_MODEL]);
  const responseModel = getString(spanAttributes[ATTR_GEN_AI_RESPONSE_MODEL]);
  const modelName = responseModel ?? requestModel;
  if (modelName) {
    set(attrs, SemanticConventions.LLM_MODEL_NAME, modelName);
  }
  return attrs;
};

// Span kind
export const mapSpanKind = (_spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  // default to LLM for now
  let spanKind = OpenInferenceSpanKind.LLM;
  // detect agent kind
  if (AGENT_KIND_PREFIXES.some((prefix) => _spanAttributes[prefix])) {
    spanKind = OpenInferenceSpanKind.AGENT;
  }
  // detect tool execution kind
  if (TOOL_EXECUTION_PREFIXES.some((prefix) => _spanAttributes[prefix])) {
    spanKind = OpenInferenceSpanKind.TOOL;
  }

  set(attrs, SemanticConventions.OPENINFERENCE_SPAN_KIND, spanKind);

  return attrs;
};

// Invocation parameters
export const mapInvocationParameters = (
  spanAttributes: Attributes,
  prefix: string = SemanticConventions.LLM_INVOCATION_PARAMETERS,
): Attributes => {
  const attrs: Attributes = {};
  const requestModel = getString(spanAttributes[ATTR_GEN_AI_REQUEST_MODEL]);
  const maxTokens = getNumber(spanAttributes[ATTR_GEN_AI_REQUEST_MAX_TOKENS]);
  const temperature = getNumber(
    spanAttributes[ATTR_GEN_AI_REQUEST_TEMPERATURE],
  );
  const topP = getNumber(spanAttributes[ATTR_GEN_AI_REQUEST_TOP_P]);
  const topK = getNumber(spanAttributes[ATTR_GEN_AI_REQUEST_TOP_K]);
  const presencePenalty = getNumber(
    spanAttributes[ATTR_GEN_AI_REQUEST_PRESENCE_PENALTY],
  );
  const frequencyPenalty = getNumber(
    spanAttributes[ATTR_GEN_AI_REQUEST_FREQUENCY_PENALTY],
  );
  const seed = getNumber(spanAttributes[ATTR_GEN_AI_REQUEST_SEED]);
  const stopSequences = getStringArray(
    spanAttributes[ATTR_GEN_AI_REQUEST_STOP_SEQUENCES],
  );
  const invocationParameters: Record<string, unknown> = {};
  if (requestModel) invocationParameters.model = requestModel;
  if (typeof temperature === "number")
    invocationParameters.temperature = temperature;
  if (typeof topP === "number") invocationParameters.top_p = topP;
  if (typeof topK === "number") invocationParameters.top_k = topK;
  if (typeof presencePenalty === "number")
    invocationParameters.presence_penalty = presencePenalty;
  if (typeof frequencyPenalty === "number")
    invocationParameters.frequency_penalty = frequencyPenalty;
  if (typeof seed === "number") invocationParameters.seed = seed;
  if (stopSequences && stopSequences.length > 0)
    invocationParameters.stop_sequences = stopSequences;
  if (typeof maxTokens === "number")
    invocationParameters.max_completion_tokens = maxTokens;
  if (Object.keys(invocationParameters).length > 0) {
    set(attrs, prefix, safelyJSONStringify(invocationParameters));
  }
  return attrs;
};

// Input messages + input.value
export const mapInputMessagesAndInputValue = (
  spanAttributes: Attributes,
): Attributes => {
  const attrs: Attributes = {};
  let genAIInputMessages = parseJSON<GenAIInputMessage[]>(
    spanAttributes[ATTR_GEN_AI_INPUT_MESSAGES],
  );
  if (!genAIInputMessages) {
    // fallback to deprecated prompt attribute if input messages are not present
    genAIInputMessages = parseJSON<GenAIInputMessage[]>(
      spanAttributes[ATTR_GEN_AI_PROMPT],
    );
  }
  // input value includes invocation parameters
  let invocationParameters = mapInvocationParameters(
    spanAttributes,
    "inputValue",
  );
  // deconstruct the stringified invocation parameters into flat keys
  if (invocationParameters["inputValue"]) {
    const parsed = parseJSON(invocationParameters["inputValue"]);
    if (parsed) {
      invocationParameters = parsed as Attributes;
    }
  }

  const canonicalInputMessages: unknown[] = [];
  if (Array.isArray(genAIInputMessages)) {
    genAIInputMessages.forEach((msg, msgIndex) => {
      if (!msg.parts || !Array.isArray(msg.parts)) return;
      const msgPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${msgIndex}.`;
      set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_ROLE}`, msg.role);
      const processed = processMessageParts(
        attrs,
        msgPrefix,
        msg.parts as AnyPart[],
      );

      // Build canonical message for inputValue
      const base: Record<string, unknown> = { role: msg.role };
      if (msg.role === "tool" && processed.toolCallResponse?.id) {
        base["tool_call_id"] = processed.toolCallResponse.id;
      }
      if (processed.textContents.length > 0) {
        if (msg.role === "tool") {
          base["content"] = processed.textContents.map((t) => ({
            type: "text",
            text: t,
          }));
        } else {
          base["content"] = processed.textContents.join("\n");
        }
      }
      if (
        processed.toolCallsForCanonical.length > 0 &&
        msg.role === "assistant"
      ) {
        base["tool_calls"] = processed.toolCallsForCanonical;
      }
      if (Object.keys(base).length > 1) {
        canonicalInputMessages.push(base);
      }
    });
  }

  if (canonicalInputMessages.length > 0) {
    const inputValue: Record<string, unknown> = {
      ...invocationParameters,
      messages: canonicalInputMessages,
    };
    set(
      attrs,
      SemanticConventions.INPUT_VALUE,
      safelyJSONStringify(inputValue),
    );
    set(attrs, SemanticConventions.INPUT_MIME_TYPE, MimeType.JSON);
  } else if (genAIInputMessages) {
    // we could not parse out input messages, just jsonify the input value
    set(
      attrs,
      SemanticConventions.INPUT_VALUE,
      safelyJSONStringify(genAIInputMessages),
    );
    set(attrs, SemanticConventions.INPUT_MIME_TYPE, MimeType.JSON);
  }
  return attrs;
};

// Output messages + output.value
export const mapOutputMessagesAndOutputValue = (
  spanAttributes: Attributes,
): Attributes => {
  const attrs: Attributes = {};
  let genAIOutputMessages = parseJSON<
    GenAIOutputMessage[] | GenAIOutputMessage
  >(spanAttributes[ATTR_GEN_AI_OUTPUT_MESSAGES]);
  if (!genAIOutputMessages) {
    // fallback to deprecated completion attribute if output messages are not present
    genAIOutputMessages = parseJSON<GenAIOutputMessage>(
      spanAttributes[ATTR_GEN_AI_COMPLETION],
    );
  }

  const responseId = getString(spanAttributes[ATTR_GEN_AI_RESPONSE_ID]);
  const responseModel = getString(spanAttributes[ATTR_GEN_AI_RESPONSE_MODEL]);
  const requestModel = getString(spanAttributes[ATTR_GEN_AI_REQUEST_MODEL]);
  const modelName = responseModel ?? requestModel;
  const finishReasons = spanAttributes[ATTR_GEN_AI_RESPONSE_FINISH_REASONS] as
    | string[]
    | undefined;
  const finishReason =
    Array.isArray(finishReasons) && finishReasons.length > 0
      ? finishReasons[0]
      : undefined;

  // handle the deprecated completion attribute
  // if we have a single output completion message, simulate an array of messages
  if (!Array.isArray(genAIOutputMessages) && genAIOutputMessages) {
    const outputValue: GenAIOutputMessage = {
      role: "assistant",
      parts: [{ type: "text", content: genAIOutputMessages.text ?? "" }],
      finish_reason: finishReason ?? "",
    };
    genAIOutputMessages = [outputValue];
  }

  if (Array.isArray(genAIOutputMessages) && genAIOutputMessages.length > 0) {
    const first = genAIOutputMessages[0];
    if (!first) return attrs;
    const msgPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.`;
    set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_ROLE}`, first.role);
    const processed = processMessageParts(
      attrs,
      msgPrefix,
      first.parts as AnyPart[],
    );
    const content =
      processed.textContents.length > 0
        ? processed.textContents.join("\n")
        : undefined;
    const outputValue: Record<string, unknown> = {
      id: responseId,
      model: modelName,
      choices: [
        {
          index: 0,
          message: { role: first.role, content },
          finish_reason: finishReason,
        },
      ],
    };
    set(
      attrs,
      SemanticConventions.OUTPUT_VALUE,
      safelyJSONStringify(outputValue),
    );
    set(attrs, SemanticConventions.OUTPUT_MIME_TYPE, MimeType.JSON);
  } else if (genAIOutputMessages) {
    // we could not parse out output messages, just jsonify the output value
    set(
      attrs,
      SemanticConventions.OUTPUT_VALUE,
      safelyJSONStringify(genAIOutputMessages),
    );
    set(attrs, SemanticConventions.OUTPUT_MIME_TYPE, MimeType.JSON);
  }
  return attrs;
};

// Token counts
export const mapTokenCounts = (spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  const inputTokens = getNumber(spanAttributes[ATTR_GEN_AI_USAGE_INPUT_TOKENS]);
  const outputTokens = getNumber(
    spanAttributes[ATTR_GEN_AI_USAGE_OUTPUT_TOKENS],
  );
  if (typeof inputTokens === "number") {
    set(attrs, SemanticConventions.LLM_TOKEN_COUNT_PROMPT, inputTokens);
  }
  if (typeof outputTokens === "number") {
    set(attrs, SemanticConventions.LLM_TOKEN_COUNT_COMPLETION, outputTokens);
  }
  if (typeof inputTokens === "number" && typeof outputTokens === "number") {
    set(
      attrs,
      SemanticConventions.LLM_TOKEN_COUNT_TOTAL,
      inputTokens + outputTokens,
    );
  }
  return attrs;
};

// Tool execution
// https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/#execute-tool-span
export const mapToolExecution = (spanAttributes: Attributes): Attributes => {
  const attrs: Attributes = {};
  const toolName = getString(spanAttributes[ATTR_GEN_AI_TOOL_NAME]);
  const toolDescription = getString(
    spanAttributes[ATTR_GEN_AI_TOOL_DESCRIPTION],
  );
  const toolCallId = getString(spanAttributes[ATTR_GEN_AI_TOOL_CALL_ID]);
  // parse supported tool details
  // note: while openinference can track parameters, gen_ai does not provide this information
  if (toolName) {
    set(attrs, SemanticConventions.TOOL_NAME, toolName);
  }
  if (toolDescription) {
    set(attrs, SemanticConventions.TOOL_DESCRIPTION, toolDescription);
  }
  if (toolCallId) {
    set(attrs, SemanticConventions.TOOL_CALL_ID, toolCallId);
  }
  // parse input and output with mime type
  const input = getString(spanAttributes["input"]);
  const output = getString(spanAttributes["output"]);
  if (input) {
    set(attrs, SemanticConventions.INPUT_VALUE, input);
    set(attrs, SemanticConventions.INPUT_MIME_TYPE, getMimeType(input));
  }
  if (output) {
    set(attrs, SemanticConventions.OUTPUT_VALUE, output);
    set(attrs, SemanticConventions.OUTPUT_MIME_TYPE, getMimeType(output));
  }

  return attrs;
};
