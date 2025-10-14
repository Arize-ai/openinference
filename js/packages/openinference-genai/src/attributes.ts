import { Attributes } from "@opentelemetry/api";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { safelyJSONStringify } from "@arizeai/openinference-core";
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
} from "@opentelemetry/semantic-conventions/incubating";

type GenAIMessagePart =
  | { type: "text"; content: string }
  | {
      type: "tool_call";
      id?: string;
      name: string;
      arguments: unknown;
    }
  | {
      type: "tool_call_response";
      id?: string;
      response: string;
    };

type GenAIMessage = {
  role: "user" | "assistant" | "tool";
  parts: GenAIMessagePart[];
  finish_reason?: string;
  /** @deprecated use parts instead */
  text?: string;
};

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

const set = (attrs: Attributes, key: string, value: unknown) => {
  if (value === undefined || value === null) return;
  attrs[key] = value as never;
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
  set(
    attrs,
    SemanticConventions.OPENINFERENCE_SPAN_KIND,
    OpenInferenceSpanKind.LLM,
  );
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
  let genAIInputMessages = parseJSON<GenAIMessage[]>(
    spanAttributes[ATTR_GEN_AI_INPUT_MESSAGES],
  );
  if (!genAIInputMessages) {
    // fallback to deprecated prompt attribute if input messages are not present
    genAIInputMessages = parseJSON<GenAIMessage[]>(
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
      const msgPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${msgIndex}.`;
      set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_ROLE}`, msg.role);
      if (msg.role === "user") {
        const textPart = msg.parts.find((p) => p.type === "text") as
          | Extract<GenAIMessagePart, { type: "text" }>
          | undefined;
        if (textPart) {
          set(
            attrs,
            `${msgPrefix}${SemanticConventions.MESSAGE_CONTENT}`,
            textPart.content,
          );
          canonicalInputMessages.push({
            role: "user",
            content: textPart.content,
          });
        }
      } else if (msg.role === "assistant") {
        const toolCalls: unknown[] = [];
        msg.parts.forEach((p, toolIndex) => {
          if (p.type !== "tool_call") return;
          const toolPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_TOOL_CALLS}.${toolIndex}.`;
          if (p.id)
            set(
              attrs,
              `${toolPrefix}${SemanticConventions.TOOL_CALL_ID}`,
              p.id,
            );
          set(
            attrs,
            toolPrefix + SemanticConventions.TOOL_CALL_FUNCTION_NAME,
            p.name,
          );
          set(
            attrs,
            toolPrefix + SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
            safelyJSONStringify(p.arguments),
          );
          toolCalls.push({
            id: p.id,
            type: "function",
            function: {
              name: p.name,
              arguments: safelyJSONStringify(p.arguments),
            },
          });
        });
        canonicalInputMessages.push({
          role: "assistant",
          tool_calls: toolCalls,
        });
      } else if (msg.role === "tool") {
        const responsePart = msg.parts.find(
          (p) => p.type === "tool_call_response",
        ) as
          | Extract<GenAIMessagePart, { type: "tool_call_response" }>
          | undefined;
        if (responsePart) {
          set(
            attrs,
            `${msgPrefix}${SemanticConventions.MESSAGE_TOOL_CALL_ID}`,
            responsePart.id,
          );
          const contentPrefix = `${msgPrefix}${SemanticConventions.MESSAGE_CONTENTS}.0.`;
          set(
            attrs,
            `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TYPE}`,
            "text",
          );
          set(
            attrs,
            `${contentPrefix}${SemanticConventions.MESSAGE_CONTENT_TEXT}`,
            responsePart.response,
          );
          canonicalInputMessages.push({
            role: "tool",
            tool_call_id: responsePart.id,
            content: [{ type: "text", text: responsePart.response }],
          });
        }
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
  }
  return attrs;
};

// Output messages + output.value
export const mapOutputMessagesAndOutputValue = (
  spanAttributes: Attributes,
): Attributes => {
  const attrs: Attributes = {};
  let genAIOutputMessages = parseJSON<GenAIMessage[] | GenAIMessage>(
    spanAttributes[ATTR_GEN_AI_OUTPUT_MESSAGES],
  );
  if (!genAIOutputMessages) {
    // fallback to deprecated completion attribute if output messages are not present
    genAIOutputMessages = parseJSON<GenAIMessage>(
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
    const outputValue: GenAIMessage = {
      role: "assistant",
      parts: [{ type: "text", content: genAIOutputMessages.text ?? "" }],
      finish_reason: finishReason,
    };
    genAIOutputMessages = [outputValue];
  }

  if (Array.isArray(genAIOutputMessages) && genAIOutputMessages.length > 0) {
    const first = genAIOutputMessages[0];
    const msgPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.`;
    set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_ROLE}`, first.role);
    const content = first.parts?.find((p) => p.type === "text")?.content;
    if (content) {
      set(attrs, `${msgPrefix}${SemanticConventions.MESSAGE_CONTENT}`, content);
    }
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
