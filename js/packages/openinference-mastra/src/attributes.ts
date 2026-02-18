import {
  safelyJSONParse,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import { addOpenInferenceAttributesToSpan as addOpenInferenceAttributesToSpanVercel } from "@arizeai/openinference-vercel/utils";

import { diag } from "@opentelemetry/api";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";

// openinference-vercel is typed against OTel v1 ReadableSpan, while Mastra uses OTel v2.
// At runtime we only need span.attributes, so keep the call but erase the type.
const addOpenInferenceAttributesToSpan =
  addOpenInferenceAttributesToSpanVercel as unknown as (span: {
    attributes: Record<string, unknown>;
  }) => void;

import {
  MASTRA_AGENT_SPAN_NAME_PREFIXES,
  MASTRA_INTERNAL_SPAN_NAME_PREFIX,
  MastraSpanAttributes,
  MastraSpanNames,
} from "./constants.js";
import { addOpenInferenceProjectResourceAttributeSpan } from "./utils.js";

/**
 * Add the OpenInference span kind to the given Mastra span.
 *
 * This function will add the OpenInference span kind to the given Mastra span.
 */
const addOpenInferenceSpanKind = (
  span: ReadableSpan,
  kind: OpenInferenceSpanKind,
) => {
  span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] = kind;
};

/**
 * Add the SessionId to the given Mastra span
 */
const addSessionId = (span: ReadableSpan, sessionId: string) => {
  span.attributes[SemanticConventions.SESSION_ID] = sessionId;
};

/**
 * Get the OpenInference span kind for the given Mastra span.
 *
 * This function will return the OpenInference span kind for the given Mastra span, if it has already been set.
 */
const getOpenInferenceSpanKind = (span: ReadableSpan) => {
  return span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] as
    | OpenInferenceSpanKind
    | undefined;
};

/**
 * Determines whether a span represents an agent operation.
 *
 * @param span - The span to check.
 * @returns `true` if the span is an agent operation, `false` otherwise.
 */
export const isAgentOperation = (span: ReadableSpan): boolean => {
  const spanName = span.name.toLowerCase();
  return !!MASTRA_AGENT_SPAN_NAME_PREFIXES.some((prefix) =>
    spanName.startsWith(prefix),
  );
};

/**
 * Get the closest OpenInference span kind for the given Mastra span.
 *
 * This function will attempt to detect the closest OpenInference span kind for the given Mastra span,
 * based on the span's name and parent span ID.
 */
const getOpenInferenceSpanKindFromMastraSpan = (
  span: ReadableSpan,
): OpenInferenceSpanKind | null => {
  // We prefer the mastra operation over the openinference span kind
  if (isAgentOperation(span)) {
    return OpenInferenceSpanKind.AGENT;
  }
  // Fallback to existing OpenInference span kind if set
  const oiKind = getOpenInferenceSpanKind(span);
  if (oiKind) {
    return oiKind;
  }
  return null;
};

/**
 * Gets the Mastra thread ID or session ID from the attributes
 */
const getSessionIdFromMastraSpan = (span: ReadableSpan): string | null => {
  const sessionId = span.attributes[SemanticConventions.SESSION_ID];
  if (typeof sessionId === "number" || typeof sessionId === "string") {
    // We coerce to string. See https://opentelemetry.io/docs/specs/semconv/general/session/
    return String(sessionId);
  }
  // Map Mastra session ID
  const threadId = span.attributes.threadId;
  if (typeof threadId === "number" || typeof threadId === "string") {
    return String(threadId);
  }
  return null;
};

/**
 * Enrich a Mastra span with OpenInference attributes.
 *
 * This function will add additional attributes to the span, based on the Mastra span's attributes.
 *
 * It will attempt to detect the closest OpenInference span kind for the given Mastra span, and then
 * enrich the span with the appropriate attributes based on the span kind and current attributes.
 *
 * @param span - The Mastra span to enrich.
 * @param shouldMarkAsAgent - Whether this span should be marked as an AGENT span
 */
export const addOpenInferenceAttributesToMastraSpan = (span: ReadableSpan) => {
  const kind = getOpenInferenceSpanKindFromMastraSpan(span);
  if (kind) {
    addOpenInferenceSpanKind(span, kind);
  }
  const sessionId = getSessionIdFromMastraSpan(span);
  if (typeof sessionId === "string") {
    addSessionId(span, sessionId);
  }
};

/**
 * Processes a span with all standard Mastra OpenInference attributes.
 *
 * This function applies the complete set of OpenInference attribute processing:
 * - Adds project resource attributes
 * - Adds Vercel-compatible OpenInference attributes
 * - Adds Mastra-specific OpenInference attributes
 *
 * @param span - The span to process.
 */
export const processMastraSpanAttributes = (span: ReadableSpan): void => {
  addOpenInferenceProjectResourceAttributeSpan(span);
  addOpenInferenceAttributesToSpan(
    span as unknown as { attributes: Record<string, unknown> },
  );
  addOpenInferenceAttributesToMastraSpan(span);
};

/**
 * Gets the span ID from a ReadableSpan.
 *
 * @param span - The span to get the ID from.
 * @returns The span ID or "unknown" if not available.
 */
export const getSpanId = (span: ReadableSpan): string | null => {
  return span.spanContext?.()?.spanId || null;
};

/**
 * Gets the trace ID from a ReadableSpan.
 *
 * @param span - The span to get the trace ID from.
 * @returns The trace ID or "unknown" if not available.
 */
export const getTraceId = (span: ReadableSpan): string | null => {
  return span.spanContext?.()?.traceId || null;
};

/**
 * Type definition for a chat message with role and content
 */
interface ChatMessage {
  role: string;
  content: string | object;
}

/**
 * Type guard for a chat message
 */
const isChatMessage = (message: unknown): message is ChatMessage => {
  return (
    typeof message === "object" &&
    message != null &&
    "role" in message &&
    typeof message.role === "string" &&
    "content" in message &&
    (typeof message.content === "string" || typeof message.content === "object")
  );
};

/**
 * Type guard for an array of chat messages
 */
const isChatMessages = (messages: unknown): messages is ChatMessage[] => {
  return Array.isArray(messages) && messages.every(isChatMessage);
};

/**
 * Extracts the last user message from an array of chat messages.
 *
 * @param messages - Array of chat messages to search through
 * @returns The content of the last user message as a string, or undefined if no user message is found
 */
const extractLastUserMessage = (
  messages: ChatMessage[],
): string | undefined => {
  // Find the last user message
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role === "user" && message.content) {
      return typeof message.content === "string"
        ? message.content
        : JSON.stringify(message.content);
    }
  }
};

/**
 * Extracts user input from Mastra spans for session I/O.
 *
 * Looks for user input in the following order:
 * 1. agent.getMostRecentUserMessage.result attribute
 * 2. agent.generate.argument.0 (direct user input)
 * 3. agent.stream.argument.0 conversation messages (last user message)
 *
 * @param spans - Array of spans to search through from a given trace.
 * @returns The extracted user input or undefined if not found.
 */
export const extractMastraUserInput = (
  spans: ReadableSpan[],
): string | null | undefined => {
  for (const span of spans) {
    switch (span.name) {
      case MastraSpanNames.AGENT_GET_RECENT_MESSAGE: {
        const result =
          span.attributes[MastraSpanAttributes.AGENT_GET_RECENT_MESSAGE_RESULT];
        if (typeof result === "string") {
          const messageData = safelyJSONParse(result);
          if (
            typeof messageData === "object" &&
            messageData !== null &&
            "content" in messageData &&
            typeof messageData.content === "string"
          ) {
            return messageData.content;
          }
          // We could log a warning here, but avoiding for now
        }
        break;
      }

      case MastraSpanNames.AGENT_GENERATE: {
        const argument =
          span.attributes[MastraSpanAttributes.AGENT_GENERATE_ARGUMENT];
        if (typeof argument === "string") {
          const parsedArgument = safelyJSONParse(argument);
          if (parsedArgument === null) {
            // If the argument is not a valid JSON string, return the raw string
            return argument;
          }
          // Convert parsed argument to string for consistent return type
          return typeof parsedArgument === "string"
            ? parsedArgument
            : safelyJSONStringify(parsedArgument);
        }
        break;
      }

      case MastraSpanNames.AGENT_STREAM: {
        const argument =
          span.attributes[MastraSpanAttributes.AGENT_STREAM_ARGUMENT];
        if (typeof argument === "string") {
          const messages = safelyJSONParse(argument);
          if (messages === null) {
            diag.warn("Failed to parse agent.stream.argument.0", {
              rawArgument: argument,
              spanId: span.spanContext?.()?.spanId,
            });
            continue;
          }
          if (isChatMessages(messages)) {
            const lastUserMessage = extractLastUserMessage(messages);
            if (lastUserMessage) {
              return lastUserMessage;
            }
          }
        }
        break;
      }
    }
  }
};

/**
 * Extracts agent output from Mastra spans for session I/O.
 *
 * Looks for output.value in any span (typically ai.streamText or similar).
 *
 * @param spans - Array of spans to search through.
 * @returns The extracted agent output or undefined if not found.
 */
export const extractMastraAgentOutput = (
  spans: ReadableSpan[],
): string | undefined => {
  // Look for output.value in any span (typically ai.streamText or similar)
  for (const span of spans) {
    const outputValue = span.attributes[SemanticConventions.OUTPUT_VALUE];
    if (typeof outputValue === "string") {
      return outputValue;
    }
  }
};

/**
 * Adds input and output attributes to root spans for session I/O support.
 *
 * Extracts user input and agent output from the provided spans and adds them
 * to any root spans that don't already have I/O attributes set. Handles spans
 * from different traces correctly by grouping spans by trace ID.
 *
 * This function modifies spans in place for memory efficiency.
 *
 * @param spans - Array of spans to process.
 */
export const addIOToRootSpans = (spans: ReadableSpan[]): void => {
  const rootSpans = spans.filter(
    (span) => span.parentSpanContext === undefined,
  );

  if (!rootSpans.length) {
    return;
  }

  // Group spans by trace ID
  const spansByTrace = new Map<string, ReadableSpan[]>();
  for (const span of spans) {
    const traceId = getTraceId(span);
    if (traceId) {
      let traceSpans = spansByTrace.get(traceId);
      if (!traceSpans) {
        traceSpans = [];
        spansByTrace.set(traceId, traceSpans);
      }
      traceSpans.push(span);
    }
  }

  // Process each trace separately
  for (const rootSpan of rootSpans) {
    const rootTraceId = getTraceId(rootSpan);
    if (!rootTraceId) {
      continue;
    }

    const traceSpans = spansByTrace.get(rootTraceId);
    if (!traceSpans) {
      continue;
    }

    const userInput = extractMastraUserInput(traceSpans);
    const agentOutput = extractMastraAgentOutput(traceSpans);

    // Add input and output to root span
    if (userInput && !rootSpan.attributes[SemanticConventions.INPUT_VALUE]) {
      rootSpan.attributes[SemanticConventions.INPUT_VALUE] = userInput;
    }

    if (agentOutput && !rootSpan.attributes[SemanticConventions.OUTPUT_VALUE]) {
      rootSpan.attributes[SemanticConventions.OUTPUT_VALUE] = agentOutput;
    }
  }
};

/**
 * Identifies and marks unlabeled root spans in traces containing agent operations.
 *
 * This function performs retroactive span kind assignment for root spans that were not
 * initially identified as agent operations through basic span name prefix matching.
 *
 * The function works by:
 * 1. Identifying traces that contain agent operations (spans with AGENT span kind)
 * 2. Finding root spans in those traces that lack OpenInference span kind attribution
 * 3. Marking those unlabeled root spans as AGENT spans (excluding internal operations)
 *
 * The function modifies spans in place for memory efficiency.
 *
 * @param oiContextualizedSpans - Array of spans that have been processed with OpenInference attributes.
 *                                Must include all spans from traces to ensure accurate identification.
 */
export const markUnlabeledRootSpansInAgentTraces = (
  oiContextualizedSpans: ReadableSpan[],
): void => {
  // List of trace IDs that have agent operations
  const agentTraceIds = new Set<string>();

  // Find all trace IDs that have agent operations
  for (const span of oiContextualizedSpans) {
    const oiKind = getOpenInferenceSpanKind(span);
    if (oiKind === OpenInferenceSpanKind.AGENT) {
      const traceId = getTraceId(span);
      if (traceId) {
        agentTraceIds.add(traceId);
      }
    }
  }

  // Early exit if no agent traces found
  if (!agentTraceIds.size) {
    return;
  }

  // Contextualize root spans from agent traces that are missing the OpenInference AGENT span kind
  for (const span of oiContextualizedSpans) {
    const traceId = getTraceId(span);
    if (!traceId) {
      continue;
    }
    if (
      !getOpenInferenceSpanKind(span) && // is missing the OpenInference span kind
      !span.parentSpanContext && // is root
      !span.name.startsWith(MASTRA_INTERNAL_SPAN_NAME_PREFIX) && // not internal operations
      agentTraceIds.has(traceId) // is agent trace
    ) {
      addOpenInferenceSpanKind(span, OpenInferenceSpanKind.AGENT);
    }
  }
};
