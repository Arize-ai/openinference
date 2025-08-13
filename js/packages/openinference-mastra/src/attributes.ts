import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { addOpenInferenceAttributesToSpan } from "@arizeai/openinference-vercel/utils";
import { addOpenInferenceProjectResourceAttributeSpan } from "./utils.js";

const MASTRA_AGENT_SPAN_NAME_PREFIXES = [
  "agent",
  "mastra.getAgent",
  "post /api/agents",
];

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
 * Get the closest OpenInference span kind for the given Mastra span.
 *
 * This function will attempt to detect the closest OpenInference span kind for the given Mastra span,
 * based on the span's name and parent span ID.
 */
const getOpenInferenceSpanKindFromMastraSpan = (
  span: ReadableSpan,
): OpenInferenceSpanKind | null => {
  const oiKind = getOpenInferenceSpanKind(span);
  if (oiKind) {
    return oiKind;
  }
  const spanName = span.name.toLowerCase();
  if (
    MASTRA_AGENT_SPAN_NAME_PREFIXES.some((prefix) =>
      spanName.startsWith(prefix),
    )
  ) {
    return OpenInferenceSpanKind.AGENT;
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
export const addOpenInferenceAttributesToMastraSpan = (
  span: ReadableSpan,
  shouldMarkAsAgent: boolean = false,
) => {
  const kind = getOpenInferenceSpanKindFromMastraSpan(span);
  if (kind) {
    addOpenInferenceSpanKind(span, kind);
  }

  // Mark root spans as AGENT if requested and not already set
  if (
    span.parentSpanContext === undefined &&
    !getOpenInferenceSpanKind(span) &&
    shouldMarkAsAgent
  ) {
    addOpenInferenceSpanKind(span, OpenInferenceSpanKind.AGENT);
  }

  // Map Mastra threadId to OpenInference session ID
  // Only set SESSION_ID if it doesn't already exist to avoid overwriting existing values
  const threadId = span.attributes.threadId;
  if (
    threadId &&
    (typeof threadId === "string" || typeof threadId === "number") &&
    !span.attributes[SemanticConventions.SESSION_ID]
  ) {
    span.attributes[SemanticConventions.SESSION_ID] = threadId;
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
 * @param shouldMarkAsAgent - Whether root spans should be marked as AGENT spans.
 */
export const processMastraSpanAttributes = (
  span: ReadableSpan,
  shouldMarkAsAgent: boolean = false,
): void => {
  addOpenInferenceProjectResourceAttributeSpan(span);
  addOpenInferenceAttributesToSpan({
    ...span,
    instrumentationLibrary: {
      name: "@arizeai/openinference-mastra",
    },
  });
  addOpenInferenceAttributesToMastraSpan(span, shouldMarkAsAgent);
};

/**
 * Determines whether a span represents an agent operation.
 *
 * @param span - The span to check.
 * @returns `true` if the span is an agent operation, `false` otherwise.
 */
export const isAgentOperation = (span: ReadableSpan): boolean => {
  return !!(
    span.name.startsWith("agent.") ||
    span.attributes.threadId ||
    span.attributes.componentName ||
    span.attributes.resourceId
  );
};

/**
 * Gets the span ID from a ReadableSpan.
 *
 * @param span - The span to get the ID from.
 * @returns The span ID or "unknown" if not available.
 */
export const getSpanId = (span: ReadableSpan): string => {
  return span.spanContext?.()?.spanId || "unknown";
};

/**
 * Gets the trace ID from a ReadableSpan.
 *
 * @param span - The span to get the trace ID from.
 * @returns The trace ID or "unknown" if not available.
 */
export const getTraceId = (span: ReadableSpan): string => {
  return span.spanContext?.()?.traceId || "unknown";
};

/**
 * Extracts user input from Mastra spans for session I/O.
 *
 * Looks for user input in the following order:
 * 1. agent.getMostRecentUserMessage.result attribute
 * 2. agent.stream.argument.0 conversation messages (last user message)
 *
 * @param spans - Array of spans to search through.
 * @returns The extracted user input or undefined if not found.
 */
export const extractMastraUserInput = (
  spans: ReadableSpan[],
): string | undefined => {
  // Look for the most recent user message from agent.getMostRecentUserMessage.result
  for (const span of spans) {
    if (span.name === "agent.getMostRecentUserMessage") {
      const result = span.attributes["agent.getMostRecentUserMessage.result"];
      if (typeof result === "string") {
        try {
          const messageData = JSON.parse(result);
          if (messageData.content && typeof messageData.content === "string") {
            return messageData.content;
          }
        } catch {
          // Ignore parsing errors
        }
      }
    }
  }

  // Fallback: extract from agent.stream.argument.0 (conversation messages)
  for (const span of spans) {
    if (span.name === "agent.stream") {
      const argument = span.attributes["agent.stream.argument.0"];
      if (typeof argument === "string") {
        try {
          const messages = JSON.parse(argument);
          if (Array.isArray(messages)) {
            // Find the last user message
            for (let i = messages.length - 1; i >= 0; i--) {
              const message = messages[i];
              if (message.role === "user" && message.content) {
                return typeof message.content === "string"
                  ? message.content
                  : message.content;
              }
            }
          }
        } catch {
          // Ignore parsing errors
        }
      }
    }
  }

  return undefined;
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
  return undefined;
};

/**
 * Adds input and output attributes to root spans for session I/O support.
 *
 * Extracts user input and agent output from the provided spans and adds them
 * to any root spans that don't already have I/O attributes set.
 *
 * @param spans - Array of spans to process.
 */
export const addIOToRootSpans = (spans: ReadableSpan[]): void => {
  const rootSpans = spans.filter(
    (span) => span.parentSpanContext === undefined,
  );

  if (rootSpans.length === 0) return;

  const userInput = extractMastraUserInput(spans);
  const agentOutput = extractMastraAgentOutput(spans);

  // Add input and output to root spans
  for (const rootSpan of rootSpans) {
    if (userInput && !rootSpan.attributes[SemanticConventions.INPUT_VALUE]) {
      rootSpan.attributes[SemanticConventions.INPUT_VALUE] = userInput;
      rootSpan.attributes[SemanticConventions.INPUT_MIME_TYPE] = "text/plain";
    }

    if (agentOutput && !rootSpan.attributes[SemanticConventions.OUTPUT_VALUE]) {
      rootSpan.attributes[SemanticConventions.OUTPUT_VALUE] = agentOutput;
      rootSpan.attributes[SemanticConventions.OUTPUT_MIME_TYPE] = "text/plain";
    }
  }
};

/**
 * Adds missing root spans for traces with agent operations.
 *
 * When spans are filtered (e.g., only agent spans are exported), this function
 * ensures that root spans for agent traces are also included to maintain trace
 * context for session I/O.
 *
 * @param allSpans - All spans before filtering.
 * @param filteredSpans - Spans after filtering.
 * @returns Combined array of filtered spans plus any missing root spans.
 */
export const addMissingAgentRootSpans = (
  allSpans: ReadableSpan[],
  filteredSpans: ReadableSpan[],
): ReadableSpan[] => {
  const filteredSpanIds = new Set(filteredSpans.map((s) => getSpanId(s)));

  // Check filtered spans for agent operations
  const agentTraceIds = new Set<string>();
  for (const span of filteredSpans) {
    if (isAgentOperation(span)) {
      agentTraceIds.add(getTraceId(span));
    }
  }

  // Find missing root spans for agent traces
  const missingRoots: ReadableSpan[] = [];
  for (const span of allSpans) {
    if (
      span.parentSpanContext === undefined && // is root
      agentTraceIds.has(getTraceId(span)) && // is agent trace
      !filteredSpanIds.has(getSpanId(span)) && // not already included
      !span.name.startsWith("mastra.")
    ) {
      // not internal operation
      // Process the missing root span
      processMastraSpanAttributes(span, true); // shouldMarkAsAgent = true
      missingRoots.push(span);
    }
  }

  return [...filteredSpans, ...missingRoots];
};
