/**
 * Constants for Mastra agent span names and attributes
 */

/**
 * Prefix for Mastra agent span names. Examples: "agent.generate", "mastra.getAgent"
 */
export const MASTRA_AGENT_SPAN_NAME_PREFIXES = ["agent", "mastra.getAgent"];

/**
 * Prefix for internal Mastra span names. Examples: "mastra.getServer"
 */
export const MASTRA_INTERNAL_SPAN_NAME_PREFIX = "mastra.";

/**
 * Known Mastra span names.
 */
export const MastraSpanNames = {
  AGENT_GET_RECENT_MESSAGE: "agent.getMostRecentUserMessage",
  /**
   * Span name for Mastra streaming text
   */
  AGENT_STREAM: "agent.stream",
  AGENT_GENERATE: "agent.generate",
} as const;

export const MastraSpanAttributes = {
  /**
   * Span attribute containing the contents of the most recent user message
   */
  AGENT_GET_RECENT_MESSAGE_RESULT: "agent.getMostRecentUserMessage.result",
  /**
   * Span attribute containing the arguments at index 0 when calling agent.generate
   */
  AGENT_GENERATE_ARGUMENT: "agent.generate.argument.0",
  /**
   * Span attribute containing the arguments at the first index when calling agent.stream
   */
  AGENT_STREAM_ARGUMENT: "agent.stream.argument.0",
} as const;
