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
 * Span name for Mastra getting the most recent user message
 */
export const AGENT_GET_RECENT_MESSAGE = "agent.getMostRecentUserMessage";

/**
 * Span attribute containing the contents of the most recent user message
 */
export const AGENT_GET_RECENT_MESSAGE_RESULT =
  "agent.getMostRecentUserMessage.result";

/**
 * Span name for Mastra streaming text
 */
export const AGENT_STREAM = "agent.stream";

/**
 * Span attribute containing the arguments at index 0 when calling agent.stream
 */
export const AGENT_STREAM_ARGUMENT = "agent.stream.argument.0";

/**
 * Span name for Mastra generating text
 */
export const AGENT_GENERATE = "agent.generate";

/**
 * Span attribute containing the arguments at index 0 when calling agent.generate
 */
export const AGENT_GENERATE_ARGUMENT = "agent.generate.argument.0";
