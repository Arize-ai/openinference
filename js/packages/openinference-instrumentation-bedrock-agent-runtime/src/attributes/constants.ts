/**
 * Ordered a list of known event wrapper keys inside a trace object.
 * These are ordered by the order of the events in the trace.
 */
export const TRACE_EVENT_TYPES = [
  "preProcessingTrace",
  "orchestrationTrace",
  "postProcessingTrace",
  "failureTrace",
] as const;

export type TraceEventType = (typeof TRACE_EVENT_TYPES)[number];

/**
 * Ordered a list of known chunk wrapper keys inside an event object.
 */
export const CHUNK_TYPES = [
  "invocationInput",
  "modelInvocationInput",
  "modelInvocationOutput",
  "agentCollaboratorInvocationInput",
  "agentCollaboratorInvocationOutput",
  "rationale",
  "observation",
] as const;

export type ChunkType = (typeof CHUNK_TYPES)[number];
