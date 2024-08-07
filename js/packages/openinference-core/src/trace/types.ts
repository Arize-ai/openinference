/**
 * A session that can be attached to context
 */
export type Session = {
  sessionId: string;
};

/**
 * Metadata that can be attached to context
 */
export type Metadata = Record<string, unknown>;

/**
 * A user that can be attached to context
 */
export type User = {
  userId: string;
};

/**
 * A prompt template that can be attached to context
 */
export type PromptTemplate = {
  template: string;
  variables?: Record<string, unknown>;
  version?: string;
};

/**
 * A set of tags that can be attached to context
 */
export type Tags = string[];
