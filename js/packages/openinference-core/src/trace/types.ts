/**
 * A set of attributes that can be attached to context
 */
export type SessionAttributes = {
  sessionId: string;
};

/**
 * A set of metadata attributes that can be attached to context
 */
export type MetadataAttributes = Record<string, unknown>;

/**
 * A set of user attributes that can be attached to context
 */
export type UserAttributes = {
  userId: string;
};

/**
 * A set of prompt template attributes that can be attached to context
 */
export type PromptTemplateAttributes = {
  template: string;
  variables?: Record<string, unknown>;
  version?: string;
};

/**
 * A set of tags that can be attached to context
 */
export type TagAttributes = string[];
