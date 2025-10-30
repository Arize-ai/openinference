import {
  METADATA,
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VARIABLES,
  PROMPT_TEMPLATE_VERSION,
  SESSION_ID,
  TAG_TAGS,
  USER_ID,
} from "@arizeai/openinference-semantic-conventions";

import { Attributes, Context, createContextKey } from "@opentelemetry/api";
import { isAttributeValue } from "@opentelemetry/core";

import {
  isAttributes,
  isObjectWithStringKeys,
  isStringArray,
  safelyJSONParse,
  safelyJSONStringify,
} from "../utils";

import { Metadata, PromptTemplate, Session, Tags, User } from "./types";

const CONTEXT_ATTRIBUTES_ATTRIBUTES_KEY = "attributes" as const;

export const ContextAttributes = {
  [PROMPT_TEMPLATE_TEMPLATE]: createContextKey(
    `OpenInference SDK Context Key ${PROMPT_TEMPLATE_TEMPLATE}`,
  ),
  [PROMPT_TEMPLATE_VARIABLES]: createContextKey(
    `OpenInference SDK Context Key ${PROMPT_TEMPLATE_VARIABLES}`,
  ),
  [PROMPT_TEMPLATE_VERSION]: createContextKey(
    `OpenInference SDK Context Key ${PROMPT_TEMPLATE_VERSION}`,
  ),
  [SESSION_ID]: createContextKey(`OpenInference SDK Context Key ${SESSION_ID}`),
  [METADATA]: createContextKey(`OpenInference SDK Context Key ${METADATA}`),
  [USER_ID]: createContextKey(`OpenInference SDK Context Key ${USER_ID}`),
  [TAG_TAGS]: createContextKey(`OpenInference SDK Context Key ${TAG_TAGS}`),
  [CONTEXT_ATTRIBUTES_ATTRIBUTES_KEY]: createContextKey(
    `OpenInference SDK Context Key attributes`,
  ),
} as const;

const {
  [PROMPT_TEMPLATE_TEMPLATE]: PROMPT_TEMPLATE_TEMPLATE_KEY,
  [PROMPT_TEMPLATE_VARIABLES]: PROMPT_TEMPLATE_VARIABLES_KEY,
  [PROMPT_TEMPLATE_VERSION]: PROMPT_TEMPLATE_VERSION_KEY,
  [SESSION_ID]: SESSION_ID_KEY,
  [METADATA]: METADATA_KEY,
  [USER_ID]: USER_ID_KEY,
  [TAG_TAGS]: TAG_TAGS_KEY,
  [CONTEXT_ATTRIBUTES_ATTRIBUTES_KEY]: ATTRIBUTES_KEY,
} = ContextAttributes;

/**
 * Sets the prompt template in the given OpenTelemetry context.
 * If OI tracer is used in spans created during this active context, the prompt template will be added to the span as an attribute.
 * @param context - The context to set the prompt template on.
 * @param promptTemplate - The prompt template to set on the context.
 * @returns The context
 * @example
 * ```typescript
 * context.with(
 *   setPromptTemplate(context.active(), {
 *     template: "hello {name}",
 *     variables: { name: "world" },
 *     version: "V1.0"
 *   }),
 *   () => {
 *     // Spans created here will have prompt template attributes
 *   }
 * );
 * ```
 */
export function setPromptTemplate(
  context: Context,
  promptTemplate: PromptTemplate,
): Context {
  const { template, variables, version } = promptTemplate;
  context = context.setValue(PROMPT_TEMPLATE_TEMPLATE_KEY, template);
  if (variables) {
    context = context.setValue(
      PROMPT_TEMPLATE_VARIABLES_KEY,
      safelyJSONStringify(variables),
    );
  }
  if (version) {
    context = context.setValue(PROMPT_TEMPLATE_VERSION_KEY, version);
  }
  return context;
}

/**
 * Clears the prompt template from the given OpenTelemetry context.
 * @param context - The context to clear the prompt template from.
 * @returns The context
 */
export function clearPromptTemplate(context: Context): Context {
  context = context.deleteValue(PROMPT_TEMPLATE_TEMPLATE_KEY);
  context = context.deleteValue(PROMPT_TEMPLATE_VARIABLES_KEY);
  context = context.deleteValue(PROMPT_TEMPLATE_VERSION_KEY);
  return context;
}

/**
 * Gets the prompt template from the given OpenTelemetry context.
 * @param context - The context to get the prompt template from.
 * @returns The prompt template
 * @example const promptTemplate = getPromptTemplate(context.active());
 */
export function getPromptTemplate(
  context: Context,
): Partial<PromptTemplate> | undefined {
  const maybeTemplate = context.getValue(PROMPT_TEMPLATE_TEMPLATE_KEY);
  const maybeVariables = context.getValue(PROMPT_TEMPLATE_VARIABLES_KEY);
  const maybeVersion = context.getValue(PROMPT_TEMPLATE_VERSION_KEY);
  const attributes: Partial<PromptTemplate> = {};

  if (typeof maybeTemplate === "string") {
    attributes.template = maybeTemplate;
  }
  if (typeof maybeVariables === "string") {
    const parsedVariables = safelyJSONParse(maybeVariables);
    attributes.variables = isObjectWithStringKeys(parsedVariables)
      ? parsedVariables
      : undefined;
  }
  if (typeof maybeVersion === "string") {
    attributes.version = maybeVersion;
  }

  if (Object.keys(attributes).length === 0) {
    return;
  }

  return attributes;
}

/**
 * Sets the session ID in the given OpenTelemetry context.
 * If OI tracer is used in spans created during this active context, the session ID will be added to the span as an attribute.
 * @param context - The context to set the session ID on.
 * @param session - The session to set on the context.
 * @returns The context
 * @example
 * ```typescript
 * context.with(
 *   setSession(context.active(), { sessionId: "session-123" }),
 *   () => {
 *     // Spans created here will have session ID attribute
 *   }
 * );
 * ```
 */
export function setSession(context: Context, session: Session): Context {
  const { sessionId } = session;
  return context.setValue(SESSION_ID_KEY, sessionId);
}

export function clearSession(context: Context): Context {
  return context.deleteValue(SESSION_ID_KEY);
}

/**
 * Retrieves the session ID from the given context.
 * @param context - The context object.
 * @returns {string | undefined} The session ID if it exists, otherwise undefined.
 */
export function getSession(context: Context): Session | undefined {
  const maybeSessionId = context.getValue(SESSION_ID_KEY);
  if (typeof maybeSessionId === "string") {
    return { sessionId: maybeSessionId };
  }
}

/**
 * Sets the metadata in the given OpenTelemetry context.
 * If OI tracer is used in spans created during this active context, the metadata will be added to the span as an attribute.
 * @param context - The context to set the metadata on.
 * @param metadata - The metadata to set on the context.
 * @returns The context
 * @example
 * ```typescript
 * context.with(
 *   setMetadata(context.active(), { key: "value", numeric: 1 }),
 *   () => {
 *     // Spans created here will have metadata attribute
 *   }
 * );
 * ```
 */
export function setMetadata(context: Context, metadata: Metadata): Context {
  return context.setValue(METADATA_KEY, safelyJSONStringify(metadata));
}

/**
 * Clears the metadata from the given OpenTelemetry context.
 * @param context - The context to clear the metadata from.
 * @returns The context
 */
export function clearMetadata(context: Context): Context {
  return context.deleteValue(METADATA_KEY);
}

/**
 * Gets the metadata from the given OpenTelemetry context.
 * @param context - The context to get the metadata from.
 * @returns The metadata
 * @example const metadata = getMetadata(context.active());
 */
export function getMetadata(context: Context): Metadata | undefined {
  const maybeMetadata = context.getValue(METADATA_KEY);

  if (typeof maybeMetadata === "string") {
    const parsedMetadata = safelyJSONParse(maybeMetadata);
    return isObjectWithStringKeys(parsedMetadata) ? parsedMetadata : undefined;
  }
}

/**
 * Sets the user ID in the given OpenTelemetry context.
 * @param context - The context to set the user ID on.
 * @param user - The user to set on the context.
 * @returns The context
 * @example
 * ```typescript
 * context.with(
 *   setUser(context.active(), { userId: "user-123" }),
 *   () => {
 *     // Spans created here will have user ID attribute
 *   }
 * );
 * ```
 */
export function setUser(context: Context, user: User): Context {
  const { userId } = user;
  return context.setValue(USER_ID_KEY, userId);
}

/**
 * Clears the user ID from the given OpenTelemetry context.
 * @param context - The context to clear the user ID from.
 * @returns The context
 */
export function clearUser(context: Context): Context {
  return context.deleteValue(USER_ID_KEY);
}

/**
 * Gets the user ID from the given OpenTelemetry context.
 * @param context - The context to get the user ID from.
 * @returns The user
 * @example const user = getUser(context.active());
 */
export function getUser(context: Context): User | undefined {
  const maybeUserId = context.getValue(USER_ID_KEY);
  if (typeof maybeUserId === "string") {
    return { userId: maybeUserId };
  }
}

/**
 * Sets the tags in the given OpenTelemetry context.
 * If OI tracer is used in spans created during this active context, the tags will be added to the span as an attribute.
 * @param context - The context to set the tags on.
 * @param tags - The tags to set on the context.
 * @returns The context
 * @example
 * ```typescript
 * context.with(
 *   setTags(context.active(), ["tag1", "tag2"]),
 *   () => {
 *     // Spans created here will have tags attribute
 *   }
 * );
 * ```
 */
export function setTags(context: Context, tags: Tags): Context {
  return context.setValue(TAG_TAGS_KEY, safelyJSONStringify(tags));
}

/**
 * Clears the tags from the given OpenTelemetry context.
 * @param context - The context to clear the tags from.
 * @returns The context
 */
export function clearTags(context: Context): Context {
  return context.deleteValue(TAG_TAGS_KEY);
}

/**
 * Gets the tags from the given OpenTelemetry context.
 * @param context - The context to get the tags from.
 * @returns The tags
 * @example const tags = getTags(context.active());
 */
export function getTags(context: Context): Tags | undefined {
  const maybeTags = context.getValue(TAG_TAGS_KEY);
  if (typeof maybeTags === "string") {
    const parsedTags = safelyJSONParse(maybeTags);
    return isStringArray(parsedTags) ? parsedTags : undefined;
  }
}

/**
 * Sets the attributes in the given OpenTelemetry context.
 * If OI tracer is used in spans created during this active context, the attributes will be added to the span as an attribute.
 * @param context - The context to set the attributes on.
 * @param attributes - The attributes to set on the context.
 * @returns The context
 * @example
 * ```typescript
 * context.with(
 *   setAttributes(context.active(), { custom: "value", test: "attribute" }),
 *   () => {
 *     // Spans created here will have custom attributes
 *   }
 * );
 * ```
 */
export function setAttributes(
  context: Context,
  attributes: Attributes,
): Context {
  return context.setValue(ATTRIBUTES_KEY, safelyJSONStringify(attributes));
}

/**
 * Clears the attributes from the given OpenTelemetry context.
 * @param context - The context to clear the attributes from.
 * @returns The context
 */
export function clearAttributes(context: Context): Context {
  return context.deleteValue(ATTRIBUTES_KEY);
}

/**
 * Gets the attributes from the given OpenTelemetry context.
 * @param context - The context to get the attributes from.
 * @returns The attributes
 * @example const attributes = getAttributes(context.active());
 */
export function getAttributes(context: Context): Attributes | undefined {
  const maybeAttributes = context.getValue(ATTRIBUTES_KEY);
  if (typeof maybeAttributes === "string") {
    const parsedAttributes = safelyJSONParse(maybeAttributes);
    return isAttributes(parsedAttributes) ? parsedAttributes : undefined;
  }
}

/**
 * Gets the OpenInference attributes from the given context
 * @param context
 * @example
 * ```typescript
 * // Chaining multiple attributes
 * context.with(
 *   setSession(
 *     setPromptTemplate(context.active(), {
 *       template: "hello {name}",
 *       variables: { name: "world" },
 *       version: "V1.0"
 *     }),
 *     { sessionId: "session-123" }
 *   ),
 *   () => {
 *     // Extract all attributes for span
 *     const attributes = getAttributesFromContext(context.active());
 *     span.setAttributes(attributes);
 *   }
 * );
 * ```
 * @returns {Attributes} The OpenInference attributes formatted as OpenTelemetry span attributes.
 */
export function getAttributesFromContext(context: Context): Attributes {
  let attributes: Attributes = {};
  Object.entries(ContextAttributes).forEach(([key, symbol]) => {
    const maybeValue = context.getValue(symbol);
    if (key === CONTEXT_ATTRIBUTES_ATTRIBUTES_KEY) {
      if (typeof maybeValue === "string") {
        const parsedAttributes = safelyJSONParse(maybeValue);
        if (isAttributes(parsedAttributes)) {
          attributes = { ...attributes, ...parsedAttributes };
        }
      }
      return;
    }

    if (isAttributeValue(maybeValue) && maybeValue !== undefined) {
      attributes[key] = maybeValue;
    }
  });
  return attributes;
}
