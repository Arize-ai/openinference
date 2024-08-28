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
import {
  safelyJSONStringify,
  safelyJSONParse,
  isStringArray,
  isObjectWithStringKeys,
  isAttributes,
} from "../utils";
import { Metadata, PromptTemplate, Session, Tags, User } from "./types";
import { isAttributeValue } from "@opentelemetry/core";

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

export function setPromptTemplate(
  context: Context,
  attributes: PromptTemplate,
): Context {
  const { template, variables, version } = attributes;
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

export function clearPromptTemplate(context: Context): Context {
  context = context.deleteValue(PROMPT_TEMPLATE_TEMPLATE_KEY);
  context = context.deleteValue(PROMPT_TEMPLATE_VARIABLES_KEY);
  context = context.deleteValue(PROMPT_TEMPLATE_VERSION_KEY);
  return context;
}

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

export function setSession(context: Context, attributes: Session): Context {
  const { sessionId } = attributes;
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

export function setMetadata(context: Context, attributes: Metadata): Context {
  return context.setValue(METADATA_KEY, safelyJSONStringify(attributes));
}

export function clearMetadata(context: Context): Context {
  return context.deleteValue(METADATA_KEY);
}

export function getMetadata(context: Context): Metadata | undefined {
  const maybeMetadata = context.getValue(METADATA_KEY);

  if (typeof maybeMetadata === "string") {
    const parsedMetadata = safelyJSONParse(maybeMetadata);
    return isObjectWithStringKeys(parsedMetadata) ? parsedMetadata : undefined;
  }
}

export function setUser(context: Context, attributes: User): Context {
  const { userId } = attributes;
  return context.setValue(USER_ID_KEY, userId);
}

export function clearUser(context: Context): Context {
  return context.deleteValue(USER_ID_KEY);
}

export function getUser(context: Context): User | undefined {
  const maybeUserId = context.getValue(USER_ID_KEY);
  if (typeof maybeUserId === "string") {
    return { userId: maybeUserId };
  }
}

export function setTags(context: Context, attributes: Tags): Context {
  return context.setValue(TAG_TAGS_KEY, safelyJSONStringify(attributes));
}

export function clearTags(context: Context): Context {
  return context.deleteValue(TAG_TAGS_KEY);
}

export function getTags(context: Context): Tags | undefined {
  const maybeTags = context.getValue(TAG_TAGS_KEY);
  if (typeof maybeTags === "string") {
    const parsedTags = safelyJSONParse(maybeTags);
    return isStringArray(parsedTags) ? parsedTags : undefined;
  }
}

export function setAttributes(
  context: Context,
  attributes: Attributes,
): Context {
  return context.setValue(ATTRIBUTES_KEY, safelyJSONStringify(attributes));
}

export function clearAttributes(context: Context): Context {
  return context.deleteValue(ATTRIBUTES_KEY);
}

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
 * @example span.setAttributes(getAttributesFromContext(context.active()));
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
