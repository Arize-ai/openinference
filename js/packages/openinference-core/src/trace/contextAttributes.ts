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
  isAttributeValue,
  safelyJSONParse,
  isStringArray,
  isObjectWithStringKeys,
} from "../utils";
import {
  MetadataAttributes,
  PromptTemplateAttributes,
  SessionAttributes,
  TagAttributes,
  UserAttributes,
} from "./types";

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
  attributes: createContextKey(`OpenInference SDK Context Key attributes`),
} as const;

const {
  [PROMPT_TEMPLATE_TEMPLATE]: PROMPT_TEMPLATE_TEMPLATE_KEY,
  [PROMPT_TEMPLATE_VARIABLES]: PROMPT_TEMPLATE_VARIABLES_KEY,
  [PROMPT_TEMPLATE_VERSION]: PROMPT_TEMPLATE_VERSION_KEY,
  [SESSION_ID]: SESSION_ID_KEY,
  [METADATA]: METADATA_KEY,
  [USER_ID]: USER_ID_KEY,
  [TAG_TAGS]: TAG_TAGS_KEY,
  attributes: ATTRIBUTES_KEY,
} = ContextAttributes;

export function setPromptTemplate(
  context: Context,
  attributes: PromptTemplateAttributes,
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
): Partial<PromptTemplateAttributes> | undefined {
  const maybeTemplate = context.getValue(PROMPT_TEMPLATE_TEMPLATE_KEY);
  const maybeVariables = context.getValue(PROMPT_TEMPLATE_VARIABLES_KEY);
  const maybeVersion = context.getValue(PROMPT_TEMPLATE_VERSION_KEY);
  const attributes: Partial<PromptTemplateAttributes> = {};

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

export function setSession(
  context: Context,
  attributes: SessionAttributes,
): Context {
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
export function getSession(context: Context): SessionAttributes | undefined {
  const maybeSessionId = context.getValue(SESSION_ID_KEY);
  if (typeof maybeSessionId === "string") {
    return { sessionId: maybeSessionId };
  }
}

export function setMetadata(
  context: Context,
  attributes: MetadataAttributes,
): Context {
  return context.setValue(METADATA_KEY, safelyJSONStringify(attributes));
}

export function clearMetadata(context: Context): Context {
  return context.deleteValue(METADATA_KEY);
}

export function getMetadata(
  context: Context,
): MetadataAttributes | null | undefined {
  const maybeMetadata = context.getValue(METADATA_KEY);

  if (typeof maybeMetadata === "string") {
    const parsedMetadata = safelyJSONParse(maybeMetadata);
    return isObjectWithStringKeys(parsedMetadata) ? parsedMetadata : null;
  }
}

export function setUser(context: Context, attributes: UserAttributes): Context {
  const { userId } = attributes;
  return context.setValue(USER_ID_KEY, userId);
}

export function clearUser(context: Context): Context {
  return context.deleteValue(USER_ID_KEY);
}

export function getUserId(context: Context): UserAttributes | undefined {
  const maybeUserId = context.getValue(USER_ID_KEY);
  if (typeof maybeUserId === "string") {
    return { userId: maybeUserId };
  }
}

export function setTags(context: Context, attributes: TagAttributes): Context {
  return context.setValue(TAG_TAGS_KEY, safelyJSONStringify(attributes));
}

export function clearTags(context: Context): Context {
  return context.deleteValue(TAG_TAGS_KEY);
}

export function getTags(context: Context): TagAttributes | null | undefined {
  const maybeTags = context.getValue(TAG_TAGS_KEY);
  if (typeof maybeTags === "string") {
    const parsedTags = safelyJSONParse(maybeTags);
    return isStringArray(parsedTags) ? parsedTags : null;
  }
}

export function setAttributes(
  context: Context,
  attributes: Attributes,
): Context {}

export function getAttributesFromContext(context: Context): Attributes {
  const attributes: Attributes = {};
  Object.entries(ContextAttributes).forEach(([key, symbol]) => {
    const maybeValue = context.getValue(symbol);
    if (isAttributeValue(maybeValue)) {
      attributes[key] = maybeValue;
    }
  });
  return attributes;
}
