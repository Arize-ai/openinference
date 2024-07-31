import {
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VARIABLES,
  PROMPT_TEMPLATE_VERSION,
  SESSION_ID,
} from "@arizeai/openinference-semantic-conventions";
import {
  Attributes,
  Context,
  createContextKey,
  AttributeValue,
} from "@opentelemetry/api";

const ContextAttributes = {
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
} as const;

const {
  [PROMPT_TEMPLATE_TEMPLATE]: PROMPT_TEMPLATE_TEMPLATE_KEY,
  [PROMPT_TEMPLATE_VARIABLES]: PROMPT_TEMPLATE_VARIABLES_KEY,
  [PROMPT_TEMPLATE_VERSION]: PROMPT_TEMPLATE_VERSION_KEY,
  [SESSION_ID]: SESSION_ID_KEY,
} = ContextAttributes;

export type PromptTemplateAttributes = {
  template: string;
  variables?: Record<string, unknown>;
  version?: string;
};
export function setPromptTemplate(
  context: Context,
  attributes: PromptTemplateAttributes,
): Context {
  const { template, variables, version } = attributes;
  context = context.setValue(PROMPT_TEMPLATE_TEMPLATE_KEY, template);
  if (variables) {
    context = context.setValue(
      PROMPT_TEMPLATE_VARIABLES_KEY,
      JSON.stringify(variables),
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

export function getPromptTemplate(context: Context): Attributes | undefined {
  const maybeTemplate = context.getValue(PROMPT_TEMPLATE_TEMPLATE_KEY);
  const maybeVariables = context.getValue(PROMPT_TEMPLATE_VARIABLES_KEY);
  const maybeVersion = context.getValue(PROMPT_TEMPLATE_VERSION_KEY);
  const attributes: Attributes = {};
  if (typeof maybeTemplate === "string") {
    attributes[PROMPT_TEMPLATE_TEMPLATE] = maybeTemplate;
  }
  if (typeof maybeVariables === "string") {
    attributes[PROMPT_TEMPLATE_VARIABLES] = maybeVariables;
  }
  if (typeof maybeVersion === "string") {
    attributes[PROMPT_TEMPLATE_VERSION] = maybeVersion;
  }

  if (Object.keys(attributes).length === 0) {
    return;
  }

  return attributes;
}

export type SessionAttributes = {
  sessionId: string;
};

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
export function getSessionId(context: Context): string | undefined {
  const maybeSessionId = context.getValue(SESSION_ID_KEY);
  if (typeof maybeSessionId === "string") {
    return maybeSessionId;
  }
}

export function getAttributesFromContext(context: Context): Attributes {
  const attributes: Attributes = {};
  Object.entries(ContextAttributes).forEach(([key, symbol]) => {
    const maybeValue = context.getValue(symbol);
    if (maybeValue != null) {
      // TODO add a type guard for attribute values
      attributes[key] = maybeValue as AttributeValue;
    }
  });

  return attributes;
}
