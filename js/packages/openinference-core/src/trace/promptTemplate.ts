import {
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VARIABLES,
  PROMPT_TEMPLATE_VERSION,
} from "@arizeai/openinference-semantic-conventions";
import { Attributes, Context, createContextKey } from "@opentelemetry/api";

export const PROMPT_TEMPLATE_TEMPLATE_KEY = createContextKey(
  `OpenInference SDK Context Key ${PROMPT_TEMPLATE_TEMPLATE}`,
);

export const PROMPT_TEMPLATE_VARIABLES_KEY = createContextKey(
  `OpenInference SDK Context Key ${PROMPT_TEMPLATE_VARIABLES}`,
);

export const PROMPT_TEMPLATE_VERSION_KEY = createContextKey(
  `OpenInference SDK Context Key ${PROMPT_TEMPLATE_VERSION}`,
);

export function setPromptTemplateAttributes({
  context,
  template,
  variables,
  version,
}: {
  context: Context;
  template?: string;
  variables?: Record<string, unknown>;
  version?: string;
}): Context {
  if (template) {
    context = context.setValue(PROMPT_TEMPLATE_TEMPLATE_KEY, template);
  }
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

export function clearPromptTemplateAttributes(context: Context): Context {
  context = context.deleteValue(PROMPT_TEMPLATE_TEMPLATE_KEY);
  context = context.deleteValue(PROMPT_TEMPLATE_VARIABLES_KEY);
  context = context.deleteValue(PROMPT_TEMPLATE_VERSION_KEY);
  return context;
}

export function getPromptTemplateAttributes(
  context: Context,
): Attributes | undefined {
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
