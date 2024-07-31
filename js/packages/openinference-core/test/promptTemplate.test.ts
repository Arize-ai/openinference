import { context, ContextManager } from "@opentelemetry/api";
import { AsyncHooksContextManager } from "@opentelemetry/context-async-hooks";
import {
  setPromptTemplateAttributes,
  clearPromptTemplateAttributes,
  getPromptTemplateAttributes,
} from "../src/trace/promptTemplate";
import {
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VARIABLES,
  PROMPT_TEMPLATE_VERSION,
} from "@arizeai/openinference-semantic-conventions";

describe("promptTemplate", () => {
  let contextManager: ContextManager;
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });
  it("should set prompt template attributes on the context", () => {
    const variables = {
      name: "world",
    };
    context.with(
      setPromptTemplateAttributes({
        context: context.active(),
        template: "hello {name}",
        variables,
        version: "V1.0",
      }),
      () => {
        expect(getPromptTemplateAttributes(context.active())).toStrictEqual({
          [PROMPT_TEMPLATE_TEMPLATE]: "hello {name}",
          [PROMPT_TEMPLATE_VARIABLES]: JSON.stringify(variables),
          [PROMPT_TEMPLATE_VERSION]: "V1.0",
        });
      },
    );
  });

  it("should delete prompt template attributes from the context", () => {
    const variables = {
      name: "world",
    };
    context.with(
      setPromptTemplateAttributes({
        context: context.active(),
        template: "hello {name}",
        variables,
        version: "V1.0",
      }),
      () => {
        expect(getPromptTemplateAttributes(context.active())).toStrictEqual({
          [PROMPT_TEMPLATE_TEMPLATE]: "hello {name}",
          [PROMPT_TEMPLATE_VARIABLES]: JSON.stringify(variables),
          [PROMPT_TEMPLATE_VERSION]: "V1.0",
        });
        const ctx = clearPromptTemplateAttributes(context.active());
        expect(getPromptTemplateAttributes(ctx)).toBeUndefined();
      },
    );
  });
});
