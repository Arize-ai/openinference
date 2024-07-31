import { context, ContextManager } from "@opentelemetry/api";
import { AsyncHooksContextManager } from "@opentelemetry/context-async-hooks";

import {
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VARIABLES,
  PROMPT_TEMPLATE_VERSION,
} from "@arizeai/openinference-semantic-conventions";

import {
  clearSessionId,
  getSessionId,
  setSessionId,
  setPromptTemplateAttributes,
  clearPromptTemplateAttributes,
  getPromptTemplateAttributes,
} from "../../src";

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
      setPromptTemplateAttributes(context.active(), {
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
      setPromptTemplateAttributes(context.active(), {
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

describe("session context attributes", () => {
  let contextManager: ContextManager;
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });

  it("should set session id in the context", () => {
    context.with(
      setSessionId(context.active(), { sessionId: "session-id" }),
      async () => {
        expect(getSessionId(context.active())).toBe("session-id");
      },
    );
  });

  it("should delete session id from the context", () => {
    context.with(
      setSessionId(context.active(), { sessionId: "session-id" }),
      async () => {
        expect(getSessionId(context.active())).toBe("session-id");
        const ctx = clearSessionId(context.active());
        expect(getSessionId(ctx)).toBeUndefined();
      },
    );
  });
});
