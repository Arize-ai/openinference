import { context, ContextManager } from "@opentelemetry/api";
import { AsyncHooksContextManager } from "@opentelemetry/context-async-hooks";

import {
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VARIABLES,
  PROMPT_TEMPLATE_VERSION,
} from "@arizeai/openinference-semantic-conventions";

import {
  clearSession,
  getSessionId,
  setSession,
  setPromptTemplate,
  clearPromptTemplate,
  getPromptTemplate,
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
      setPromptTemplate(context.active(), {
        template: "hello {name}",
        variables,
        version: "V1.0",
      }),
      () => {
        expect(getPromptTemplate(context.active())).toStrictEqual({
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
      setPromptTemplate(context.active(), {
        template: "hello {name}",
        variables,
        version: "V1.0",
      }),
      () => {
        expect(getPromptTemplate(context.active())).toStrictEqual({
          [PROMPT_TEMPLATE_TEMPLATE]: "hello {name}",
          [PROMPT_TEMPLATE_VARIABLES]: JSON.stringify(variables),
          [PROMPT_TEMPLATE_VERSION]: "V1.0",
        });
        const ctx = clearPromptTemplate(context.active());
        expect(getPromptTemplate(ctx)).toBeUndefined();
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
      setSession(context.active(), { sessionId: "session-id" }),
      async () => {
        expect(getSessionId(context.active())).toBe("session-id");
      },
    );
  });

  it("should delete session id from the context", () => {
    context.with(
      setSession(context.active(), { sessionId: "session-id" }),
      async () => {
        expect(getSessionId(context.active())).toBe("session-id");
        const ctx = clearSession(context.active());
        expect(getSessionId(ctx)).toBeUndefined();
      },
    );
  });
});

describe("context.with multiple attributes", () => {
  let contextManager: ContextManager;
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });
  it("should set multiple attributes on the context", () => {
    context.with(
      setSession(
        setPromptTemplate(context.active(), {
          template: "hello {name}",
          variables: { name: "world" },
          version: "V1.0",
        }),
        { sessionId: "session-id" },
      ),

      async () => {
        expect(getSessionId(context.active())).toBe("session-id");
        expect(getPromptTemplate(context.active())).toStrictEqual({
          [PROMPT_TEMPLATE_TEMPLATE]: "hello {name}",
          [PROMPT_TEMPLATE_VARIABLES]: JSON.stringify({ name: "world" }),
          [PROMPT_TEMPLATE_VERSION]: "V1.0",
        });
      },
    );
  });
});
