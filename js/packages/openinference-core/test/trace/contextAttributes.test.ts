import { context, ContextManager } from "@opentelemetry/api";
import { AsyncHooksContextManager } from "@opentelemetry/context-async-hooks";

import {
  METADATA,
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VARIABLES,
  PROMPT_TEMPLATE_VERSION,
  SESSION_ID,
} from "@arizeai/openinference-semantic-conventions";

import {
  clearSession,
  getSessionId,
  setSession,
  setPromptTemplate,
  clearPromptTemplate,
  getPromptTemplate,
  setMetadata,
  getMetadata,
  clearMetadata,
  getAttributesFromContext,
  ContextAttributes,
} from "../../src";

describe("promptTemplate context attributes", () => {
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
      () => {
        expect(getSessionId(context.active())).toBe("session-id");
      },
    );
  });

  it("should delete session id from the context", () => {
    context.with(
      setSession(context.active(), { sessionId: "session-id" }),
      () => {
        expect(getSessionId(context.active())).toBe("session-id");
        const ctx = clearSession(context.active());
        expect(getSessionId(ctx)).toBeUndefined();
      },
    );
  });
});

describe("metadata context attributes", () => {
  let contextManager: ContextManager;
  const metadataAttributes = {
    key: "value",
    numeric: 1,
    list: ["hello", "bye"],
  };
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });
  it("should set metadata attributes on the context", () => {
    context.with(setMetadata(context.active(), metadataAttributes), () => {
      expect(getMetadata(context.active())).toStrictEqual({
        [METADATA]: JSON.stringify(metadataAttributes),
      });
    });
  });
  it("should delete metadata attributes from the context", () => {
    context.with(setMetadata(context.active(), metadataAttributes), () => {
      expect(getMetadata(context.active())).toStrictEqual({
        [METADATA]: JSON.stringify(metadataAttributes),
      });
      const ctx = clearMetadata(context.active());
      expect(getMetadata(ctx)).toBeUndefined();
    });
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

      () => {
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

describe("getContextAttributes", () => {
  let contextManager: ContextManager;
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });

  it("should get all attributes off of the context", () => {
    const variables = {
      name: "world",
    };
    context.with(
      setMetadata(
        setSession(
          setPromptTemplate(context.active(), {
            template: "hello {name}",
            variables,
            version: "V1.0",
          }),
          { sessionId: "session-id" },
        ),
        { key: "value" },
      ),
      () => {
        const attributes = getAttributesFromContext(context.active());
        expect(attributes).toStrictEqual({
          [PROMPT_TEMPLATE_TEMPLATE]: "hello {name}",
          [PROMPT_TEMPLATE_VARIABLES]: JSON.stringify(variables),
          [PROMPT_TEMPLATE_VERSION]: "V1.0",
          [SESSION_ID]: "session-id",
          [METADATA]: JSON.stringify({ key: "value" }),
        });
      },
    );
  });

  it("should have attributes unset outside of the context.with scope", () => {
    context.with(
      setSession(
        setPromptTemplate(context.active(), {
          template: "hello {name}",
          variables: { name: "world" },
          version: "V1.0",
        }),
        { sessionId: "session-id" },
      ),

      () => {
        expect(getSessionId(context.active())).toBe("session-id");
        expect(getPromptTemplate(context.active())).toStrictEqual({
          [PROMPT_TEMPLATE_TEMPLATE]: "hello {name}",
          [PROMPT_TEMPLATE_VARIABLES]: JSON.stringify({ name: "world" }),
          [PROMPT_TEMPLATE_VERSION]: "V1.0",
        });
      },
    );
    expect(getAttributesFromContext(context.active())).toStrictEqual({});
  });

  it("should ignore context attributes that cannot be set as span attributes (non primitive)", () => {
    context.with(
      context
        .active()
        .setValue(ContextAttributes["session.id"], { test: "test" }),
      () => {
        expect(
          context.active().getValue(ContextAttributes["session.id"]),
        ).toStrictEqual({ test: "test" });
        const attributes = getAttributesFromContext(context.active());
        expect(attributes).toStrictEqual({});
      },
    );
  });
});
