import { context, ContextManager, Attributes } from "@opentelemetry/api";
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
  getSession,
  setSession,
  setPromptTemplate,
  clearPromptTemplate,
  getPromptTemplate,
  setMetadata,
  getMetadata,
  clearMetadata,
  getAttributesFromContext,
  ContextAttributes,
  UserAttributes,
  setUser,
  getUser,
  TagAttributes,
  setTags,
  getTags,
  clearTags,
  clearUser,
  setAttributes,
  getAttributes,
  clearAttributes,
} from "../../src";

describe("promptTemplate context", () => {
  let contextManager: ContextManager;
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });
  it("should set prompt template on the context", () => {
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
          template: "hello {name}",
          variables,
          version: "V1.0",
        });
      },
    );
  });

  it("should delete prompt template from the context", () => {
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
          template: "hello {name}",
          variables,
          version: "V1.0",
        });
        const ctx = clearPromptTemplate(context.active());
        expect(getPromptTemplate(ctx)).toBeUndefined();
      },
    );
  });
});

describe("session context", () => {
  let contextManager: ContextManager;
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });

  it("should set the session in the context", () => {
    context.with(
      setSession(context.active(), { sessionId: "session-id" }),
      () => {
        expect(getSession(context.active())).toStrictEqual({
          sessionId: "session-id",
        });
      },
    );
  });

  it("should delete the session from the context", () => {
    context.with(
      setSession(context.active(), { sessionId: "session-id" }),
      () => {
        expect(getSession(context.active())).toStrictEqual({
          sessionId: "session-id",
        });
        const ctx = clearSession(context.active());
        expect(getSession(ctx)).toBeUndefined();
      },
    );
  });
});

describe("metadata context", () => {
  let contextManager: ContextManager;
  const metadata = {
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
  it("should set metadata on the context", () => {
    context.with(setMetadata(context.active(), metadata), () => {
      expect(getMetadata(context.active())).toStrictEqual(metadata);
    });
  });
  it("should delete metadata from the context", () => {
    context.with(setMetadata(context.active(), metadata), () => {
      expect(getMetadata(context.active())).toStrictEqual(metadata);
      const ctx = clearMetadata(context.active());
      expect(getMetadata(ctx)).toBeUndefined();
    });
  });
});

describe("user context", () => {
  let contextManager: ContextManager;
  const userAttributes: UserAttributes = { userId: "user-id" };
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });
  it("should set user on the context", () => {
    context.with(setUser(context.active(), userAttributes), () => {
      expect(getUser(context.active())).toStrictEqual(userAttributes);
    });
  });
  it("should delete user from the context", () => {
    context.with(setUser(context.active(), userAttributes), () => {
      expect(getUser(context.active())).toStrictEqual(userAttributes);
      const ctx = clearUser(context.active());
      expect(getUser(ctx)).toBeUndefined();
    });
  });
});

describe("tags context", () => {
  let contextManager: ContextManager;
  const tagsAttributes: TagAttributes = ["tag1", "tag2"];
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });
  it("should set tags on the context", () => {
    context.with(setTags(context.active(), tagsAttributes), () => {
      expect(getTags(context.active())).toStrictEqual(tagsAttributes);
    });
  });
  it("should delete tags from the context", () => {
    context.with(setTags(context.active(), tagsAttributes), () => {
      expect(getTags(context.active())).toStrictEqual(tagsAttributes);
      const ctx = clearTags(context.active());
      expect(getTags(ctx)).toBeUndefined();
    });
  });
});

describe("attributes context", () => {
  let contextManager: ContextManager;
  const attributes: Attributes = { hello: "world", test: "test" };
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });
  it("should set attributes attributes on the context", () => {
    context.with(setAttributes(context.active(), attributes), () => {
      expect(getAttributes(context.active())).toStrictEqual(attributes);
    });
  });
  it("should delete attributes attributes from the context", () => {
    context.with(setAttributes(context.active(), attributes), () => {
      expect(getAttributes(context.active())).toStrictEqual(attributes);
      const ctx = clearAttributes(context.active());
      expect(getAttributes(ctx)).toBeUndefined();
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
        expect(getSession(context.active())).toStrictEqual({
          sessionId: "session-id",
        });
        expect(getPromptTemplate(context.active())).toStrictEqual({
          template: "hello {name}",
          variables: { name: "world" },
          version: "V1.0",
        });
      },
    );
  });
});

describe("getContextAttributes", () => {
  const variables = {
    name: "world",
  };
  let contextManager: ContextManager;
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });

  it("should get all attributes off of the context", () => {
    context.with(
      setAttributes(
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
        { test: "attribute", test2: "attribute2" },
      ),
      () => {
        const attributes = getAttributesFromContext(context.active());
        expect(attributes).toStrictEqual({
          [PROMPT_TEMPLATE_TEMPLATE]: "hello {name}",
          [PROMPT_TEMPLATE_VARIABLES]: JSON.stringify(variables),
          [PROMPT_TEMPLATE_VERSION]: "V1.0",
          [SESSION_ID]: "session-id",
          [METADATA]: JSON.stringify({ key: "value" }),
          test: "attribute",
          test2: "attribute2",
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
        const attributes = getAttributesFromContext(context.active());
        expect(attributes).toStrictEqual({
          [PROMPT_TEMPLATE_TEMPLATE]: "hello {name}",
          [PROMPT_TEMPLATE_VARIABLES]: JSON.stringify(variables),
          [PROMPT_TEMPLATE_VERSION]: "V1.0",
          [SESSION_ID]: "session-id",
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
