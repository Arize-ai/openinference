import { context, ContextManager } from "@opentelemetry/api";
import { AsyncHooksContextManager } from "@opentelemetry/context-async-hooks";
import {
  clearSessionId,
  getSessionId,
  setSessionId,
} from "../src/trace/session";

describe("session", () => {
  let contextManager: ContextManager;
  beforeEach(() => {
    contextManager = new AsyncHooksContextManager().enable();
    context.setGlobalContextManager(contextManager);
  });
  afterEach(() => {
    context.disable();
  });

  it("should set session id in the context", () => {
    context.with(setSessionId(context.active(), "session-id"), async () => {
      expect(getSessionId(context.active())).toBe("session-id");
    });
  });

  it("should delete session id from the context", () => {
    context.with(setSessionId(context.active(), "session-id"), async () => {
      expect(getSessionId(context.active())).toBe("session-id");
      const ctx = clearSessionId(context.active());
      expect(getSessionId(ctx)).toBeUndefined();
    });
  });
});
