import { context, ContextManager } from "@opentelemetry/api";
import { AsyncHooksContextManager } from "@opentelemetry/context-async-hooks";
import { getSessionId, setSessionId } from "../src/trace/session";

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
});
