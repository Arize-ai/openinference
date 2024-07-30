import { SESSION_ID } from "@arizeai/openinference-semantic-conventions";
import { Context, createContextKey } from "@opentelemetry/api";

export const SESSION_ID_KEY = createContextKey(
  `OpenInference SDK Context Key ${SESSION_ID}`,
);

export function setSessionId(context: Context, sessionId: string): Context {
  return context.setValue(SESSION_ID_KEY, sessionId);
}

export function clearSessionId(context: Context): Context {
  return context.deleteValue(SESSION_ID_KEY);
}

export function getSessionId(context: Context): string | undefined {
  const maybeSessionId = context.getValue(SESSION_ID_KEY);
  if (typeof maybeSessionId === "string") {
    return maybeSessionId;
  }
}
