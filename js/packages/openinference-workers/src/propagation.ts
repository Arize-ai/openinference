import { context, propagation, trace, type Context } from "@opentelemetry/api";

/** Copy headers and inject W3C trace context without modifying the caller's headers. */
export function injectTraceHeaders({
  headers = {},
  ctx = context.active(),
}: {
  headers?: HeadersInit;
  ctx?: Context;
} = {}): Record<string, string> {
  const carrier = new Headers(headers);
  propagation.inject(ctx, carrier, {
    set: (target, key, value) => target.set(key, value),
  });
  return Object.fromEntries(carrier.entries());
}

/** Extract an inbound parent, retaining context attributes but never an ambient span. */
export function extractTraceContext({
  headers,
}: {
  headers: Headers | Record<string, string | undefined>;
}): Context {
  const carrier =
    headers instanceof Headers
      ? headers
      : new Headers(
          Object.entries(headers).filter(
            (entry): entry is [string, string] => entry[1] !== undefined,
          ),
        );
  return propagation.extract(trace.deleteSpan(context.active()), carrier, {
    get: (target, key) => target.get(key) ?? undefined,
    keys: (target) => [...target.keys()],
  });
}
