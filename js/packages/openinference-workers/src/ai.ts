import type {
  Ai,
  AiTextGenerationToolOutput,
  RoleScopedChatInput,
  ExecutionContext,
} from "@cloudflare/workers-types";
import { context, diag, SpanStatusCode, type Span } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";

import {
  getInputAttributes,
  getOutputAttributes,
  type OITracer,
} from "@arizeai/openinference-core";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions as SC,
} from "@arizeai/openinference-semantic-conventions";

export interface InstrumentAiOptions<T extends Pick<Ai, "run">> {
  ai: T;
  /** Text-generation model IDs to trace; other AI tasks pass through unchanged. */
  models: readonly string[];
  tracer: OITracer;
  /** Supply the current request's execution context, including for streamed results. */
  execution: Pick<ExecutionContext, "waitUntil">;
  flush: () => Promise<void>;
  /** Maximum captured stream characters; larger outputs are omitted, not truncated into invalid JSON. */
  maxStreamCaptureLength?: number;
}

// Gateway model responses are deliberately open-ended in Cloudflare's SDK.
function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function object(value: unknown): Record<string, unknown> | undefined {
  return isObject(value) ? value : undefined;
}
function safely(fn: () => void) {
  try {
    fn();
  } catch {
    diag.debug("OpenInference could not read AI attributes");
  }
}
function messages(span: Span, prefix: string, values: unknown) {
  if (!Array.isArray(values)) return;
  values.forEach((value: unknown, i) => {
    const m = object(value);
    if (!m) return;
    for (const [key, attribute] of [
      ["role", SC.MESSAGE_ROLE],
      ["content", SC.MESSAGE_CONTENT],
      ["name", SC.MESSAGE_NAME],
      ["tool_call_id", SC.MESSAGE_TOOL_CALL_ID],
    ] as const) {
      if (typeof m[key] === "string") span.setAttribute(`${prefix}.${i}.${attribute}`, m[key]);
    }
    if (Array.isArray(m.tool_calls))
      m.tool_calls.forEach((value: unknown, j) => {
        const call = object(value);
        if (!call) return;
        const fn = object(call.function) ?? call;
        const p = `${prefix}.${i}.${SC.MESSAGE_TOOL_CALLS}.${j}`;
        if (typeof call.id === "string") span.setAttribute(`${p}.${SC.TOOL_CALL_ID}`, call.id);
        if (typeof fn.name === "string")
          span.setAttribute(`${p}.${SC.TOOL_CALL_FUNCTION_NAME}`, fn.name);
        if (fn.arguments !== undefined)
          span.setAttribute(
            `${p}.${SC.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`,
            typeof fn.arguments === "string" ? fn.arguments : JSON.stringify(fn.arguments),
          );
      });
  });
}
function output(span: Span, value: unknown) {
  const result = object(value);
  if (!result) return;
  span.setAttributes(
    getOutputAttributes({ value: JSON.stringify(value), mimeType: MimeType.JSON }),
  );
  if (typeof result.model === "string") span.setAttribute(SC.LLM_MODEL_NAME, result.model);
  if (typeof result.response === "string" || result.tool_calls) {
    messages(span, SC.LLM_OUTPUT_MESSAGES, [
      { role: "assistant", content: result.response, tool_calls: result.tool_calls },
    ]);
  }
  if (Array.isArray(result.choices)) {
    messages(
      span,
      SC.LLM_OUTPUT_MESSAGES,
      result.choices.map((choice: unknown) => object(choice)?.message),
    );
  }
  const usage = object(result.usage);
  if (usage)
    for (const [key, attr] of [
      ["prompt_tokens", SC.LLM_TOKEN_COUNT_PROMPT],
      ["completion_tokens", SC.LLM_TOKEN_COUNT_COMPLETION],
      ["total_tokens", SC.LLM_TOKEN_COUNT_TOTAL],
    ] as const) {
      if (typeof usage[key] === "number") span.setAttribute(attr, usage[key]);
    }
}

/** Wrap a Cloudflare AI binding without changing its overloads or mutating it.
 * Instruments text generation (messages/prompt); batch, raw Response, and WebSocket
 * modes pass through. Create the wrapper per request so waitUntil uses that request.
 */
export function instrumentAi<T extends Pick<Ai, "run">>({
  ai,
  models,
  tracer,
  execution,
  flush,
  maxStreamCaptureLength = 65_536,
}: InstrumentAiOptions<T>): T {
  if (!Number.isSafeInteger(maxStreamCaptureLength) || maxStreamCaptureLength < 0)
    throw new RangeError("maxStreamCaptureLength must be a non-negative safe integer");
  // The apply trap always restores the original binding as the method receiver.
  // oxlint-disable-next-line typescript/unbound-method
  const run = new Proxy(ai.run, {
    apply(original, _receiver, args: unknown[]) {
      const [model, input, options] = args;
      const data = object(input);
      const opts = object(options);
      if (
        !models.includes(String(model)) ||
        isTracingSuppressed(context.active()) ||
        !data ||
        (!Array.isArray(data.messages) && typeof data.prompt !== "string") ||
        opts?.queueRequest ||
        opts?.returnRawResponse ||
        opts?.websocket
      ) {
        return Reflect.apply(original, ai, args);
      }
      return tracer.startActiveSpan(
        "Workers AI.run",
        {
          attributes: {
            [SC.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
            [SC.LLM_MODEL_NAME]: String(model),
            [SC.LLM_SYSTEM]: "cloudflare",
          },
        },
        async (span) => {
          let ended = false;
          let resolveDone: () => void = () => {};
          const done = new Promise<void>((resolve) => {
            resolveDone = resolve;
          });
          execution.waitUntil(
            done.then(flush).catch(() => {
              diag.warn("OpenInference AI flush failed");
            }),
          );
          const finish = (failed = false) => {
            if (ended) return;
            ended = true;
            if (failed) {
              // Provider error bodies can echo private prompts; keep diagnostics content-free.
              span.setStatus({
                code: SpanStatusCode.ERROR,
                message: "Workers AI call failed or was cancelled",
              });
              span.recordException(new Error("Workers AI call failed or was cancelled"));
            }
            span.end();
            resolveDone();
          };
          safely(() => {
            span.setAttributes(
              getInputAttributes({ value: JSON.stringify(input), mimeType: MimeType.JSON }),
            );
            messages(
              span,
              SC.LLM_INPUT_MESSAGES,
              data.messages ?? [{ role: "user", content: data.prompt }],
            );
            const parameters = Object.fromEntries(
              Object.entries(data).filter(
                ([key]) => !["prompt", "messages", "tools", "functions"].includes(key),
              ),
            );
            span.setAttribute(SC.LLM_INVOCATION_PARAMETERS, JSON.stringify(parameters));
            if (Array.isArray(data.tools))
              data.tools.forEach((tool: unknown, i) =>
                span.setAttribute(
                  `${SC.LLM_TOOLS}.${i}.${SC.TOOL_JSON_SCHEMA}`,
                  JSON.stringify(tool),
                ),
              );
          });
          try {
            const result: unknown = await Reflect.apply(original, ai, args);
            if (result instanceof ReadableStream) {
              return captureStream({ stream: result, span, finish, limit: maxStreamCaptureLength });
            }
            safely(() => output(span, result));
            finish();
            return result;
          } catch (error) {
            finish(true);
            throw error;
          }
        },
      );
    },
  });
  return new Proxy(ai, {
    get(target, key) {
      if (key === "run") return run;
      const value: unknown = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

function captureStream({
  stream,
  span,
  finish,
  limit,
}: {
  stream: ReadableStream<Uint8Array>;
  span: Span;
  finish: (failed?: boolean) => void;
  limit: number;
}) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let captured = "";
  let overflow = false;
  function complete() {
    safely(() => {
      if (overflow || captured.length > limit) {
        span.setAttribute("cloudflare.ai.output_omitted", true);
        return;
      }
      let text = "";
      let last: Record<string, unknown> = {};
      const tools: unknown[] = [];
      const choices = new Map<number, StreamMessage>();
      for (const event of captured.replace(/\r\n/g, "\n").split("\n\n")) {
        const payload = event
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (!payload || payload === "[DONE]") continue;
        const chunk = object(JSON.parse(payload));
        if (!chunk) continue;
        if (chunk.error) {
          finish(true);
          return;
        }
        if (typeof chunk.response === "string") text += chunk.response;
        if (Array.isArray(chunk.tool_calls)) tools.push(...chunk.tool_calls);
        mergeChoices({ choices, values: chunk.choices });
        last = { ...last, ...chunk };
      }
      // Transport padding and the last delta are not the assembled model output.
      delete last.p;
      delete last.choices;
      output(span, {
        ...last,
        response: text,
        ...(tools.length ? { tool_calls: tools } : {}),
        ...(choices.size
          ? {
              choices: [...choices.entries()]
                .sort(([a], [b]) => a - b)
                .map(([, message]) => ({ message })),
            }
          : {}),
      });
    });
    finish();
  }
  return new ReadableStream<Uint8Array>(
    {
      async pull(controller) {
        try {
          const { done, value } = await reader.read();
          if (done) {
            captured += decoder.decode();
            complete();
            controller.close();
            reader.releaseLock();
            return;
          }
          if (!overflow) {
            captured += decoder.decode(value, { stream: true });
            if (captured.length > limit) {
              captured = "";
              overflow = true;
            }
          }
          controller.enqueue(value);
        } catch (error) {
          finish(true);
          controller.error(error);
          reader.releaseLock();
        }
      },
      async cancel(reason) {
        finish(true);
        try {
          await reader.cancel(reason);
        } finally {
          reader.releaseLock();
        }
      },
    },
    { highWaterMark: 0 },
  );
}

type StreamMessage = Pick<RoleScopedChatInput, "role" | "content"> & {
  tool_calls: AiTextGenerationToolOutput[];
};
function mergeChoices({
  choices,
  values,
}: {
  choices: Map<number, StreamMessage>;
  values: unknown;
}) {
  if (!Array.isArray(values)) return;
  for (const value of values) {
    const choice = object(value);
    if (!choice) continue;
    const index = typeof choice.index === "number" ? choice.index : 0;
    const delta = object(choice.delta);
    if (!delta) continue;
    const message = choices.get(index) ?? { role: "assistant", content: "", tool_calls: [] };
    if (typeof delta.role === "string") message.role = delta.role;
    if (typeof delta.content === "string") message.content += delta.content;
    mergeToolCalls({ tools: message.tool_calls, values: delta.tool_calls });
    choices.set(index, message);
  }
}
function mergeToolCalls({
  tools,
  values,
}: {
  tools: AiTextGenerationToolOutput[];
  values: unknown;
}) {
  if (!Array.isArray(values)) return;
  for (const value of values) {
    const call = object(value);
    if (
      !call ||
      typeof call.index !== "number" ||
      !Number.isSafeInteger(call.index) ||
      call.index < 0 ||
      call.index > 1024
    )
      continue;
    const tool = tools[call.index] ?? {
      id: "",
      type: "function",
      function: { name: "", arguments: "" },
    };
    const fn = object(call.function);
    if (typeof call.id === "string") tool.id = call.id;
    if (typeof fn?.name === "string") tool.function.name += fn.name;
    if (typeof fn?.arguments === "string") tool.function.arguments += fn.arguments;
    tools[call.index] = tool;
  }
}
