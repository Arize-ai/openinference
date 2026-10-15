import type { Attributes } from "@opentelemetry/api";
import { context, diag, SpanKind, SpanStatusCode, trace } from "@opentelemetry/api";

import type { OITracer } from "@arizeai/openinference-core";
import {
  getInputAttributes,
  getOutputAttributes,
  getToolAttributes,
} from "@arizeai/openinference-core";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { MODULE_NAME } from "./constants";
import { recordSpanError } from "./spanLifecycle";
import type { ToolRunnerParams, ToolRunner, RunnableTool } from "./types";
import { safelyGetAttributes, toJSONOutput } from "./utils";

/**
 * type-guard for a tool the tool runner runs on the client: a tool definition
 * with the `run` and `parse` the runner calls. Toolsets, which have a `run`
 * but no `parse`, are left alone.
 */
function isRunnableTool(tool: ToolRunnerParams["tools"][number]): tool is RunnableTool {
  return (
    tool != null &&
    typeof tool === "object" &&
    "run" in tool &&
    typeof tool.run === "function" &&
    "parse" in tool &&
    typeof tool.parse === "function"
  );
}

/**
 * Traces the runnable tools of one tool runner, while keeping the caller's own
 * tool objects everywhere the caller can see them.
 *
 * The runner calls `tool.run()` from module-private code, on the tools of its
 * private params, so the only place to trace a run from is the tool object
 * itself. Each runnable tool is therefore handed to the runner as a wrapper that
 * traces `run`, and the runner's public surface maps between the two:
 * `params` and the `setMessagesParams()` mutator show the caller's tools, and
 * the tools passed to `setMessagesParams()` and `addTools()` are traced too.
 */
export class RunnableToolTracer {
  private readonly traced = new WeakMap<object, RunnableTool>();
  private readonly originals = new WeakMap<object, RunnableTool>();

  constructor(private readonly oiTracer: OITracer) {}

  /** The traced wrapper of a runnable tool, the same one each time */
  trace<Tool extends ToolRunnerParams["tools"][number]>(tool: Tool): Tool | RunnableTool {
    if (!isRunnableTool(tool) || this.originals.has(tool)) {
      return tool;
    }
    let wrapper = this.traced.get(tool);
    if (wrapper == null) {
      wrapper = traceRunnableTool({ tool, oiTracer: this.oiTracer });
      this.traced.set(tool, wrapper);
      this.originals.set(wrapper, tool);
    }
    return wrapper;
  }

  /** The caller's tool behind a traced wrapper */
  original<Tool extends ToolRunnerParams["tools"][number]>(tool: Tool): Tool | RunnableTool {
    return (tool != null && typeof tool === "object" && this.originals.get(tool)) || tool;
  }

  traceParams(params: ToolRunnerParams): ToolRunnerParams {
    try {
      if (!Array.isArray(params?.tools)) {
        return params;
      }
      return { ...params, tools: params.tools.map((tool) => this.trace(tool)) };
    } catch (error) {
      diag.warn(`Failed to trace the ${MODULE_NAME} tool runner's tools`, error);
      return params;
    }
  }

  originalParams(params: ToolRunnerParams): ToolRunnerParams {
    if (!Array.isArray(params?.tools)) {
      return params;
    }
    return { ...params, tools: params.tools.map((tool) => this.original(tool)) };
  }
}

/**
 * Wraps a runnable tool in an object whose `run` is a TOOL span, a child of the
 * span active when the runner calls it. Every other property is read from the
 * tool itself, with its methods bound to it, so a class tool keeps working with
 * private (`#`) fields, and a frozen tool can be wrapped too.
 */
function traceRunnableTool({
  tool,
  oiTracer,
}: {
  tool: RunnableTool;
  oiTracer: OITracer;
}): RunnableTool {
  const run = tool.run.bind(tool);
  const tracedRun: RunnableTool["run"] = (args, runContext) => {
    const span = oiTracer.startSpan(tool.name, {
      kind: SpanKind.INTERNAL,
      attributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
        ...safelyGetAttributes(() => getRunnableToolAttributes({ tool, args })),
      },
    });
    let result: ReturnType<RunnableTool["run"]>;
    try {
      result = context.with(trace.setSpan(context.active(), span), () => run(args, runContext));
    } catch (error) {
      recordSpanError(span, error);
      span.end();
      throw error;
    }
    return Promise.resolve(result).then(
      (output) => {
        span.setAttributes(
          safelyGetAttributes(() =>
            getOutputAttributes(typeof output === "string" ? output : toJSONOutput(output)),
          ),
        );
        span.setStatus({ code: SpanStatusCode.OK });
        span.end();
        return output;
      },
      (error: unknown) => {
        recordSpanError(span, error);
        span.end();
        throw error;
      },
    );
  };
  // A plain object rather than a Proxy: a Proxy must return a frozen tool's
  // own `run` and methods unchanged, so it cannot trace a frozen tool.
  const wrapper: RunnableTool = Object.create(Object.getPrototypeOf(tool));
  const boundMethods = new Map<unknown, unknown>();
  for (const [property, descriptor] of getToolProperties(tool)) {
    Object.defineProperty(wrapper, property, {
      configurable: true,
      enumerable: descriptor.enumerable ?? false,
      get: () => {
        const value: unknown = Reflect.get(tool, property, tool);
        if (typeof value !== "function") {
          return value;
        }
        let bound = boundMethods.get(value);
        if (bound == null) {
          bound = value.bind(tool);
          boundMethods.set(value, bound);
        }
        return bound;
      },
    });
  }
  Object.defineProperty(wrapper, "run", {
    configurable: true,
    enumerable: Object.getOwnPropertyDescriptor(tool, "run")?.enumerable ?? false,
    writable: true,
    value: tracedRun,
  });
  return wrapper;
}

/**
 * The properties of a tool, its own and those of its classes, each with the
 * descriptor seen first, as `tool.property` resolves them.
 */
function getToolProperties(tool: object): Map<PropertyKey, PropertyDescriptor> {
  const properties = new Map<PropertyKey, PropertyDescriptor>();
  let current: object | null = tool;
  while (current != null && current !== Object.prototype) {
    for (const property of Reflect.ownKeys(current)) {
      const descriptor = Object.getOwnPropertyDescriptor(current, property);
      if (property !== "constructor" && descriptor != null && !properties.has(property)) {
        properties.set(property, descriptor);
      }
    }
    current = Object.getPrototypeOf(current);
  }
  return properties;
}

/**
 * The attributes of a tool run: the tool, its parameters schema and the
 * arguments it was called with.
 */
function getRunnableToolAttributes({
  tool,
  args,
}: {
  tool: RunnableTool;
  args: unknown;
}): Attributes {
  const description =
    "description" in tool && typeof tool.description === "string" && tool.description
      ? tool.description
      : undefined;
  const inputSchema: unknown = "input_schema" in tool ? tool.input_schema : undefined;
  // A tool without a schema of its own (e.g. a typed client tool) has no
  // parameters to record, which the core helper always records.
  const toolAttributes: Attributes = isStringRecord(inputSchema)
    ? getToolAttributes({ name: tool.name, description, parameters: inputSchema })
    : {
        [SemanticConventions.TOOL_NAME]: tool.name,
        ...(description != null && { [SemanticConventions.TOOL_DESCRIPTION]: description }),
      };
  return { ...toolAttributes, ...getInputAttributes(toJSONOutput(args)) };
}

function isStringRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Maps the runner's public view of its params back to the caller's tools, and
 * traces the tools given to it later, see {@link RunnableToolTracer}.
 */
export function instrumentToolRunnerTools({
  runner,
  tools,
}: {
  runner: ToolRunner;
  tools: RunnableToolTracer;
}) {
  const prototype: unknown = Object.getPrototypeOf(runner);
  if (prototype == null || typeof prototype !== "object") {
    return;
  }
  Object.defineProperty(runner, "params", {
    configurable: true,
    get: () => {
      const params: ToolRunnerParams = Reflect.get(prototype, "params", runner);
      return tools.originalParams(params);
    },
  });

  // setMessagesParams() and addTools() are wrapped only where the SDK has
  // them: SDKs before 0.133 have no addTools().
  if (typeof runner.setMessagesParams === "function") {
    instrumentSetMessagesParams({ runner, tools });
  }
  if (typeof runner.addTools === "function") {
    const addTools = runner.addTools.bind(runner);
    Object.defineProperty(runner, "addTools", {
      configurable: true,
      writable: true,
      value: (...added: Parameters<ToolRunner["addTools"]>) =>
        addTools(...added.map((tool) => tools.trace(tool))),
    });
  }
}

/**
 * Traces the tools of the params given to `setMessagesParams()`, and hands its
 * mutator the caller's tools.
 */
function instrumentSetMessagesParams({
  runner,
  tools,
}: {
  runner: ToolRunner;
  tools: RunnableToolTracer;
}) {
  const setMessagesParams = runner.setMessagesParams.bind(runner);
  Object.defineProperty(runner, "setMessagesParams", {
    configurable: true,
    writable: true,
    value: (
      paramsOrMutator: ToolRunnerParams | ((prevParams: ToolRunnerParams) => ToolRunnerParams),
    ) => {
      if (typeof paramsOrMutator === "function") {
        setMessagesParams((prevParams) =>
          tools.traceParams(paramsOrMutator(tools.originalParams(prevParams))),
        );
      } else {
        setMessagesParams(tools.traceParams(paramsOrMutator));
      }
    },
  });
}
