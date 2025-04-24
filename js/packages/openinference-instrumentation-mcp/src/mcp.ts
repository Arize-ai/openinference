import { context, diag, propagation } from "@opentelemetry/api";
import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import type {
  JSONRPCMessage,
  JSONRPCRequest,
} from "@modelcontextprotocol/sdk/types";
import type * as ClientSSEModule from "@modelcontextprotocol/sdk/client/sse";
import type * as ClientStdioModule from "@modelcontextprotocol/sdk/client/stdio";
import type * as ClientStreamableHTTPModule from "@modelcontextprotocol/sdk/client/streamableHttp";
import type * as ServerSSEModule from "@modelcontextprotocol/sdk/server/sse";
import type * as ServerStdioModule from "@modelcontextprotocol/sdk/server/stdio";
import type * as ServerStreamableHTTPModule from "@modelcontextprotocol/sdk/server/streamableHttp";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport";

import { VERSION } from "./version";

const CLIENT_SSE_MODULE_NAME = "@modelcontextprotocol/sdk/client/sse";
const SERVER_SSE_MODULE_NAME = "@modelcontextprotocol/sdk/server/sse";
const CLIENT_STDIO_MODULE_NAME = "@modelcontextprotocol/sdk/client/stdio";
const SERVER_STDIO_MODULE_NAME = "@modelcontextprotocol/sdk/server/stdio";
const CLIENT_STREAMABLE_HTTP_MODULE_NAME =
  "@modelcontextprotocol/sdk/client/streamableHttp";
const SERVER_STREAMABLE_HTTP_MODULE_NAME =
  "@modelcontextprotocol/sdk/server/streamableHttp";

/**
 * Flags to check if a module has already been patched.
 * Note: This is a fallback in case the module is made immutable (e.x. Deno, webpack, etc.)
 */
const patchedModules = {
  [CLIENT_SSE_MODULE_NAME]: false,
  [SERVER_SSE_MODULE_NAME]: false,
  [CLIENT_STDIO_MODULE_NAME]: false,
  [SERVER_STDIO_MODULE_NAME]: false,
  [CLIENT_STREAMABLE_HTTP_MODULE_NAME]: false,
  [SERVER_STREAMABLE_HTTP_MODULE_NAME]: false,
};

/**
 * function to check if instrumentation is enabled / disabled for the module
 */
export function isPatched(moduleName: keyof typeof patchedModules) {
  return patchedModules[moduleName];
}

/**
 * Instrumentation for the MCP SDK which propagates context between client and server to allow
 * traces to be connected across tool calls.
 * @param instrumentationConfig The config for the instrumentation @see {@link InstrumentationConfig}
 */
export class MCPInstrumentation extends InstrumentationBase<InstrumentationConfig> {
  constructor({
    instrumentationConfig,
  }: {
    /**
     * The config for the instrumentation
     * @see {@link InstrumentationConfig}
     */
    instrumentationConfig?: InstrumentationConfig;
  } = {}) {
    super(
      "@arizeai/openinference-instrumentation-mcp",
      VERSION,
      instrumentationConfig,
    );
  }

  /**
   * Manually instruments the MCP client and/or server modules. Currently, auto-instrumentation does not work
   * with the MCP SDK and this method must be used to enable instrumentation.
   * @param {Object} modules - The modules to manually instrument.
   * @param {typeof ClientSSEModule} modules.clientSSEModule - The MCP client SSE module, e.g. require('@modelcontextprotocol/sdk/client/sse.js')
   * @param {typeof ServerSSEModule} modules.serverSSEModule - The MCP server SSE module, e.g. require('@modelcontextprotocol/sdk/server/sse.js')
   * @param {typeof ClientStdioModule} modules.clientStdioModule - The MCP client stdio module, e.g. require('@modelcontextprotocol/sdk/server/stdio.js')
   * @param {typeof ServerStdioModule} modules.serverStdioModule - The MCP server stdio module, e.g. require('@modelcontextprotocol/sdk/client/stdio.js')
   * @param {typeof ClientStreamableHTTPModule} modules.clientStreamableHTTPModule - The MCP client streamable HTTP module, e.g. require('@modelcontextprotocol/sdk/client/streamableHttp.js')
   * @param {typeof ServerStreamableHTTPModule} modules.serverStreamableHTTPModule - The MCP server streamable HTTP module, e.g. require('@modelcontextprotocol/sdk/server/streamableHttp.js')
   */
  public manuallyInstrument({
    clientSSEModule,
    serverSSEModule,
    clientStdioModule,
    serverStdioModule,
    clientStreamableHTTPModule,
    serverStreamableHTTPModule,
  }: {
    clientSSEModule?: typeof ClientSSEModule;
    serverSSEModule?: typeof ServerSSEModule;
    clientStdioModule?: typeof ClientStdioModule;
    serverStdioModule?: typeof ServerStdioModule;
    clientStreamableHTTPModule?: typeof ClientStreamableHTTPModule;
    serverStreamableHTTPModule?: typeof ServerStreamableHTTPModule;
  }) {
    if (clientSSEModule) {
      diag.debug(`Manually instrumenting ${CLIENT_SSE_MODULE_NAME}`);
      this._patchClientSSEModule(clientSSEModule);
    }
    if (serverSSEModule) {
      diag.debug(`Manually instrumenting ${SERVER_SSE_MODULE_NAME}`);
      this._patchServerSSEModule(serverSSEModule);
    }
    if (clientStdioModule) {
      diag.debug(`Manually instrumenting ${CLIENT_STDIO_MODULE_NAME}`);
      this._patchClientStdioModule(clientStdioModule);
    }
    if (serverStdioModule) {
      diag.debug(`Manually instrumenting ${SERVER_STDIO_MODULE_NAME}`);
      this._patchServerStdioModule(serverStdioModule);
    }
    if (clientStreamableHTTPModule) {
      diag.debug(
        `Manually instrumenting ${CLIENT_STREAMABLE_HTTP_MODULE_NAME}`,
      );
      this._patchClientStreamableHTTPModule(clientStreamableHTTPModule);
    }
    if (serverStreamableHTTPModule) {
      diag.debug(
        `Manually instrumenting ${SERVER_STREAMABLE_HTTP_MODULE_NAME}`,
      );
      this._patchServerStreamableHTTPModule(serverStreamableHTTPModule);
    }
  }

  protected override init() {
    return [
      new InstrumentationNodeModuleDefinition(
        "@modelcontextprotocol/sdk/client/sse.js",
        [">=1.0.0"],
        this._patchClientSSEModule.bind(this),
        this._unpatchClientSSEModule.bind(this),
      ),
      new InstrumentationNodeModuleDefinition(
        "@modelcontextprotocol/sdk/server/sse.js",
        [">=1.0.0"],
        this._patchServerSSEModule.bind(this),
        this._unpatchServerSSEModule.bind(this),
      ),
      new InstrumentationNodeModuleDefinition(
        "@modelcontextprotocol/sdk/client/stdio.js",
        [">=1.0.0"],
        this._patchClientStdioModule.bind(this),
        this._unpatchClientStdioModule.bind(this),
      ),
      new InstrumentationNodeModuleDefinition(
        "@modelcontextprotocol/sdk/server/stdio.js",
        [">=1.0.0"],
        this._patchServerStdioModule.bind(this),
        this._unpatchServerStdioModule.bind(this),
      ),
      new InstrumentationNodeModuleDefinition(
        "@modelcontextprotocol/sdk/client/streamableHttp.js",
        [">=1.10.0"],
        this._patchClientStreamableHTTPModule.bind(this),
        this._unpatchClientStreamableHTTPModule.bind(this),
      ),
      new InstrumentationNodeModuleDefinition(
        "@modelcontextprotocol/sdk/server/streamableHttp.js",
        [">=1.10.0"],
        this._patchServerStreamableHTTPModule.bind(this),
        this._unpatchServerStreamableHTTPModule.bind(this),
      ),
    ];
  }

  private _patchClientSSEModule(
    module: typeof ClientSSEModule,
    moduleVersion?: string,
  ) {
    return this._patchTransport(
      module,
      CLIENT_SSE_MODULE_NAME,
      moduleVersion,
      module.SSEClientTransport,
    );
  }

  private _unpatchClientSSEModule(module: typeof ClientSSEModule) {
    return this._unpatchTransport(
      module,
      CLIENT_SSE_MODULE_NAME,
      module.SSEClientTransport,
    );
  }

  private _patchServerSSEModule(
    module: typeof ServerSSEModule,
    moduleVersion?: string,
  ) {
    return this._patchTransport(
      module,
      SERVER_SSE_MODULE_NAME,
      moduleVersion,
      module.SSEServerTransport,
    );
  }

  private _unpatchServerSSEModule(module: typeof ServerSSEModule) {
    return this._unpatchTransport(
      module,
      SERVER_SSE_MODULE_NAME,
      module.SSEServerTransport,
    );
  }

  private _patchClientStdioModule(
    module: typeof ClientStdioModule,
    moduleVersion?: string,
  ) {
    return this._patchTransport(
      module,
      CLIENT_STDIO_MODULE_NAME,
      moduleVersion,
      module.StdioClientTransport,
    );
  }

  private _unpatchClientStdioModule(module: typeof ClientStdioModule) {
    return this._unpatchTransport(
      module,
      CLIENT_STDIO_MODULE_NAME,
      module.StdioClientTransport,
    );
  }

  private _patchServerStdioModule(
    module: typeof ServerStdioModule,
    moduleVersion?: string,
  ) {
    return this._patchTransport(
      module,
      SERVER_STDIO_MODULE_NAME,
      moduleVersion,
      module.StdioServerTransport,
    );
  }

  private _unpatchServerStdioModule(module: typeof ServerStdioModule) {
    return this._unpatchTransport(
      module,
      SERVER_STDIO_MODULE_NAME,
      module.StdioServerTransport,
    );
  }

  private _patchClientStreamableHTTPModule(
    module: typeof ClientStreamableHTTPModule,
    moduleVersion?: string,
  ) {
    return this._patchTransport(
      module,
      CLIENT_STREAMABLE_HTTP_MODULE_NAME,
      moduleVersion,
      module.StreamableHTTPClientTransport,
    );
  }

  private _unpatchClientStreamableHTTPModule(
    module: typeof ClientStreamableHTTPModule,
  ) {
    return this._unpatchTransport(
      module,
      CLIENT_STREAMABLE_HTTP_MODULE_NAME,
      module.StreamableHTTPClientTransport,
    );
  }

  private _patchServerStreamableHTTPModule(
    module: typeof ServerStreamableHTTPModule,
    moduleVersion?: string,
  ) {
    return this._patchTransport(
      module,
      SERVER_STREAMABLE_HTTP_MODULE_NAME,
      moduleVersion,
      module.StreamableHTTPServerTransport,
    );
  }

  private _unpatchServerStreamableHTTPModule(
    module: typeof ServerStreamableHTTPModule,
  ) {
    return this._unpatchTransport(
      module,
      SERVER_STREAMABLE_HTTP_MODULE_NAME,
      module.StreamableHTTPServerTransport,
    );
  }

  private _patchTransport<M, U extends Transport>(
    module: M & { openInferencePatched?: boolean },
    moduleName: keyof typeof patchedModules,
    moduleVersion: string | undefined,
    transportClass: { prototype: U },
  ) {
    diag.debug(`Applying patch for ${moduleName}@${moduleVersion}`);
    if (module?.openInferencePatched || patchedModules[moduleName]) {
      return module;
    }

    this._wrap(transportClass.prototype, "send", this._getTransportSendPatch());

    this._wrap(
      transportClass.prototype,
      "start",
      this._getTransportStartPatch(),
    );

    patchedModules[moduleName] = true;
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      module.openInferencePatched = true;
    } catch (e) {
      diag.debug(`Failed to set ${moduleName} patched flag on the module`, e);
    }

    return module;
  }

  private _unpatchTransport<M, U extends Transport>(
    module: M & { openInferencePatched?: boolean },
    moduleName: keyof typeof patchedModules,
    transportClass: { prototype: U },
  ) {
    this._unwrap(transportClass.prototype, "send");
    this._unwrap(transportClass.prototype, "start");

    patchedModules[moduleName] = false;
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      module.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${moduleName} patched flag on the module`, e);
    }

    return module;
  }

  private _getTransportSendPatch() {
    return (original: Transport["send"]) => {
      return function send(
        this: Transport,
        ...args: Parameters<Transport["send"]>
      ) {
        const message = args[0];
        if (!MCPInstrumentation._isJSONRPCRequest(message)) {
          return original.apply(this, args);
        }
        if (!message.params) {
          message.params = {};
        }
        if (!message.params._meta) {
          message.params._meta = {};
        }
        propagation.inject(context.active(), message.params._meta);
        return original.apply(this, args);
      };
    };
  }

  private _getTransportStartPatch() {
    return (original: Transport["start"]) => {
      return function start(
        this: Transport,
        ...args: Parameters<Transport["start"]>
      ) {
        const onmessage = this.onmessage;
        if (!onmessage) {
          return original.apply(this, args);
        }

        this.onmessage = (...args) => {
          const message = args[0] as JSONRPCRequest;
          if (!MCPInstrumentation._isJSONRPCRequest(message)) {
            return onmessage.apply(this, args);
          }

          const ctx = propagation.extract(
            context.active(),
            message.params?._meta,
          );
          return context.with(ctx, () => {
            return onmessage.apply(this, args);
          });
        };

        return original.apply(this, args);
      };
    };
  }

  // A request has a method and request id. We check both primarily to differentiate from
  // a notification which only has method. If in the future we find we should propagate context
  // on notifications too, we can loosen the check.
  private static _isJSONRPCRequest(
    message: JSONRPCMessage,
  ): message is JSONRPCRequest {
    return (
      typeof message === "object" &&
      !!message &&
      "method" in message &&
      typeof message.method === "string" &&
      "id" in message
    );
  }
}
