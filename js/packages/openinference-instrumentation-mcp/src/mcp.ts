import { context, diag, propagation } from "@opentelemetry/api";
import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import type {
  JSONRPCRequest,
  Notification,
  Request,
  Result,
} from "@modelcontextprotocol/sdk/types";
import type { Client } from "@modelcontextprotocol/sdk/client/index";
import type * as ClientModule from "@modelcontextprotocol/sdk/client/index";
import type { Server } from "@modelcontextprotocol/sdk/server/index";
import type * as ServerModule from "@modelcontextprotocol/sdk/server/index";

import { VERSION } from "./version";

const CLIENT_MODULE_NAME = "@modelcontextprotocol/sdk/client/index";
const SERVER_MODULE_NAME = "@modelcontextprotocol/sdk/server/index";

/**
 * Flag to check if the client module has been patched
 * Note: This is a fallback in case the module is made immutable (e.x. Deno, webpack, etc.)
 */
let _isClientOpenInferencePatched = false;
/**
 * function to check if client instrumentation is enabled / disabled
 */
export function isClientPatched() {
  return _isClientOpenInferencePatched;
}
/**
 * Flag to check if the server module has been patched
 * Note: This is a fallback in case the module is made immutable (e.x. Deno, webpack, etc.)
 */
let _isServerOpenInferencePatched = false;
/**
 * function to check if server instrumentation is enabled / disabled
 */
export function isServerPatched() {
  return _isServerOpenInferencePatched;
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
   * @param {typeof ClientModule} modules.clientModule - The MCP client module, e.g. require('@modelcontextprotocol/sdk/client/index.js')
   */
  public manuallyInstrument({
    clientModule,
    serverModule,
  }: {
    clientModule?: typeof ClientModule;
    serverModule?: typeof ServerModule;
  }) {
    if (clientModule) {
      diag.debug(`Manually instrumenting ${CLIENT_MODULE_NAME}`);
      this._patchClientModule(clientModule);
    }
    if (serverModule) {
      diag.debug(`Manually instrumenting ${SERVER_MODULE_NAME}`);
      this._patchServerModule(serverModule);
    }
  }

  protected override init() {
    return [
      new InstrumentationNodeModuleDefinition(
        "@modelcontextprotocol/sdk/client/index.js",
        [">=1.0.0"],
        this._patchClientModule.bind(this),
        this._unpatchClientModule.bind(this),
      ),
      new InstrumentationNodeModuleDefinition(
        "@modelcontextprotocol/sdk/server/index.js",
        [">=1.0.0"],
        this._patchServerModule.bind(this),
        this._unpatchServerModule.bind(this),
      ),
    ];
  }

  private _patchClientModule(
    module: typeof ClientModule & { openInferencePatched?: boolean },
    moduleVersion?: string,
  ) {
    diag.debug(`Applying patch for ${CLIENT_MODULE_NAME}@${moduleVersion}`);
    if (module?.openInferencePatched || _isClientOpenInferencePatched) {
      return module;
    }

    this._wrap(
      module.Client.prototype,
      "request",
      this._getClientRequestPatch(),
    );

    _isClientOpenInferencePatched = true;
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      module.openInferencePatched = true;
    } catch (e) {
      diag.debug(
        `Failed to set ${CLIENT_MODULE_NAME} patched flag on the module`,
        e,
      );
    }

    return module;
  }

  private _unpatchClientModule(moduleExports: typeof ClientModule) {
    this._unwrap(moduleExports.Client.prototype, "request");
    return moduleExports;
  }

  private _getClientRequestPatch<
    SendRequestT extends Request,
    SendNotificationT extends Notification,
    SendResultT extends Result,
  >() {
    return (
      original: Client<SendRequestT, SendNotificationT, SendResultT>["request"],
    ) => {
      return function request(
        this: Client<SendRequestT, SendNotificationT, SendResultT>,
        ...args: Parameters<
          Client<SendRequestT, SendNotificationT, SendResultT>["request"]
        >
      ) {
        const [request] = args;
        request.method;
        if (!request.params) {
          request.params = {};
        }
        if (!request.params._meta) {
          request.params._meta = {};
        }
        propagation.inject(context.active(), request.params._meta);
        return original.apply(this, args);
      };
    };
  }

  private _patchServerModule(
    module: typeof ServerModule & { openInferencePatched?: boolean },
    moduleVersion?: string,
  ) {
    diag.debug(`Applying patch for ${SERVER_MODULE_NAME}@${moduleVersion}`);
    if (module?.openInferencePatched || _isServerOpenInferencePatched) {
      return module;
    }
    this._wrap(
      module.Server.prototype as unknown as {
        _onrequest: (...args: object[]) => void;
      },
      "_onrequest",
      this._getServerOnRequestPatch(),
    );

    _isServerOpenInferencePatched = true;
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      module.openInferencePatched = true;
    } catch (e) {
      diag.debug(
        `Failed to set ${SERVER_MODULE_NAME} patched flag on the module`,
        e,
      );
    }
    return module;
  }

  private _unpatchServerModule(moduleExports: typeof ServerModule) {
    this._unwrap(
      moduleExports.Server.prototype as unknown as {
        _onrequest: (...args: object[]) => void;
      },
      "_onrequest",
    );
    return moduleExports;
  }

  private _getServerOnRequestPatch<
    SendRequestT extends Request,
    SendNotificationT extends Notification,
    SendResultT extends Result,
  >() {
    return (
      original: Server<
        SendRequestT,
        SendNotificationT,
        SendResultT
      >["_onrequest"],
    ) => {
      return function request(
        this: Server<SendRequestT, SendNotificationT, SendResultT>,
        ...args: Parameters<
          Server<SendRequestT, SendNotificationT, SendResultT>["_onrequest"]
        >
      ) {
        const [request] = args as [JSONRPCRequest];
        const ctx = propagation.extract(
          context.active(),
          request.params?._meta,
        );
        return context.with(ctx, () => {
          return original.apply(this, args);
        });
      };
    };
  }
}
