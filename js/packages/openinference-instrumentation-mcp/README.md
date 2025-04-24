# OpenInference Instrumentation for MCP Typescript SDK

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-mcp.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-mcp)

This module provides automatic instrumentation for the [MCP Typescript SDK](https://github.com/modelcontextprotocol/typescript-sdk). which may be used in conjunction
with [@opentelemetry/sdk-trace-node](https://github.com/open-telemetry/opentelemetry-js/tree/main/packages/opentelemetry-sdk-trace-node).

## Installation

```shell
npm install --save @arizeai/openinference-instrumentation-mcp
```

## Usage

To load the MCP instrumentation, manually instrument the `@modelcontextprotocol/sdk/client/index` and/or `@modelcontextprotocol/sdk/server/index` module.
The client and server must be manually instrumented due to the non-traditional module structure in `@modelcontextprotocol/sdk`. Additional instrumentations can
be registered as usual using the `registerInstrumentations` function.

For example, if using stdio transport,

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { MCPInstrumentation } from "@arizeai/openinference-instrumentation-mcp";
import * as MCPClientStdioModule from "@modelcontextprotocol/sdk/client/stdio";
import * as MCPServerStdioModule from "@modelcontextprotocol/sdk/server/stdio";

const provider = new NodeTracerProvider();
provider.register();

const mcpInstrumentation = new MCPInstrumentation();
// MCP must be manually instrumented as it doesn't have a traditional module structure
mcpInstrumentation.manuallyInstrument({
  clientStdioModule: MCPClientStdioModule,
  serverStdioModule: MCPServerStdioModule,
});
```

For more information on OpenTelemetry Node.js SDK, see the [OpenTelemetry Node.js SDK documentation](https://opentelemetry.io/docs/instrumentation/js/getting-started/nodejs/).
