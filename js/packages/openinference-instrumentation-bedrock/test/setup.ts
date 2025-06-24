import { Polly } from "@pollyjs/core";
import NodeHttpAdapter from "@pollyjs/adapter-node-http";

// Register adapters (no persistence for initial implementation)
Polly.register(NodeHttpAdapter);

// Global test timeout for VCR interactions
jest.setTimeout(30000);