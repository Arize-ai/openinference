/**
 * TypeScript declarations for test-data-generators.js
 */

export interface TestDataOptions {
  initialPrompt?: string;
  toolUseId?: string;
  toolName?: string;
  toolInput?: any;
  toolResult?: string;
  followupPrompt?: string;
  tools?: any[];
  modelId?: string;
  maxTokens?: number;
}

export interface TestDataResult {
  modelId: string;
  body: string;
}

export function generateToolResultMessage(
  options?: TestDataOptions,
): TestDataResult;
export function generateBasicTextMessage(options?: any): TestDataResult;
export function generateToolCallMessage(options?: any): TestDataResult;
export function generateMultiModalMessage(options?: any): TestDataResult;
export function generateConverseMessage(options?: any): any;
export function generateConverseWithTools(options?: any): any;
export function generateAgentMessage(options?: any): any;
export function generateStreamingVariants(
  baseRequest: any,
  apiType?: string,
): any;
export function generateToolDefinition(
  name: string,
  description: string,
  schema: any,
): any;

export const commonTools: {
  weather: any;
  calculator: any;
  webSearch: any;
};

export const errorScenarios: {
  invalidModel: (baseRequest: any) => any;
  malformedBody: (baseRequest: any) => any;
  missingRegion: (baseRequest: any) => any;
  invalidToolSchema: (baseRequest: any) => any;
};

export function generateTestSuite(): any;

export const defaults: {
  modelId: string;
  region: string;
  maxTokens: number;
  anthropicVersion: string;
};
