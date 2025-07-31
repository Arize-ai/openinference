/**
 * Test Data Generators for AWS Bedrock Instrumentation Tests
 *
 * This module provides utilities for generating varied test inputs to ensure
 * comprehensive coverage of Bedrock API scenarios including tool calls.
 */

import { ToolDefinition } from "../../src/types/bedrock-types";

// Type definitions for the functions we actually use
interface ToolCallMessageOptions {
  prompt?: string;
  tools?: ToolDefinition[];
  modelId?: string;
  maxTokens?: number;
}

interface ToolResultMessageOptions {
  initialPrompt?: string;
  toolUseId?: string;
  toolName?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  toolInput?: Record<string, any>;
  toolResult?: string;
  followupPrompt?: string;
  tools?: ToolDefinition[];
  modelId?: string;
  maxTokens?: number;
}

interface TestMessageResult {
  modelId: string;
  body: string;
}

/**
 * Default test configuration
 */
const defaults = {
  modelId: "anthropic.claude-3-sonnet-20240229-v1:0",
  maxTokens: 100,
  anthropicVersion: "bedrock-2023-05-31",
};

/**
 * Generates tool definition for function calling
 */
function generateToolDefinition(
  name: string,
  description: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  schema: { properties: Record<string, any>; required?: string[] },
): ToolDefinition {
  return {
    name,
    description,
    input_schema: {
      type: "object",
      properties: schema.properties || {},
      required: schema.required || [],
    },
  };
}

/**
 * Common tool definitions for testing
 */
const commonTools = {
  weather: generateToolDefinition(
    "get_weather",
    "Get current weather for a location",
    {
      properties: {
        location: {
          type: "string",
          description: "The city and state, e.g. San Francisco, CA",
        },
        unit: {
          type: "string",
          enum: ["celsius", "fahrenheit"],
          description: "Temperature unit",
        },
      },
      required: ["location"],
    },
  ),

  calculator: generateToolDefinition(
    "calculate",
    "Perform mathematical calculations",
    {
      properties: {
        expression: {
          type: "string",
          description: "Mathematical expression to evaluate",
        },
      },
      required: ["expression"],
    },
  ),

  webSearch: generateToolDefinition(
    "web_search",
    "Search the web for information",
    {
      properties: {
        query: {
          type: "string",
          description: "Search query",
        },
        num_results: {
          type: "integer",
          description: "Number of results to return",
          minimum: 1,
          maximum: 10,
        },
      },
      required: ["query"],
    },
  ),
};

/**
 * Generates InvokeModel request with tool definitions
 */
function generateToolCallMessage(
  options: ToolCallMessageOptions = {},
): TestMessageResult {
  const {
    prompt = "What's the weather like in San Francisco?",
    tools = [commonTools.weather],
    modelId = defaults.modelId,
    maxTokens = defaults.maxTokens,
  } = options;

  const body: Record<string, unknown> = {
    anthropic_version: defaults.anthropicVersion,
    max_tokens: maxTokens,
    tools,
    messages: [
      {
        role: "user",
        content: prompt,
      },
    ],
  };

  return {
    modelId,
    body: JSON.stringify(body),
  };
}

/**
 * Generates message with tool result
 */
function generateToolResultMessage(
  options: ToolResultMessageOptions = {},
): TestMessageResult {
  const {
    initialPrompt = "What's the weather in Paris?",
    toolUseId = "toolu_123",
    toolName = "get_weather",
    toolInput = { location: "Paris, France" },
    toolResult = "The weather in Paris is currently 22°C and sunny.",
    followupPrompt = "Great! What should I wear?",
    tools = [commonTools.weather],
    modelId = defaults.modelId,
    maxTokens = defaults.maxTokens,
  } = options;

  const body: Record<string, unknown> = {
    anthropic_version: defaults.anthropicVersion,
    max_tokens: maxTokens,
    tools,
    messages: [
      {
        role: "user",
        content: initialPrompt,
      },
      {
        role: "assistant",
        content: [
          {
            type: "tool_use",
            id: toolUseId,
            name: toolName,
            input: toolInput,
          },
        ],
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: toolUseId,
            content: toolResult,
          },
          {
            type: "text",
            text: followupPrompt,
          },
        ],
      },
    ],
  };

  return {
    modelId,
    body: JSON.stringify(body),
  };
}

export {
  // Used functions only
  generateToolCallMessage,
  generateToolResultMessage,
  commonTools,
};
