/**
 * Test Data Generators for AWS Bedrock Instrumentation Tests
 *
 * This module provides utilities for generating varied test inputs to ensure
 * comprehensive coverage of Bedrock API scenarios including tool calls,
 * multi-modal content, and different message structures.
 */

/**
 * Default test configuration
 */
const defaults = {
  modelId: "anthropic.claude-3-sonnet-20240229-v1:0",
  region: "us-east-1",
  maxTokens: 100,
  anthropicVersion: "bedrock-2023-05-31",
};

/**
 * Generates basic text message for InvokeModel API
 */
function generateBasicTextMessage(options: any = {}) {
  const {
    prompt = "Hello, how are you today?",
    modelId = defaults.modelId,
    maxTokens = defaults.maxTokens,
    systemPrompt = null,
  } = options;

  const body: any = {
    anthropic_version: defaults.anthropicVersion,
    max_tokens: maxTokens,
    messages: [
      {
        role: "user",
        content: prompt,
      },
    ],
  };

  if (systemPrompt) {
    body.system = systemPrompt;
  }

  return {
    modelId,
    body: JSON.stringify(body),
  };
}

/**
 * Generates tool definition for function calling
 */
function generateToolDefinition(name: string, description: string, schema: any) {
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
function generateToolCallMessage(options: any = {}) {
  const {
    prompt = "What's the weather like in San Francisco?",
    tools = [commonTools.weather],
    modelId = defaults.modelId,
    maxTokens = defaults.maxTokens,
  } = options;

  const body: any = {
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
function generateToolResultMessage(options: any = {}) {
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

  const body: any = {
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

/**
 * Generates multi-modal message with image
 */
function generateMultiModalMessage(options: any = {}) {
  const {
    textPrompt = "What do you see in this image?",
    imageData = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
    mediaType = "image/png",
    modelId = defaults.modelId,
    maxTokens = defaults.maxTokens,
  } = options;

  const body: any = {
    anthropic_version: defaults.anthropicVersion,
    max_tokens: maxTokens,
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: textPrompt,
          },
          {
            type: "image",
            source: {
              type: "base64",
              media_type: mediaType,
              data: imageData,
            },
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

/**
 * Generates Converse API request (modern API)
 */
function generateConverseMessage(options: any = {}) {
  const {
    messages = [{ role: "user", content: [{ text: "Hello!" }] }],
    system = null,
    toolConfig = null,
    inferenceConfig = { maxTokens: defaults.maxTokens },
    modelId = defaults.modelId,
  } = options;

  const request: any = {
    modelId,
    messages,
    inferenceConfig,
  };

  if (system) {
    request.system = Array.isArray(system) ? system : [{ text: system }];
  }

  if (toolConfig) {
    request.toolConfig = toolConfig;
  }

  return request;
}

/**
 * Generates Converse API request with tools
 */
function generateConverseWithTools(options: any = {}) {
  const {
    prompt = "Calculate 15 * 23",
    tools = [
      {
        toolSpec: {
          name: "calculator",
          description: "Perform mathematical calculations",
          inputSchema: {
            json: {
              type: "object",
              properties: {
                expression: { type: "string" },
              },
              required: ["expression"],
            },
          },
        },
      },
    ],
    modelId = defaults.modelId,
  } = options;

  return generateConverseMessage({
    messages: [{ role: "user", content: [{ text: prompt }] }],
    toolConfig: { tools },
    modelId,
  });
}

/**
 * Generates InvokeAgent request for agent workflows
 */
function generateAgentMessage(options: any = {}) {
  const {
    agentId = "test-agent-123",
    agentAliasId = "test-alias",
    sessionId = "test-session-456",
    inputText = "Help me plan a trip to Japan",
    enableTrace = true,
  } = options;

  return {
    agentId,
    agentAliasId,
    sessionId,
    inputText,
    enableTrace,
  };
}

/**
 * Generates streaming request variants
 */
function generateStreamingVariants(baseRequest: any, apiType: string = "invokeModel") {
  const variants: any = {
    invokeModel: {
      ...baseRequest,
      // InvokeModelWithResponseStream has same structure
    },
    converse: {
      ...baseRequest,
      // ConverseStream has same structure
    },
    agent: {
      ...baseRequest,
      // Agent calls are always streaming
    },
  };

  return variants[apiType] || baseRequest;
}

/**
 * Error scenario generators
 */
const errorScenarios = {
  invalidModel: (baseRequest: any) => ({
    ...baseRequest,
    modelId: "invalid-model-id",
  }),

  malformedBody: (baseRequest: any) => ({
    ...baseRequest,
    body: '{"invalid": json',
  }),

  missingRegion: (baseRequest: any) => ({
    ...baseRequest,
    // Will cause region-related errors
  }),

  invalidToolSchema: (baseRequest: any) => {
    const parsed = JSON.parse(baseRequest.body);
    parsed.tools = [
      {
        name: "invalid_tool",
        // Missing required fields
      },
    ];
    return {
      ...baseRequest,
      body: JSON.stringify(parsed),
    };
  },
};

/**
 * Generates test data for different complexity levels
 */
function generateTestSuite() {
  return {
    basic: {
      simple: generateBasicTextMessage(),
      withSystem: generateBasicTextMessage({
        systemPrompt: "You are a helpful assistant.",
      }),
      longPrompt: generateBasicTextMessage({
        prompt: "Tell me a detailed story about " + "adventure ".repeat(50),
      }),
    },

    tools: {
      singleTool: generateToolCallMessage(),
      multipleTools: generateToolCallMessage({
        tools: [commonTools.weather, commonTools.calculator],
      }),
      toolResult: generateToolResultMessage(),
    },

    multimodal: {
      textAndImage: generateMultiModalMessage(),
      multipleImages: generateMultiModalMessage({
        // Would need to be expanded for multiple images
      }),
    },

    converse: {
      basic: generateConverseMessage(),
      withTools: generateConverseWithTools(),
      withSystem: generateConverseMessage({
        system: "You are a helpful assistant",
        messages: [{ role: "user", content: [{ text: "Hello!" }] }],
      }),
    },

    agent: {
      basic: generateAgentMessage(),
      withTrace: generateAgentMessage({ enableTrace: true }),
    },

    errors: {
      invalidModel: errorScenarios.invalidModel(generateBasicTextMessage()),
      malformedBody: errorScenarios.malformedBody(generateBasicTextMessage()),
      invalidTool: errorScenarios.invalidToolSchema(generateToolCallMessage()),
    },
  };
}

export {
  // Individual generators
  generateBasicTextMessage,
  generateToolCallMessage,
  generateToolResultMessage,
  generateMultiModalMessage,
  generateConverseMessage,
  generateConverseWithTools,
  generateAgentMessage,
  generateStreamingVariants,

  // Tool definitions
  generateToolDefinition,
  commonTools,

  // Error scenarios
  errorScenarios,

  // Test suite
  generateTestSuite,

  // Constants
  defaults,
};
