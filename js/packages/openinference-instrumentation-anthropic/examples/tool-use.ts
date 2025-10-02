/* eslint-disable no-console */
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { registerInstrumentations } from '@opentelemetry/instrumentation';
import { AnthropicInstrumentation } from '@arizeai/openinference-instrumentation-anthropic';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import Anthropic from '@anthropic-ai/sdk';

// Configure the tracer provider
const provider = new NodeTracerProvider();
provider.addSpanProcessor(
  new SimpleSpanProcessor(
    new OTLPTraceExporter({
      url: 'http://localhost:6006/v1/traces',
    })
  )
);
provider.register();

// Register the Anthropic instrumentation
registerInstrumentations({
  instrumentations: [
    new AnthropicInstrumentation(),
  ],
});

async function toolUseExample() {
  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  // Define a simple tool
  const tools = [
    {
      name: 'get_weather',
      description: 'Get the current weather in a given location',
      input_schema: {
        type: 'object' as const,
        properties: {
          location: {
            type: 'string' as const,
            description: 'The city and state, e.g. San Francisco, CA',
          },
        },
        required: ['location'],
      },
    },
  ];

  const message = await anthropic.messages.create({
    model: 'claude-3-sonnet-20240229',
    max_tokens: 1000,
    tools,
    messages: [
      {
        role: 'user',
        content: 'What is the weather like in San Francisco?',
      },
    ],
  });

  console.log('Response:', message.content);

  // Handle tool use in the response
  for (const content of message.content) {
    if (content.type === 'tool_use') {
      console.log('Tool called:', content.name);
      console.log('Tool input:', content.input);
      
      // In a real application, you would call the actual tool here
      // and then send the result back to Claude
      const toolResult = {
        role: 'user' as const,
        content: [
          {
            type: 'tool_result' as const,
            tool_use_id: content.id,
            content: 'The weather in San Francisco is sunny, 72°F',
          },
        ],
      };

      // Continue the conversation with the tool result
      const followUp = await anthropic.messages.create({
        model: 'claude-3-sonnet-20240229',
        max_tokens: 1000,
        tools,
        messages: [
          {
            role: 'user',
            content: 'What is the weather like in San Francisco?',
          },
          message,
          toolResult,
        ],
      });

      console.log('Follow-up response:', followUp.content);
    }
  }
}

toolUseExample().catch(console.error);
