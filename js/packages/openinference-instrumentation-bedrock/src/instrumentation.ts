import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import { diag, SpanKind, SpanStatusCode } from "@opentelemetry/api";
import { OITracer, TraceConfigOptions } from "@arizeai/openinference-core";
import { VERSION } from "./version";
import { 
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand 
} from "@aws-sdk/client-bedrock-runtime";
import { extractInvokeModelRequestAttributes } from "./attributes/request-attributes";
import { extractInvokeModelResponseAttributes } from "./attributes/response-attributes";

const MODULE_NAME = "@aws-sdk/client-bedrock-runtime";

/**
 * Track if the Bedrock instrumentation is patched
 */
let _isBedrockPatched = false;

/**
 * Check if Bedrock instrumentation is enabled/disabled
 */
export function isPatched() {
  return _isBedrockPatched;
}

export interface BedrockInstrumentationConfig extends InstrumentationConfig {
  traceConfig?: TraceConfigOptions;
}

export class BedrockInstrumentation extends InstrumentationBase<BedrockInstrumentationConfig> {
  static readonly COMPONENT = "@arizeai/openinference-instrumentation-bedrock";
  static readonly VERSION = VERSION;

  private oiTracer: OITracer;

  constructor(config: BedrockInstrumentationConfig = {}) {
    super(
      BedrockInstrumentation.COMPONENT,
      BedrockInstrumentation.VERSION,
      config,
    );

    this.oiTracer = new OITracer({
      tracer: this.tracer,
      traceConfig: config.traceConfig,
    });
  }

  protected init(): InstrumentationModuleDefinition<unknown>[] {
    const module = new InstrumentationNodeModuleDefinition<unknown>(
      MODULE_NAME,
      ["^3.0.0"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return [module];
  }

  private patch(moduleExports: any, moduleVersion?: string) {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);

    if (moduleExports?.BedrockRuntimeClient) {
      // eslint-disable-next-line @typescript-eslint/no-this-alias
      const instrumentation = this;

      // Wrap the client's send method to intercept commands
      this._wrap(
        moduleExports.BedrockRuntimeClient.prototype,
        "send",
        (original: any) => {
          return function patchedSend(this: unknown, command: any) {
            if (command?.constructor?.name === "InvokeModelCommand") {
              return instrumentation._handleInvokeModelCommand(
                command,
                original,
                this,
              );
            }

            if (command?.constructor?.name === "InvokeModelWithResponseStreamCommand") {
              return instrumentation._handleInvokeModelWithResponseStreamCommand(
                command,
                original,
                this,
              );
            }

            // Pass through other commands without instrumentation
            return original.apply(this, [command]);
          };
        },
      );

      _isBedrockPatched = true;
    }

    return moduleExports;
  }

  private _handleInvokeModelCommand(
    command: InvokeModelCommand,
    original: any,
    client: any,
  ) {
    const requestAttributes = extractInvokeModelRequestAttributes(command);

    const span = this.oiTracer.startSpan("bedrock.invoke_model", {
      kind: SpanKind.CLIENT,
      attributes: requestAttributes,
    });

    try {
      const result = original.apply(client, [command]);

      // AWS SDK v3 send() method always returns a Promise
      return result
        .then((response: any) => {
          extractInvokeModelResponseAttributes(span, response);
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
          return response;
        })
        .catch((error: any) => {
          span.recordException(error);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: error.message,
          });
          span.end();
          throw error;
        });
    } catch (error: any) {
      // Handle errors that occur before the Promise is returned (e.g. invalid parameters)
      span.recordException(error);
      span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      span.end();
      throw error;
    }
  }

  private _handleInvokeModelWithResponseStreamCommand(
    command: InvokeModelWithResponseStreamCommand,
    original: any,
    client: any,
  ) {
    const requestAttributes = extractInvokeModelRequestAttributes(command as any);

    const span = this.oiTracer.startSpan("bedrock.invoke_model", {
      kind: SpanKind.CLIENT,
      attributes: requestAttributes,
    });

    try {
      const result = original.apply(client, [command]);

      // AWS SDK v3 send() method always returns a Promise
      return result
        .then(async (response: any) => {
          // For streaming responses, we need to process the stream to extract response attributes
          try {
            // Process the streaming response
            let accumulatedResponse = {
              id: "",
              content: [] as any[],
              usage: {} as any,
            };

            if (response.body) {
              for await (const chunk of response.body) {
                if (chunk.chunk?.bytes) {
                  const text = new TextDecoder().decode(chunk.chunk.bytes);
                  const lines = text.split('\n').filter(line => line.trim());
                  
                  for (const line of lines) {
                    // Try to parse the line as JSON directly (Bedrock streaming format)
                    if (line.trim()) {
                      try {
                        const data = JSON.parse(line);
                        
                        // Handle different event types
                        if (data.type === 'message_start' && data.message) {
                          accumulatedResponse.id = data.message.id;
                          accumulatedResponse.usage = data.message.usage || {};
                        }
                        
                        if (data.type === 'content_block_start' && data.content_block) {
                          accumulatedResponse.content.push(data.content_block);
                        }
                        
                        if (data.type === 'content_block_delta' && data.delta?.text) {
                          // Find the last text content block and append the delta
                          const lastTextBlock = accumulatedResponse.content.find(block => block.type === 'text');
                          if (lastTextBlock) {
                            lastTextBlock.text = (lastTextBlock.text || '') + data.delta.text;
                          } else {
                            // Create a new text block if none exists
                            accumulatedResponse.content.push({
                              type: 'text',
                              text: data.delta.text
                            });
                          }
                        }
                        
                        if (data.type === 'message_delta' && data.usage) {
                          accumulatedResponse.usage = { ...accumulatedResponse.usage, ...data.usage };
                        }
                      } catch (e) {
                        // Skip malformed JSON
                      }
                    }
                  }
                }
              }
            }

            // Extract response attributes from the accumulated response
            // Convert accumulated response to the format expected by extractInvokeModelResponseAttributes
            const mockResponse = {
              body: new TextEncoder().encode(JSON.stringify(accumulatedResponse)),
              contentType: "application/json",
              $metadata: {}
            };
            extractInvokeModelResponseAttributes(span, mockResponse as any);
            span.setStatus({ code: SpanStatusCode.OK });
          } catch (streamError: any) {
            span.recordException(streamError);
            span.setStatus({
              code: SpanStatusCode.ERROR,
              message: streamError.message,
            });
          }
          
          span.end();
          return response;
        })
        .catch((error: any) => {
          span.recordException(error);
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: error.message,
          });
          span.end();
          throw error;
        });
    } catch (error: any) {
      // Handle errors that occur before the Promise is returned (e.g. invalid parameters)
      span.recordException(error);
      span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      span.end();
      throw error;
    }
  }

  private unpatch(moduleExports: any, moduleVersion?: string) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);

    if (moduleExports?.BedrockRuntimeClient) {
      this._unwrap(moduleExports.BedrockRuntimeClient.prototype, "send");
      _isBedrockPatched = false;
    }
  }
}
