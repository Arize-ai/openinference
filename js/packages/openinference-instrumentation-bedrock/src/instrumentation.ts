import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
} from "@opentelemetry/instrumentation";
import {
  diag,
  context,
  trace, 
  SpanKind,
  SpanStatusCode,
  Span,
} from "@opentelemetry/api";
import {
  SemanticConventions,
  OpenInferenceSpanKind,
  MimeType,
  LLMSystem,
} from "@arizeai/openinference-semantic-conventions";
import {
  OITracer,
  TraceConfigOptions,
} from "@arizeai/openinference-core";
import { TextDecoder } from "util";
import { VERSION } from "./version";

const MODULE_NAME = "@aws-sdk/client-bedrock-runtime";

export interface BedrockInstrumentationConfig extends InstrumentationConfig {
  traceConfig?: TraceConfigOptions;
}

export class BedrockInstrumentation extends InstrumentationBase<BedrockInstrumentationConfig> {
  static readonly COMPONENT = "@arizeai/openinference-instrumentation-bedrock";
  static readonly VERSION = VERSION;

  private oiTracer: OITracer;

  constructor(config: BedrockInstrumentationConfig = {}) {
    super(BedrockInstrumentation.COMPONENT, BedrockInstrumentation.VERSION, config);
    
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
      
      // Patch the send method of BedrockRuntimeClient
      this._wrap(
        moduleExports.BedrockRuntimeClient.prototype,
        "send",
        (original: any) => {
          return function patchedSend(this: unknown, command: any) {
            // Check if this is an InvokeModelCommand
            if (command?.constructor?.name === 'InvokeModelCommand') {
              return instrumentation._handleInvokeModelCommand(command, original, this);
            }
            
            // For other commands, pass through without instrumentation for now
            return original.apply(this, [command]);
          };
        }
      );
    }
    
    return moduleExports;
  }

  private _handleInvokeModelCommand(command: any, original: any, client: any) {
    // Extract request attributes
    const modelId = command.input?.modelId || 'unknown';
    const requestBody = command.input?.body ? JSON.parse(command.input.body) : {};
    
    // Start span with basic attributes
    const spanAttributes: Record<string, any> = {
      [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      [SemanticConventions.LLM_SYSTEM]: 'bedrock',
      [SemanticConventions.LLM_MODEL_NAME]: modelId,
      [SemanticConventions.INPUT_VALUE]: command.input?.body || '',
      [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
    };

    // Extract input messages for Anthropic Claude format
    if (requestBody.messages && Array.isArray(requestBody.messages)) {
      requestBody.messages.forEach((message: any, index: number) => {
        if (message.role && message.content) {
          spanAttributes[`${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_ROLE}`] = message.role;
          spanAttributes[`${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_CONTENT}`] = message.content;
        }
      });
    }

    const span = this.oiTracer.startSpan('bedrock.invoke_model', {
      kind: SpanKind.CLIENT,
      attributes: spanAttributes,
    });

    // Execute the original command
    try {
      const result = original.apply(client, [command]);
      
      // Handle the result (which should be a Promise)
      if (result && typeof result.then === 'function') {
        return result.then((response: any) => {
          this._processInvokeModelResponse(span, response);
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
          return response;
        }).catch((error: any) => {
          span.recordException(error);
          span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
          span.end();
          throw error;
        });
      } else {
        // Synchronous result (unlikely for AWS SDK)
        this._processInvokeModelResponse(span, result);
        span.setStatus({ code: SpanStatusCode.OK });
        span.end();
        return result;
      }
    } catch (error: any) {
      span.recordException(error);
      span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
      span.end();
      throw error;
    }
  }

  private _processInvokeModelResponse(span: Span, response: any) {
    try {
      // Parse response body
      if (response.body) {
        const responseText = new TextDecoder().decode(response.body);
        const responseBody = JSON.parse(responseText);
        
        // Set basic response attributes
        span.setAttributes({
          [SemanticConventions.OUTPUT_VALUE]: responseText,
          [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
        });

        // Extract input/output messages for Anthropic Claude responses
        if (responseBody.content && Array.isArray(responseBody.content)) {
          responseBody.content.forEach((content: any, index: number) => {
            if (content.type === 'text') {
              span.setAttributes({
                [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_ROLE}`]: 'assistant',
                [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.${index}.${SemanticConventions.MESSAGE_CONTENT}`]: content.text,
              });
            }
          });
        }

        // Extract token usage
        if (responseBody.usage) {
          if (responseBody.usage.input_tokens) {
            span.setAttributes({
              [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: responseBody.usage.input_tokens,
            });
          }
          if (responseBody.usage.output_tokens) {
            span.setAttributes({
              [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: responseBody.usage.output_tokens,
            });
          }
          if (responseBody.usage.input_tokens && responseBody.usage.output_tokens) {
            span.setAttributes({
              [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: responseBody.usage.input_tokens + responseBody.usage.output_tokens,
            });
          }
        }
      }
    } catch (error) {
      diag.warn('Failed to process InvokeModel response:', error);
    }
  }

  private unpatch(moduleExports: any, moduleVersion?: string) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);
    
    if (moduleExports?.BedrockRuntimeClient) {
      this._unwrap(moduleExports.BedrockRuntimeClient.prototype, "send");
    }
  }
}
