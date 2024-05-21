import { type Configuration, registerOTel } from '@vercel/otel'
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base'
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto'
import { OpenAIInstrumentation } from '@arizeai/openinference-instrumentation-openai'
import * as OpenAI from 'openai'
import { diag, DiagConsoleLogger, DiagLogLevel } from '@opentelemetry/api'

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.INFO)

const openAIInstrumentation = new OpenAIInstrumentation({})
openAIInstrumentation.manuallyInstrument(OpenAI)
let config: Configuration = {
  attributes: {
    'openinference.project.name': 'openinference-nextjs-3'
  },
  instrumentations: [openAIInstrumentation],
  instrumentationConfig: { fetch: { enabled: false } },
  attributesFromHeaders: {
    client: 'X-Client'
  }
}

config = {
  ...config,
  spanProcessors: [
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        // This is the url where your phoenix server is running
        url: 'http://localhost:6006/v1/traces'
      })
    ),
    // new SimpleSpanProcessor(new ConsoleSpanExporter()),
    ...(config.spanProcessors ?? [])
  ]
}
registerOTel(config)
