import { type Configuration, registerOTel } from '@vercel/otel'
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base'
import { OTLPTraceExporter as ProtoOTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto'

let config: Configuration = {
  attributes: {
    'openinference.project.name': 'openinference-nextjs'
  },
  instrumentationConfig: { fetch: { enabled: false } },
  attributesFromHeaders: {
    client: 'X-Client'
  }
}

if (process.env.TEST_FETCH_RESOURCE_NAME_TEMPLATE) {
  config = {
    ...config,
    instrumentationConfig: {
      ...config.instrumentationConfig
    }
  }
}
config = {
  ...config,
  spanProcessors: [
    new SimpleSpanProcessor(
      new ProtoOTLPTraceExporter({
        // This is the url where your phoenix server is running
        url: 'http://localhost:6006/v1/traces'
      })
    ),
    // new SimpleSpanProcessor(new ConsoleSpanExporter()),
    ...(config.spanProcessors ?? [])
  ]
}
registerOTel(config)
