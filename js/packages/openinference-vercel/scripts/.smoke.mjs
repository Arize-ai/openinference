import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";

const exporter = new InMemorySpanExporter();
const provider = new BasicTracerProvider();
provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
const tracer = provider.getTracer("smoke");
const span = tracer.startSpan("hello");
span.setAttribute("foo", "bar");
span.end();
await new Promise((r) => setTimeout(r, 200));
const spans = exporter.getFinishedSpans();
console.log("SMOKE_OK spans=" + spans.length + " name=" + (spans[0] && spans[0].name));
await provider.shutdown();
