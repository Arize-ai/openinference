# Mission

**Trace the frontier. Inform the future of agent telemetry.**

Agents change faster than standards can. New frameworks, model APIs, and
protocols reach production every month, and the teams running them need to see
what their agents do today. OpenInference makes agents observable now, and
takes what it learns to [OpenTelemetry](https://opentelemetry.io), so the
standard grows from what works in production.

## Trace the frontier

Standards are rightly deliberate. Instrumentation can't wait for them. When a
framework or provider starts to matter, we instrument it, closing the gap
between what agents do and what telemetry can see. Each instrumentor is a
working answer to "what should a trace of this look like?", shipped and in use
today rather than waiting on consensus.

## Inform the future

OpenTelemetry's [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
are written in the open, by consensus, across vendors. That is how a lasting
standard should be made, and OpenInference is built to feed it:

- **On OpenTelemetry, not beside it.** OpenInference spans are OpenTelemetry
  spans, exported over OTLP to any compatible backend.
- **Upstream by design.** We bring what works in OpenInference to the GenAI
  conventions. Arize has
  [granted the OpenInference instrumentation code](https://github.com/open-telemetry/community/issues/3467)
  to OpenTelemetry to accelerate its GenAI instrumentation coverage.
- **Converging as the standard lands.** The Python instrumentation can emit
  the GenAI attributes alongside its own (`enable_genai_semconv`), and
  [`@arizeai/openinference-genai`](https://github.com/Arize-ai/openinference/tree/main/js/packages/openinference-genai)
  converts GenAI spans to OpenInference in JavaScript. Adopting the standard
  never means re-instrumenting.

## Further reading

- [OpenTelemetry semantic conventions for generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [The OpenInference code grant to OpenTelemetry](https://github.com/open-telemetry/community/issues/3467) (open-telemetry/community#3467)
- [Observation Deck, episode 5: "To OTel with Love"](https://www.youtube.com/watch?v=FSrw6Fnmj1Q&t=761s) (Dynatrace)
- [The OpenInference specification](./spec/README.md)
