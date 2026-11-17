# Mission

We want to trace the frontier of AI, and use what we learn there to help shape where agent telemetry goes next.

Agents move fast. A new framework or model API comes out and people are running it in production a week later. If
you're the one responsible for that agent, you need to see what it's doing now. Waiting a year for a spec to catch up
isn't an option.

## Why OpenInference exists

OpenTelemetry is the right foundation for all of this. We build on it and we want it to succeed. The
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) are being worked out in the open,
across a lot of vendors, and that takes time. It should. That's how you end up with a standard people can rely on.

The instrumentation can't wait for that, though. So when a framework or provider starts to matter, we instrument it,
ship it, and see what holds up once real people use it. Every instrumentor is our best answer to "what should a trace
of this actually look like?", and we find out quickly if we got it wrong.

## How it feeds back into OpenTelemetry

OpenInference isn't meant to be a separate thing off to the side.

- OpenInference spans are OpenTelemetry spans. Send them to any OTLP backend you like.
- We
  [code-granted the OpenInference instrumentation](https://github.com/open-telemetry/community/issues/3467)
  to OpenTelemetry so its GenAI coverage can catch up faster.
- As the GenAI conventions settle, we follow them. In Python, `enable_genai_semconv` emits the GenAI attributes next to
  ours. In JavaScript,
  [`@arizeai/openinference-genai`](https://github.com/Arize-ai/openinference/tree/main/js/packages/openinference-genai)
  maps GenAI spans to OpenInference. You shouldn't have to re-instrument to move to the standard.

OpenInference is where we figure things out quickly. OpenTelemetry is where the things that work end up.

## Further reading

- [OpenTelemetry semantic conventions for generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [The OpenInference code grant to OpenTelemetry](https://github.com/open-telemetry/community/issues/3467)
- [Observation Deck, episode 5: "To OTel with Love"](https://www.youtube.com/watch?v=FSrw6Fnmj1Q&t=761s) (Dynatrace)
- [The OpenInference specification](./spec/README.md)
