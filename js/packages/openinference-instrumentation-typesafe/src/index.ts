/**
 * OpenInference instrumentation for `@typesafe-ai/sdk`.
 *
 * @packageDocumentation
 *
 * ## Public API
 *
 * - {@link TypeSafeInstrumentation} — register with OpenTelemetry, or call
 *   `manuallyInstrument` for ESM / bundled apps
 *
 * ## What gets traced
 *
 * `TypeSafeClient.systemOne` → one `LLM` span per call (including SDK retries).
 * `client.models.list()` is not instrumented. Spans use structured JSON
 * `input.value` / `output.value`; no chat-message attributes are emitted.
 *
 * @see README.md for install, configuration, and examples
 */
export { TypeSafeInstrumentation } from "./instrumentation";
