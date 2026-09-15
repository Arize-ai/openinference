// Manual-instrumentation example: build OpenInference LLM spans by
// hand using openinference-semantic-conventions constants. Useful when
// you need fine control over the span shape or want to instrument code
// paths the auto-instrumentor doesn't cover.
//
// For most OpenAI users, ../auto-instrumentation/ is simpler.
//
// Run:
//
//	OPENAI_API_KEY=... [ARIZE_SPACE_ID=... ARIZE_API_KEY=...] \
//	  go run . -backend={ax|phoenix}
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	semconv "github.com/Arize-ai/openinference/go/openinference-semantic-conventions"
)

func main() {
	backend := flag.String("backend", "phoenix", "where to send traces: ax | phoenix")
	flag.Parse()

	ctx := context.Background()
	tp, err := newTracerProvider(ctx, *backend)
	if err != nil {
		log.Printf("tracer setup: %v", err)
		return
	}
	defer func() {
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = tp.Shutdown(shutdownCtx)
	}()

	// Use the local TracerProvider directly instead of otel.Tracer(...);
	// the latter resolves against the global TracerProvider, which we
	// never installed, so spans would otherwise vanish into the no-op
	// provider.
	tracer := tp.Tracer("openai-manual-example")
	client := openai.NewClient(option.WithAPIKey(mustGetenv("OPENAI_API_KEY"))) // no middleware — we span by hand

	userPrompt := "In one sentence, what is observability?"
	ctx, span := tracer.Start(ctx, "openai.chat.completions.create")
	defer span.End()
	span.SetAttributes(
		attribute.String(semconv.OpenInferenceSpanKind, semconv.SpanKindLLM),
		attribute.String(semconv.LLMSystem, semconv.LLMSystemOpenAI),
		attribute.String(semconv.LLMProvider, semconv.LLMProviderOpenAI),
		attribute.String(semconv.LLMModelName, string(shared.ChatModelGPT4o)),
		attribute.String(semconv.LLMInputMessageRoleKey(0), "user"),
		attribute.String(semconv.LLMInputMessageContentKey(0), userPrompt),
		attribute.String(semconv.InputValue, userPrompt),
	)

	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model:    shared.ChatModelGPT4o,
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage(userPrompt)},
	})
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		log.Printf("openai: %v", err)
		return
	}

	out := resp.Choices[0].Message.Content
	span.SetAttributes(
		attribute.String(semconv.LLMOutputMessageRoleKey(0), "assistant"),
		attribute.String(semconv.LLMOutputMessageContentKey(0), out),
		attribute.String(semconv.OutputValue, out),
		attribute.Int64(semconv.LLMTokenCountPrompt, resp.Usage.PromptTokens),
		attribute.Int64(semconv.LLMTokenCountCompletion, resp.Usage.CompletionTokens),
		attribute.Int64(semconv.LLMTokenCountTotal, resp.Usage.TotalTokens),
		attribute.String(semconv.LLMFinishReason, resp.Choices[0].FinishReason),
	)
	log.Println(out)
}

func newTracerProvider(ctx context.Context, backend string) (*sdktrace.TracerProvider, error) {
	var opts []otlptracehttp.Option
	switch backend {
	case "ax":
		opts = []otlptracehttp.Option{
			otlptracehttp.WithEndpoint("otlp.arize.com"),
			otlptracehttp.WithHeaders(map[string]string{
				"space_id": mustGetenv("ARIZE_SPACE_ID"),
				"api_key":  mustGetenv("ARIZE_API_KEY"),
			}),
		}
	case "phoenix":
		opts = []otlptracehttp.Option{
			otlptracehttp.WithEndpoint(getenvOr("PHOENIX_ENDPOINT", "localhost:6006")),
			otlptracehttp.WithInsecure(),
		}
	default:
		log.Fatalf("unknown backend %q (use ax or phoenix)", backend)
	}
	exp, err := otlptracehttp.New(ctx, opts...)
	if err != nil {
		return nil, err
	}
	res, _ := resource.New(ctx, resource.WithAttributes(
		attribute.String("service.name", "openai-manual-example"),
		attribute.String("openinference.project.name", "openai-manual-example"),
	))
	return sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
	), nil
}

func mustGetenv(k string) string {
	v := os.Getenv(k)
	if v == "" {
		log.Fatalf("environment variable %s is required", k)
	}
	return v
}

func getenvOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
