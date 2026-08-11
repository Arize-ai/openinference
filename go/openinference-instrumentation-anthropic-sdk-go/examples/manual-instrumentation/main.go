// Manual-instrumentation example: build OpenInference LLM spans by
// hand using openinference-semantic-conventions constants. Useful when
// you've wrapped an LLM call that doesn't have a first-party
// instrumentor, or when you want fine control over span shape.
//
// For most Anthropic users, the auto-instrumentation example in
// ../auto-instrumentation/ is simpler and equivalent.
//
// Run with either backend:
//
//	ARIZE_SPACE_ID=... ARIZE_API_KEY=... ANTHROPIC_API_KEY=... \
//	  go run . -backend=ax
//	ANTHROPIC_API_KEY=... go run . -backend=phoenix
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
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

	tracer := tp.Tracer("anthropic-manual-example")
	client := anthropic.NewClient() // no middleware — we span by hand

	userPrompt := "In one sentence, what is observability?"
	ctx, span := tracer.Start(ctx, "anthropic.messages.create")
	defer span.End()
	span.SetAttributes(
		attribute.String(semconv.OpenInferenceSpanKind, semconv.SpanKindLLM),
		attribute.String(semconv.LLMSystem, semconv.LLMSystemAnthropic),
		attribute.String(semconv.LLMProvider, semconv.LLMProviderAnthropic),
		attribute.String(semconv.LLMModelName, "claude-3-5-sonnet-latest"),
		attribute.String(semconv.LLMInputMessageRoleKey(0), "user"),
		attribute.String(semconv.LLMInputMessageContentKey(0), userPrompt),
		attribute.String(semconv.InputValue, userPrompt),
	)

	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 100,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(userPrompt)),
		},
	})
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		log.Printf("anthropic: %v", err)
		return
	}

	var output string
	for _, block := range resp.Content {
		output += block.Text
	}
	span.SetAttributes(
		attribute.String(semconv.LLMOutputMessageRoleKey(0), "assistant"),
		attribute.String(semconv.LLMOutputMessageContentKey(0), output),
		attribute.String(semconv.OutputValue, output),
		attribute.Int64(semconv.LLMTokenCountPrompt, resp.Usage.InputTokens),
		attribute.Int64(semconv.LLMTokenCountCompletion, resp.Usage.OutputTokens),
		attribute.Int64(semconv.LLMTokenCountTotal, resp.Usage.InputTokens+resp.Usage.OutputTokens),
		attribute.String(semconv.LLMFinishReason, string(resp.StopReason)),
	)
	log.Println(output)
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
		attribute.String("service.name", "anthropic-manual-example"),
		attribute.String("openinference.project.name", "anthropic-manual-example"),
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
