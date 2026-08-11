// Auto-instrumentation example: every call to
// client.Chat.Completions.New emits an OpenInference LLM span
// automatically via the openinference-instrumentation-openai-go
// middleware.
//
// Run with either backend:
//
//	# Arize AX
//	ARIZE_SPACE_ID=... ARIZE_API_KEY=... OPENAI_API_KEY=... \
//	  go run . -backend=ax
//
//	# Self-hosted Phoenix (default localhost:6006)
//	OPENAI_API_KEY=... go run . -backend=phoenix
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
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	openaiotel "github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go"
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

	// Plug the OpenInference middleware into the SDK so every
	// /v1/chat/completions request emits an LLM-kind span. We pass
	// the locally-built TracerProvider's tracer directly rather than
	// calling otel.Tracer(...) — that helper resolves against the
	// global TracerProvider, which we never installed, so spans
	// would otherwise vanish into the no-op provider.
	client := openai.NewClient(
		option.WithAPIKey(mustGetenv("OPENAI_API_KEY")),
		option.WithMiddleware(openaiotel.Middleware(tp.Tracer("openai-auto-example"))),
	)

	resp, err := client.Chat.Completions.New(ctx, openai.ChatCompletionNewParams{
		Model: shared.ChatModelGPT4o,
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("In one sentence, what is observability?"),
		},
	})
	if err != nil {
		log.Printf("openai: %v", err)
		return
	}
	log.Println(resp.Choices[0].Message.Content)
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
		attribute.String("service.name", "openai-auto-example"),
		attribute.String("openinference.project.name", "openai-auto-example"),
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
