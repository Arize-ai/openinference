// Auto-instrumentation example: every call to client.Messages.New
// emits an OpenInference LLM span automatically via the SDK middleware.
//
// Run with either backend:
//
//	# Arize AX
//	ARIZE_SPACE_ID=... ARIZE_API_KEY=... ANTHROPIC_API_KEY=... \
//	  go run . -backend=ax
//
//	# Self-hosted Phoenix (default localhost:6006)
//	ANTHROPIC_API_KEY=... go run . -backend=phoenix
//
// See ../manual-instrumentation/ for the manual-span pattern using
// semconv constants directly.
package main

import (
	"context"
	"crypto/tls"
	"flag"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	anthropicotel "github.com/Arize-ai/openinference/go/openinference-instrumentation-anthropic-sdk-go"
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

	// One line wires up the auto-instrumentor: every /v1/messages call
	// emits an LLM-kind span with the OpenInference attributes.
	client := anthropic.NewClient(
		option.WithMiddleware(anthropicotel.Middleware(tp.Tracer("anthropic-auto-example"))),
	)

	resp, err := client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     "claude-3-5-sonnet-latest",
		MaxTokens: 100,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("In one sentence, what is observability?")),
		},
	})
	if err != nil {
		log.Printf("anthropic: %v", err)
		return
	}
	for _, block := range resp.Content {
		log.Println(block.Text)
	}
}

// newTracerProvider builds the OTel TracerProvider for the requested
// backend. Arize AX uses an OTLP/HTTP exporter pointed at otlp.arize.com
// with space_id + api_key headers; Phoenix uses the same exporter but
// against the local Phoenix collector.
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
	// Avoid implicit system-cert paths in CI when running with -insecure.
	_ = tls.Config{}

	exp, err := otlptracehttp.New(ctx, opts...)
	if err != nil {
		return nil, err
	}
	res, _ := resource.New(ctx, resource.WithAttributes(
		attribute.String("service.name", "anthropic-auto-example"),
		attribute.String("openinference.project.name", "anthropic-auto-example"),
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

var _ = http.DefaultTransport
