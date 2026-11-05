// Decisions API example: one POST /v1/decisions call that asks a
// predicate, a choice and a score question about a customer message. The
// openinference-instrumentation-openai-go middleware emits one
// OpenInference DECISION span for the call.
//
// openai-go v1 has no Decisions client, so the request goes through the
// generic client.Post, which runs the same middleware chain. On
// github.com/openai/openai-go/v3 (v3.73.0 or later), client.Decisions.New
// sends the same request and is traced the same way.
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
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	openaiotel "github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go"
)

type decisionRequest struct {
	Model     string     `json:"model"`
	Input     string     `json:"input"`
	Questions []question `json:"questions"`
}

type question struct {
	Type         string   `json:"type"`
	Name         string   `json:"name"`
	Instructions string   `json:"instructions"`
	Choices      []choice `json:"choices,omitempty"`
	Levels       []level  `json:"levels,omitempty"`
}

type choice struct {
	Value       string `json:"value"`
	Description string `json:"description,omitempty"`
}

type level struct {
	Label string `json:"label"`
}

type decision struct {
	Model   string `json:"model"`
	Answers []struct {
		Type        string  `json:"type"`
		Name        string  `json:"name"`
		Probability float64 `json:"probability"`
		Choice      string  `json:"choice"`
		Score       float64 `json:"score"`
	} `json:"answers"`
}

func main() {
	backend := flag.String("backend", "phoenix", "where to send traces: ax | phoenix")
	model := flag.String("model", "gpt-6-luna", "OpenAI decision model to call")
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

	client := openai.NewClient(
		option.WithAPIKey(mustGetenv("OPENAI_API_KEY")),
		option.WithMiddleware(openaiotel.Middleware(tp.Tracer("openai-decisions-example"))),
	)

	var res decision
	err = client.Post(ctx, "decisions", decisionRequest{
		Model: *model,
		Input: "I was charged twice for my order and the box arrived crushed.",
		Questions: []question{
			{Type: "predicate", Name: "is_complaint", Instructions: "Is this message a complaint?"},
			{
				Type:         "choice",
				Name:         "department",
				Instructions: "Which department should handle this message?",
				Choices: []choice{
					{Value: "billing", Description: "Payments, charges and refunds"},
					{Value: "shipping", Description: "Delivery and damaged packages"},
				},
			},
			{
				Type:         "score",
				Name:         "severity",
				Instructions: "How severe is the customer's issue?",
				Levels:       []level{{Label: "low"}, {Label: "medium"}, {Label: "high"}},
			},
		},
	}, &res)
	if err != nil {
		log.Printf("openai: %v", err)
		return
	}
	for _, a := range res.Answers {
		switch a.Type {
		case "predicate":
			log.Printf("%s: probability %.2f", a.Name, a.Probability)
		case "choice":
			log.Printf("%s: %s", a.Name, a.Choice)
		case "score":
			log.Printf("%s: score %.2f", a.Name, a.Score)
		case "refusal":
			log.Printf("%s: refused", a.Name)
		}
	}
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
		attribute.String("service.name", "openai-decisions-example"),
		attribute.String("openinference.project.name", "openai-decisions-example"),
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
