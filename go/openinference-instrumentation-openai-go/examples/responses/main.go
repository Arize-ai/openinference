// Responses API example: a tool loop through client.Responses.New. The
// openinference-instrumentation-openai-go middleware emits one
// OpenInference LLM span per /v1/responses call: the first returns a
// function_call, the second sends the function_call_output back with
// previous_response_id and gets the final answer.
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
	"encoding/json"
	"flag"
	"log"
	"os"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/responses"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	openaiotel "github.com/Arize-ai/openinference/go/openinference-instrumentation-openai-go"
)

func main() {
	backend := flag.String("backend", "phoenix", "where to send traces: ax | phoenix")
	model := flag.String("model", "gpt-6.1-sol", "OpenAI model to call")
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
		option.WithMiddleware(openaiotel.Middleware(tp.Tracer("openai-responses-example"))),
	)

	tools := []responses.ToolUnionParam{
		responses.ToolParamOfFunction("get_weather", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required":             []string{"city"},
			"additionalProperties": false,
		}, true),
	}

	// First call: the model asks for the get_weather tool.
	first, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model:        *model,
		Instructions: openai.String("Use the get_weather tool to answer weather questions."),
		Input:        responses.ResponseNewParamsInputUnion{OfString: openai.String("What's the weather in Paris?")},
		Tools:        tools,
	})
	if err != nil {
		log.Printf("openai: %v", err)
		return
	}

	// Run each requested tool call and collect its output.
	var outputs responses.ResponseInputParam
	for _, item := range first.Output {
		if item.Type != "function_call" {
			continue
		}
		call := item.AsFunctionCall()
		var args struct {
			City string `json:"city"`
		}
		_ = json.Unmarshal([]byte(call.Arguments), &args)
		result, _ := json.Marshal(getWeather(args.City))
		outputs = append(outputs, responses.ResponseInputItemParamOfFunctionCallOutput(call.CallID, string(result)))
	}
	if len(outputs) == 0 {
		log.Println(first.OutputText())
		return
	}

	// Second call: send the tool output back and get the final answer.
	final, err := client.Responses.New(ctx, responses.ResponseNewParams{
		Model:              *model,
		PreviousResponseID: openai.String(first.ID),
		Input:              responses.ResponseNewParamsInputUnion{OfInputItemList: outputs},
		Tools:              tools,
	})
	if err != nil {
		log.Printf("openai: %v", err)
		return
	}
	log.Println(final.OutputText())
}

func getWeather(city string) map[string]any {
	return map[string]any{"city": city, "temperature_c": 18, "conditions": "partly cloudy"}
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
		attribute.String("service.name", "openai-responses-example"),
		attribute.String("openinference.project.name", "openai-responses-example"),
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
