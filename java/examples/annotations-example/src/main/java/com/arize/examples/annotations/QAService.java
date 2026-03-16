package com.arize.examples.annotations;

import com.arize.instrumentation.annotations.*;
import java.util.Map;

public class QAService {

    @TraceAgent(name = "qa-agent")
    public String answer(String question) {
        String context = retrieve(question);
        Map<String, Object> weather = getWeather("San Francisco");
        return generate(question, context, weather);
    }

    @TraceChain(name = "retriever")
    public String retrieve(String query) {
        // Simulated retrieval
        return "OpenInference is an open standard for AI tracing that works with OpenTelemetry.";
    }

    @TraceTool(name = "weather", description = "Gets current weather for a location")
    public Map<String, Object> getWeather(String location) {
        // Simulated tool call
        return Map.of("temp", 68, "condition", "foggy", "location", location);
    }

    @TraceLLM(name = "generator")
    public String generate(String question, String context, @SpanIgnore Map<String, Object> weather) {
        // Simulated LLM call
        return "Based on the context: " + context + " The weather is " + weather.get("temp") + "°F.";
    }
}
