package com.arize.examples.annotation;

import com.arize.instrumentation.annotation.*;
import java.util.Map;

public class QAService {

    @Agent(name = "qa-agent")
    public String answer(String question) {
        String context = retrieve(question);
        Map<String, Object> weather = getWeather("San Francisco");
        return generate(question, context, weather);
    }

    @Chain(name = "retriever")
    public String retrieve(String query) {
        // Simulated retrieval
        return "OpenInference is an open standard for AI tracing that works with OpenTelemetry.";
    }

    @Tool(name = "weather", description = "Gets current weather for a location")
    public Map<String, Object> getWeather(String location) {
        // Simulated tool call
        return Map.of("temp", 68, "condition", "foggy", "location", location);
    }

    @LLM(name = "generator")
    public String generate(String question, String context, @SpanIgnore Map<String, Object> weather) {
        // Simulated LLM call
        return "Based on the context: " + context + " The weather is " + weather.get("temp") + "°F.";
    }
}
