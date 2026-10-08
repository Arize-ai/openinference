package com.arize.instrumentation.langchain4j;

import com.github.tomakehurst.wiremock.WireMockServer;
import com.github.tomakehurst.wiremock.core.WireMockConfiguration;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import java.io.File;
import java.net.URL;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;

public class BaseInstrumentationSetup {

    private static final String PROJECT_ROOT = resolveProjectRoot();
    static final String CASSETTES_ROOT = PROJECT_ROOT + "/src/test/resources/cassettes/";

    public InMemorySpanExporter spanExporter;
    public SdkTracerProvider tracerProvider;
    public static WireMockServer wireMock;

    @BeforeAll
    static void startWireMock() {
        wireMock = new WireMockServer(WireMockConfiguration.options().dynamicPort());
        wireMock.start();
    }

    @AfterAll
    static void stopWireMock() {
        wireMock.stop();
    }

    @BeforeEach
    void setUp() {
        wireMock.resetAll();
        spanExporter = InMemorySpanExporter.create();
        tracerProvider = SdkTracerProvider.builder()
                .addSpanProcessor(SimpleSpanProcessor.create(spanExporter))
                .build();
    }

    @AfterEach
    void tearDown() {
        tracerProvider.shutdown();
    }

    private static String resolveProjectRoot() {
        try {
            URL location = BaseInstrumentationSetup.class
                    .getProtectionDomain()
                    .getCodeSource()
                    .getLocation();

            File current = new File(location.toURI());

            // Start from compiled test classes directory
            if (current.isFile()) {
                current = current.getParentFile();
            }

            // Walk up until we find a Gradle root (build.gradle or settings.gradle)
            while (current != null) {
                File buildGradle = new File(current, "build.gradle");
                File settingsGradle = new File(current, "settings.gradle");
                File settingsGradleKts = new File(current, "settings.gradle.kts");

                if (buildGradle.exists() || settingsGradle.exists() || settingsGradleKts.exists()) {
                    return current.getAbsolutePath();
                }

                current = current.getParentFile();
            }

            throw new RuntimeException("Could not locate project root (no build.gradle or settings.gradle found)");

        } catch (Exception e) {
            throw new RuntimeException("Cannot resolve project root", e);
        }
    }
}
