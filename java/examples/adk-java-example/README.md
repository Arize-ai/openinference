# ADK Java Example

This module runs a Google ADK for Java `LlmAgent` with one function tool under the
OpenInference ADK Java agent and exports the trace to Phoenix.

```bash
# 1. Start Phoenix locally
docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest

# 2. Provide a Gemini key and (optionally) a project name
export GOOGLE_API_KEY=your-key       # GEMINI_API_KEY also works
export PROJECT_NAME=adk-java-example # default

# 3. Run the example (builds the agent jar and passes it as -javaagent)
cd java
./gradlew :examples:adk-java-example:run
```

ADK creates its own OpenTelemetry spans (`invocation`, `agent_run [...]`, `call_llm`,
`tool_call [...]`) through `GlobalOpenTelemetry`. The `-javaagent` adds the OpenInference span
kinds, messages, tool arguments, token counts, session and user ids to those spans, so the
application only registers a global OpenTelemetry SDK before its first ADK call
(see `WeatherToolExample.initializeOpenTelemetry`). View the trace at `http://localhost:6006`
in the project named by `PROJECT_NAME`.

To run outside Gradle, build the fat jar and pass it to the JVM:

```bash
./gradlew :instrumentation:openinference-instrumentation-adk-java:shadowJar
java -javaagent:./instrumentation/openinference-instrumentation-adk-java/build/libs/openinference-instrumentation-adk-java-<version>-all.jar \
     -cp <your-classpath> com.arize.examples.adk.WeatherToolExample
```
