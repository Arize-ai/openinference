# Annotation Example

This module demonstrates annotation-based tracing with the OpenInference ByteBuddy agent.

```bash
# 1. Start Phoenix locally
export PHOENIX_API_KEY=your-key    # optional
export PROJECT_NAME=annotation-example
export ARIZE_API_KEY=              # optional cloud destination

# 2a. Run the example (runtime attach)
cd java
./gradlew :examples:annotation-example:run

# 2b. Run with -javaagent instead
./gradlew :instrumentation:openinference-instrumentation-annotation:jar
java -javaagent:./instrumentation/openinference-instrumentation-annotation/build/libs/openinference-instrumentation-annotation.jar \
     -cp ./examples/annotation-example/build/libs/annotation-example.jar com.arize.examples.annotation.QAApplication
```

The example installs the agent at runtime via `OpenInferenceAgentInstaller`, registers an
`OITracer`, and emits spans for the annotated `QAService`. View the resulting trace tree at
`http://localhost:6006` (Phoenix) or in your Arize workspace if the relevant environment
variables are set. When running with `-javaagent` the ByteBuddy bootstrap happens automatically,
but your application still needs to call `OpenInferenceAgent.register(...)` so the agent receives
an `OITracer`.
