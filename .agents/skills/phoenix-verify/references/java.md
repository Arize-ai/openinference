# Phoenix Verify: Java

Read this file only when the instrumentor under test is Java. `<pkg>` is the short name
(`langchain4j`, `springAI`, `annotation`).

## Where examples live

`java/examples/<name>-example/` (`annotation`, `programmatic`, `langchain4j`, `spring-ai`), each
a Gradle `application` module in the same composite build as the instrumentors, so they always
compile against the working tree. The existing examples read `PROJECT_NAME` from the
environment, export over OTLP gRPC to `http://localhost:4317`, and call `forceFlush()` then
`shutdown()` before exit; `ProgrammaticDemo.java` is the template to copy.

## Setup, proof, run

The build declares a Java 17 toolchain, so `JAVA_HOME` must point at a JDK 17 or newer; on a
JDK 8 shell every Gradle task fails while configuring the Spring AI example.

```bash
cd java && ./gradlew :examples:<name>-example:dependencies --configuration runtimeClasspath | grep 'project :'
# proof: the instrumentor appears as `project :instrumentation:openinference-instrumentation-<pkg>`
PROJECT_NAME=<pkg>-<scenario> ./gradlew :examples:<name>-example:run
```

For before/after, run the same example twice with `PROJECT_NAME=<pkg>-<scenario>-before` and
`-after`; no scratchpad copy is needed because the project name already comes from the
environment.

## Context attributes and suppression

| Context attributes | Suppress tracing |
| --- | --- |
| `ContextAttributes` in `com.arize.instrumentation` | `try (Scope s = SuppressTracing.begin()) { ... }` |

Make one traced call and one suppressed call in the same run, then assert the span count is 1
and the traced span carries `session.id`, `user.id`, `metadata.<key>`, and `tag.tags`.

## Getting a "before" build

| Situation | "Before" |
| --- | --- |
| Fix not yet applied | Run, apply the change, run again |
| Fix already in the working tree or branch | `git worktree add "$SCRATCH/wt-main" origin/main` and run the example from that worktree's `java/` |
| Parity with the last Maven Central release | In the example's `build.gradle`, swap `project(':instrumentation:...')` for the published coordinate at the released version, run, then revert |

## Wiring a new example into the repo

- Register the module in `java/settings.gradle` and add a `## Examples` note to the instrumentor
  README.
- New dependencies go in the example module's `build.gradle`.
- Lint as CI does with `./gradlew spotlessCheck` (`spotlessApply` to fix).
