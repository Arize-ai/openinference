package com.arize.instrumentation.adk;

import static com.arize.semconv.trace.SemanticConventions.*;

import com.google.adk.agents.BaseAgent;
import com.google.adk.sessions.Session;
import io.opentelemetry.api.trace.Span;
import java.lang.reflect.Method;
import java.util.Map;
import net.bytebuddy.asm.Advice;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import tools.jackson.databind.json.JsonMapper;

/**
 * Advice class for intercepting the traceToolCall method in Telemetry.
 */
public class RunnerAdvice {

    public static final Logger log = LoggerFactory.getLogger(RunnerAdvice.class);

    public static final JsonMapper JSON_MAPPER = new JsonMapper();

    @SuppressWarnings("unchecked")
    @Advice.OnMethodExit()
    public static void onExit(
            @Advice.FieldValue("agent") BaseAgent agent,
            @Advice.Origin Method method,
            @Advice.AllArguments Object[] args) {

        log.info("Enhancing method {} onExit.", method.getName());

        Session session = (Session) args[0];
        Map<String, Object> stateDelta = (Map<String, Object>) args[3];

        Span span = Span.current();

        span.setAttribute(INPUT_VALUE, JSON_MAPPER.writeValueAsString(args));
        span.setAttribute(INPUT_MIME_TYPE, MimeType.JSON.getValue());
        span.setAttribute(USER_ID, session.userId());
        String sessionId = session.id();
        if (sessionId.contains("::")) {
            sessionId = sessionId.substring(0, sessionId.indexOf("::"));
        }
        span.setAttribute(SESSION_ID, sessionId);
        span.setAttribute(METADATA, JSON_MAPPER.writeValueAsString(stateDelta));
        span.setAttribute(OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.CHAIN.getValue());

        // get instance agent property
        span.setAttribute(AGENT_NAME, agent.name());
    }
}
