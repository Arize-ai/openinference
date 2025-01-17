import json

import autogen
from autogen import ConversableAgent
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Context, Status, StatusCode


class AutogenInstrumentor:
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self._original_generate = None
        self._original_initiate_chat = None

    def _safe_json_dumps(self, obj):
        try:
            return json.dumps(obj)
        except (TypeError, ValueError):
            return json.dumps(str(obj))

    def instrument(self):
        self._original_generate = ConversableAgent.generate_reply
        self._original_initiate_chat = ConversableAgent.initiate_chat
        instrumentor = self

        def wrapped_generate(self, messages=None, sender=None, **kwargs):
            try:
                current_context = trace.get_current_span().get_span_context()

                with instrumentor.tracer.start_as_current_span(
                    self.__class__.__name__,
                    context=trace.set_span_in_context(trace.get_current_span()),
                    links=[trace.Link(current_context)],
                ) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "AGENT")
                    span.set_attribute(
                        SpanAttributes.INPUT_VALUE, instrumentor._safe_json_dumps(messages)
                    )
                    span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")
                    span.set_attribute("agent.type", self.__class__.__name__)

                    response = instrumentor._original_generate(
                        self, messages=messages, sender=sender, **kwargs
                    )

                    span.set_attribute(
                        SpanAttributes.OUTPUT_VALUE, instrumentor._safe_json_dumps(response)
                    )
                    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")

                    return response

            except Exception as e:
                if "span" in locals():
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                raise

        def wrapped_initiate_chat(self, recipient, *args, **kwargs):
            try:
                message = kwargs.get("message", args[0] if args else None)
                current_context = trace.get_current_span().get_span_context()

                with instrumentor.tracer.start_as_current_span(
                    "Autogen",
                    context=trace.set_span_in_context(trace.get_current_span()),
                    links=[trace.Link(current_context)],
                ) as span:
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "AGENT")
                    span.set_attribute(
                        SpanAttributes.INPUT_VALUE, instrumentor._safe_json_dumps(message)
                    )
                    span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")

                    result = instrumentor._original_initiate_chat(self, recipient, *args, **kwargs)

                    if hasattr(result, "chat_history") and result.chat_history:
                        last_message = result.chat_history[-1]["content"]
                        span.set_attribute(
                            SpanAttributes.OUTPUT_VALUE, instrumentor._safe_json_dumps(last_message)
                        )
                    else:
                        span.set_attribute(
                            SpanAttributes.OUTPUT_VALUE, instrumentor._safe_json_dumps(result)
                        )

                    span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, "application/json")

                    return result

            except Exception as e:
                if "span" in locals():
                    span.set_status(Status(StatusCode.ERROR))
                    span.record_exception(e)
                raise

        # Replace methods
        ConversableAgent.generate_reply = wrapped_generate
        ConversableAgent.initiate_chat = wrapped_initiate_chat

        return self

    def uninstrument(self):
        """Restore original behavior"""
        if self._original_generate and self._original_initiate_chat:
            ConversableAgent.generate_reply = self._original_generate
            ConversableAgent.initiate_chat = self._original_initiate_chat
            self._original_generate = None
            self._original_initiate_chat = None
        return self


class SpanAttributes:
    OPENINFERENCE_SPAN_KIND = "openinference.span.kind"
    INPUT_VALUE = "input.value"
    INPUT_MIME_TYPE = "input.mime_type"
    OUTPUT_VALUE = "output.value"
    OUTPUT_MIME_TYPE = "output.mime_type"
