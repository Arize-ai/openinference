from __future__ import annotations

import json
import logging
import re
from json import JSONDecodeError
from typing import Any, Iterator, Mapping, Tuple, TypeVar

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import Message, get_llm_input_message_attributes
from openinference.instrumentation.bedrock.utils import _finish
from openinference.semconv.trace import (
    DocumentAttributes,
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

_AnyT = TypeVar("_AnyT")

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def fix_loose_json_string(s: str) -> list[dict[str, Any]]:
    loose_str = s.strip()
    if loose_str.startswith("[") and loose_str.endswith("]"):
        loose_str = loose_str[1:-1]

    # Find each dict-like string inside
    obj_strings = re.findall(r"\{.*?\}", loose_str)

    fixed_objects = []
    for obj_str in obj_strings:
        # Convert key=value to "key": "value"
        obj_fixed = re.sub(r"(\w+)=", r'"\1":', obj_str)

        # Add quotes around values that aren't already quoted
        obj_fixed = re.sub(r':\s*([^"{},\[\]]+)', r': "\1"', obj_fixed)

        # Make sure it's valid JSON
        obj_fixed = obj_fixed.replace("'", '"')

        try:
            fixed_obj = json.loads(obj_fixed)
            fixed_objects.append(fixed_obj)
        except json.JSONDecodeError:
            continue
    return fixed_objects


def sanitize_json_input(bad_json_str: str) -> str:
    # Escape single backslashes that are NOT part of known escape sequences
    # This will turn: \B -> \\B, \1 -> \\1, etc., but leave \n, \u1234, etc.
    def escape_bad_backslashes(match: Any) -> Any:
        return match.group(0).replace("\\", "\\\\")

    # This matches backslashes not followed by valid escape chars
    invalid_escape_re = re.compile(r'\\(?!["\\/bfnrtu])')
    cleaned = invalid_escape_re.sub(escape_bad_backslashes, bad_json_str)
    return cleaned


def safe_json_loads(json_str: str) -> Any:
    try:
        return json.loads(json_str)
    except JSONDecodeError:
        return json.loads(sanitize_json_input(json_str))


class _ResponseAccumulator:
    def __init__(
        self, span: Span, tracer: Tracer, request: Mapping[str, Any], idx: int = 0
    ) -> None:
        self._span = span
        self._request_parameters = request
        self.tracer = tracer
        self._is_finished: bool = False
        self.trace_values: dict[str, Any] = dict()
        self.chain_spans: dict[str, Span] = dict()
        self.trace_inputs_flags: dict[str, dict[str, bool]] = dict()

    def __call__(self, obj: _AnyT) -> _AnyT:
        try:
            span = self._span
            if isinstance(obj, dict):
                if "chunk" in obj:
                    if "bytes" in obj["chunk"]:
                        output_text = obj["chunk"]["bytes"].decode("utf-8")
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_text)
                elif "trace" in obj:
                    self._process_trace_event(obj["trace"]["trace"])
            elif isinstance(obj, (StopIteration, StopAsyncIteration)):
                self._finish_tracing()
            elif isinstance(obj, BaseException):
                self._finish_chain_spans()
                span.record_exception(obj)
                span.set_status(Status(StatusCode.ERROR, str(obj)))
                span.end()
        except Exception as e:
            logger.exception(e)
            self._span.record_exception(e)
            self._span.set_status(Status(StatusCode.ERROR))
            self._span.end()
            raise e
        return obj

    def _get_attributes_from_usage(
        self, usage: dict[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if input_tokens := usage.get("inputTokens"):
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, input_tokens
        if output_tokens := usage.get("outputTokens"):
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, output_tokens
        if (input_tokens := usage.get("inputTokens")) and (
            output_tokens := usage.get("outputTokens")
        ):
            yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, input_tokens + output_tokens

    def _get_messages_object(self, input_text: str) -> list[Message]:
        messages = list()
        input_messages = safe_json_loads(input_text)
        if system_message := input_messages.get("system"):
            messages.append(Message(content=system_message, role="system"))

        for message in input_messages.get("messages", []):
            role = message.get("role")
            if content := message.get("content"):
                parsed_contents = fix_loose_json_string(content) or [content]
                for parsed_content in parsed_contents:
                    message_content = content
                    if isinstance(parsed_content, dict):
                        if parsed_content_type := parsed_content.get("type"):
                            message_content = parsed_content.get(parsed_content_type)
                    messages.append(Message(content=message_content, role=role))
        return messages

    def _get_attributes_from_model_invocation_input_data(
        self, input_text: str
    ) -> Iterator[Tuple[str, Any]]:
        try:
            for k, v in get_llm_input_message_attributes(
                self._get_messages_object(input_text)
            ).items():
                yield k, v
        except Exception:
            messages = [Message(role="assistant", content=input_text)]
            for k, v in get_llm_input_message_attributes(messages).items():
                yield k, v

    def _get_attributes_from_model_invocation_input(
        self, trace_data: dict[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        model_invocation_input_parameters = trace_data.get("modelInvocationInput")
        if not model_invocation_input_parameters:
            return
        if input_text := model_invocation_input_parameters.get("text"):
            yield SpanAttributes.INPUT_VALUE, input_text
            yield from self._get_attributes_from_model_invocation_input_data(input_text)
        if foundation_model := model_invocation_input_parameters.get("foundationModel"):
            yield SpanAttributes.LLM_MODEL_NAME, foundation_model
        if inference_configuration := model_invocation_input_parameters.get(
            "inferenceConfiguration"
        ):
            yield SpanAttributes.LLM_INVOCATION_PARAMETERS, json.dumps(inference_configuration)

    def _get_attributes_from_message(
        self, message: dict[str, Any], role: str
    ) -> Iterator[Tuple[str, Any]]:
        if message.get("type") == "text":
            yield f"{MESSAGE_CONTENT}", message.get("text")
            yield f"{MESSAGE_ROLE}", role
        if message.get("type") == "tool_use":
            tool_prefix = f"{MESSAGE_TOOL_CALLS}.0"
            yield f"{tool_prefix}.{TOOL_CALL_ID}", message.get("id")
            yield f"{tool_prefix}.{TOOL_CALL_FUNCTION_NAME}", message.get("name")
            yield (
                f"{tool_prefix}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                json.dumps(message.get("input")),
            )
            yield f"{MESSAGE_ROLE}", "tool"

    def _get_attributes_from_model_invocation_output_params(
        self, model_output: dict[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if raw_response := model_output.get("rawResponse"):
            if output_text := raw_response.get("content"):
                try:
                    yield OUTPUT_VALUE, output_text
                    data = json.loads(str(output_text))
                    yield LLM_MODEL_NAME, data.get("model")
                    for idx, content in enumerate(data.get("content") or []):
                        for key, value in self._get_attributes_from_message(
                            content, content.get("role", "assistant")
                        ):
                            yield f"{LLM_OUTPUT_MESSAGES}.{idx}.{key}", value
                except Exception:
                    yield f"{LLM_OUTPUT_MESSAGES}.{0}.{MESSAGE_CONTENT}", output_text
                    yield f"{LLM_OUTPUT_MESSAGES}.{0}.{MESSAGE_ROLE}", "assistant"
        if output_text := model_output.get("parsedResponse", {}).get("text"):
            # This block will be executed for Post Processing trace
            yield OUTPUT_VALUE, output_text

        if output_text := model_output.get("parsedResponse", {}).get("rationale"):
            # This block will be executed for Pre Processing trace
            yield OUTPUT_VALUE, output_text

    def _get_attributes_from_model_invocation_output(
        self, trace_data: dict[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        model_invocation_output_parameters = trace_data.get("modelInvocationOutput") or {}

        if metadata := model_invocation_output_parameters.get("metadata"):
            yield from self._get_attributes_from_usage(metadata.get("usage"))

        if inference_configuration := model_invocation_output_parameters.get(
            "inferenceConfiguration"
        ):
            yield LLM_INVOCATION_PARAMETERS, inference_configuration
        yield from self._get_attributes_from_model_invocation_output_params(
            model_invocation_output_parameters
        )

    def _get_attributes_from_code_interpreter_input(
        self, code_input: dict[str, Any]
    ) -> Iterator[Tuple[str, Any]]:
        yield OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value
        yield TOOL_NAME, "code_interpreter"
        yield (
            TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
            json.dumps({"code": code_input.get("code", ""), "files": code_input.get("files", "")}),
        )
        yield TOOL_DESCRIPTION, "Executes code and returns results"
        yield (
            TOOL_PARAMETERS,
            json.dumps({"code": {"type": "string", "description": "Code to execute"}}),
        )
        yield SpanAttributes.INPUT_VALUE, code_input.get("code", "")
        yield (
            SpanAttributes.METADATA,
            json.dumps(
                {
                    "invocation_type": "code_execution",
                    "execution_context": code_input.get("context", {}),
                }
            ),
        )

    def _get_attributes_from_knowledge_base_lookup_input(
        self, kb_data: dict[str, Any]
    ) -> Iterator[Tuple[str, Any]]:
        metadata = {
            "invocation_type": "knowledge_base_lookup",
            "knowledge_base_id": kb_data.get("knowledgeBaseId"),
        }
        yield SpanAttributes.METADATA, json.dumps(metadata)
        yield SpanAttributes.INPUT_VALUE, kb_data.get("text")

    def _get_attributes_from_action_group_invocation_input(
        self, action_input: dict[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value

        prefix = f"{LLM_INPUT_MESSAGES}.{0}.{MESSAGE_TOOL_CALLS}.0"
        yield f"{prefix}.{TOOL_CALL_FUNCTION_NAME}", action_input.get("function", "")
        yield (
            f"{prefix}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
            json.dumps(
                {
                    "name": action_input.get("function", ""),
                    "arguments": action_input.get("parameters", {}),
                }
            ),
        )
        yield f"{LLM_INPUT_MESSAGES}.{0}.{MESSAGE_ROLE}", "assistant"
        yield TOOL_NAME, action_input.get("function", "")
        yield TOOL_DESCRIPTION, action_input.get("description", "")
        yield TOOL_PARAMETERS, json.dumps(action_input.get("parameters", []))
        yield (
            TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
            json.dumps(
                {
                    "name": action_input.get("function", ""),
                    "arguments": action_input.get("parameters", {}),
                }
            ),
        )
        yield TOOL_CALL_FUNCTION_NAME, action_input.get("function", "")
        yield (
            SpanAttributes.LLM_INVOCATION_PARAMETERS,
            json.dumps(
                {
                    "invocation_type": "action_group_invocation",
                    "action_group_name": action_input.get("actionGroupName"),
                    "execution_type": action_input.get("executionType"),
                    "invocation_id": action_input.get("invocationId"),
                    "verb": action_input.get("verb"),
                }
            ),
        )

    def _get_attributes_from_code_interpreter_output(
        self, code_invocation_output: dict[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if output_text := code_invocation_output.get("executionOutput"):
            yield SpanAttributes.OUTPUT_VALUE, output_text
        if execution_error := code_invocation_output.get("executionError"):
            yield SpanAttributes.OUTPUT_VALUE, execution_error
        if code_invocation_output.get("executionTimeout"):
            yield SpanAttributes.OUTPUT_VALUE, "Execution Timeout Error"
        if files := code_invocation_output.get("files"):
            yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}", json.dumps(files)
            yield f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}", "tool"

    def _get_attributes_from_knowledge_base_lookup_output(
        self, knowledge_base_lookup_output: dict[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        retrieved_refs = knowledge_base_lookup_output.get("retrievedReferences", [])
        for i, ref in enumerate(retrieved_refs):
            if document_id := ref.get("metadata", {}).get("x-amz-bedrock-kb-chunk-id", ""):
                yield f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_ID}", document_id

            if document_content := ref.get("content", {}).get("text"):
                yield f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_CONTENT}", document_content

            if document_score := ref.get("score", 0.0):
                yield f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_SCORE}", document_score
            metadata = json.dumps(
                {
                    "location": ref.get("location", {}),
                    "metadata": ref.get("metadata", {}),
                    "type": ref.get("content", {}).get("type"),
                }
            )
            yield f"{RETRIEVAL_DOCUMENTS}.{i}.{DOCUMENT_METADATA}", metadata

    def _get_attributes_from_action_group_invocation_output(
        self, tool_output: dict[str, Any]
    ) -> Iterator[Tuple[str, Any]]:
        yield SpanAttributes.OUTPUT_VALUE, tool_output.get("text")

    def _get_event_type(self, trace_data: dict[str, Any]) -> str:
        """
        Identifies the type of trace event from the provided trace data.

        This method checks the trace data for specific event types such as
        'preProcessingTrace', 'orchestrationTrace', 'postProcessingTrace',
        or 'failureTrace'. If a matching event type is found, it is returned.
        Otherwise, the method returns `None`.

        Args:
            trace_data (dict[str, Any]): The trace data containing information
            about the event.

        Returns:
            str | None: The identified event type if found, otherwise `None`.
        """

        trace_events = [
            "preProcessingTrace",
            "orchestrationTrace",
            "postProcessingTrace",
            "failureTrace",
        ]
        for trace_event in trace_events:
            if trace_event in trace_data:
                return trace_event
        return ""

    def _get_attributes_from_invocation_input(
        self, trace_data: dict[str, Any]
    ) -> Iterator[Tuple[str, Any]]:
        if invocation_input := trace_data.get("invocationInput"):
            if "actionGroupInvocationInput" in invocation_input:
                yield OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value
                yield from self._get_attributes_from_action_group_invocation_input(
                    invocation_input["actionGroupInvocationInput"]
                )
            if "codeInterpreterInvocationInput" in invocation_input:
                yield OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.TOOL.value
                yield from self._get_attributes_from_code_interpreter_input(
                    invocation_input["codeInterpreterInvocationInput"]
                )
            if "knowledgeBaseLookupInput" in invocation_input:
                yield OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value
                yield from self._get_attributes_from_knowledge_base_lookup_input(
                    invocation_input["knowledgeBaseLookupInput"]
                )

    def _get_attributes_from_observation(
        self, trace_data: dict[str, Any]
    ) -> Iterator[Tuple[str, Any]]:
        if observation := trace_data.get("observation"):
            if "actionGroupInvocationOutput" in observation:
                yield from self._get_attributes_from_action_group_invocation_output(
                    observation["actionGroupInvocationOutput"]
                )
            if "codeInterpreterInvocationOutput" in observation:
                yield from self._get_attributes_from_code_interpreter_output(
                    observation["codeInterpreterInvocationOutput"]
                )
            if "knowledgeBaseLookupOutput" in observation:
                yield from self._get_attributes_from_knowledge_base_lookup_output(
                    observation["knowledgeBaseLookupOutput"]
                )

    def _get_attributes_from_pre_and_post_processing_trace(
        self, trace_data: dict[str, Any], trace_event: str
    ) -> None:
        """
        Processes pre-processing and post-processing trace events.

        This method validates and stores `modelInvocationInput` and `modelInvocationOutput`
        events for the specified trace type. If both events are present, it delegates
        the request to create child spans for the trace.

        Args:
            trace_data (dict[str, Any]): The trace data containing information about
            the event to be processed.
            trace_event (str): The type of trace event (e.g., preProcessingTrace,
            postProcessingTrace).

        Returns:
            None
        """
        self.trace_values.setdefault(trace_event, {})
        if "modelInvocationInput" in trace_data:
            self.trace_values[trace_event]["modelInvocationInput"] = trace_data
        if "modelInvocationOutput" in trace_data:
            self.trace_values[trace_event]["modelInvocationOutput"] = trace_data
            self._create_model_invocation_span(trace_data, trace_event)

        if "rationale" in trace_data:
            preprocessing_span = self._initialize_chain_span(trace_event)
            if rationale_text := trace_data.get("rationale", {}).get("text"):
                preprocessing_span.set_attribute(SpanAttributes.OUTPUT_VALUE, rationale_text)

    def _add_model_invocation_attributes_to_parent_span(
        self, trace_event: str, event_type: str
    ) -> None:
        self.trace_inputs_flags.setdefault(trace_event, {})
        model_output = self.trace_values[trace_event].get(event_type, {}).get(event_type)
        if model_output and not self.trace_inputs_flags.get(trace_event, {}).get("has_input_value"):
            parent_trace = self._initialize_chain_span(trace_event)
            try:
                for message in self._get_messages_object(model_output.get("text")):
                    if message.get("role") == "user" and (input_value := message.get("content")):
                        parent_trace.set_attribute(INPUT_VALUE, input_value)
                        self.trace_inputs_flags[trace_event]["has_input_value"] = True
                        break
            except Exception:
                parent_trace.set_attribute(INPUT_VALUE, model_output.get("text"))
                self.trace_inputs_flags[trace_event]["has_input_value"] = True

    def _add_invocation_attributes_to_parent_span(
        self, trace_event: str, trace_data: dict[str, Any]
    ) -> None:
        self.trace_inputs_flags.setdefault(trace_event, {})
        if self.trace_inputs_flags.get(trace_event, {}).get("has_input_value"):
            return
        parent_trace = self._initialize_chain_span(trace_event)

        if invocation_input := trace_data.get("invocationInput"):
            if input_value := invocation_input.get("actionGroupInvocationInput", {}).get("text"):
                parent_trace.set_attribute(INPUT_VALUE, input_value)
                self.trace_inputs_flags[trace_event]["has_input_value"] = True

            if input_value := invocation_input.get("codeInterpreterInvocationInput", {}).get(
                "code"
            ):
                parent_trace.set_attribute(INPUT_VALUE, input_value)
                self.trace_inputs_flags[trace_event]["has_input_value"] = True

            if input_value := invocation_input.get("knowledgeBaseLookupInput", {}).get("text"):
                parent_trace.set_attribute(INPUT_VALUE, input_value)
                self.trace_inputs_flags[trace_event]["has_input_value"] = True

    def _set_parent_trace_output(self, trace_event: str, event_type: str) -> None:
        parent_trace = self._initialize_chain_span(trace_event)
        model_output = self.trace_values[trace_event][event_type].get(event_type)

        if output_text := model_output.get("parsedResponse", {}).get("text"):
            # This block will be executed for Post Processing trace
            parent_trace.set_attribute(OUTPUT_VALUE, output_text)

        if output_text := model_output.get("parsedResponse", {}).get("rationale"):
            # This block will be executed for Pre Processing trace
            parent_trace.set_attribute(OUTPUT_VALUE, output_text)

    def _create_model_invocation_span(self, trace_data: dict[str, Any], trace_event: str) -> None:
        """
        Creates child traces for preProcessing, orchestration, and postProcessing LLM traces.

        This function sets attributes for the child traces and assigns the input and output values
        to the corresponding group traces.

        Args:
            trace_data (dict[str, Any]): The trace data containing information about the
            model invocation.
            trace_event (str): The type of trace event (e.g., preProcessingTrace,
            orchestrationTrace, etc.).

        Returns:
            None
        """
        if "modelInvocationOutput" not in trace_data:
            return
        parent_trace = self._initialize_chain_span(trace_event)
        with self.tracer.start_as_current_span(
            name="LLM", context=trace_api.set_span_in_context(parent_trace)
        ) as model_invocation_span:
            request_attributes = dict(
                self._get_attributes_from_model_invocation_input(
                    self.trace_values[trace_event]["modelInvocationInput"]
                )
            )
            model_invocation_span.set_attributes(request_attributes)
            response_attributes = dict(
                self._get_attributes_from_model_invocation_output(
                    self.trace_values[trace_event]["modelInvocationOutput"]
                )
            )
            model_invocation_span.set_attributes(response_attributes)
            model_invocation_span.set_attribute(
                OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value
            )
            model_invocation_span.set_status(Status(StatusCode.OK))
            self._add_model_invocation_attributes_to_parent_span(
                trace_event, "modelInvocationInput"
            )
            self._set_parent_trace_output(trace_event, "modelInvocationOutput")
            self.trace_values[trace_event] = dict()

    def _create_invocation_span(self, trace_data: dict[str, Any], trace_event: str) -> None:
        orchestration_trace_values = self.trace_values[trace_event]
        if "observation" not in trace_data or not orchestration_trace_values.get("invocationInput"):
            return
        invocation_input_trace = orchestration_trace_values["invocationInput"].get(
            "invocationInput"
        )
        with self.tracer.start_as_current_span(
            name=invocation_input_trace.get("invocationType").lower(),
            context=trace_api.set_span_in_context(self._initialize_chain_span(trace_event)),
        ) as invocation_span:
            request_attributes = dict(
                self._get_attributes_from_invocation_input(
                    orchestration_trace_values["invocationInput"]
                )
            )
            invocation_span.set_attributes(request_attributes)
            response_attributes = dict(
                self._get_attributes_from_observation(orchestration_trace_values["observation"])
            )
            invocation_span.set_attributes(response_attributes)
            self._add_invocation_attributes_to_parent_span(
                trace_event, orchestration_trace_values["invocationInput"]
            )
            invocation_span.set_status(Status(StatusCode.OK))

    def _get_attributes_from_orchestration_trace(
        self, trace_data: dict[str, Any], trace_event: str
    ) -> Any:
        events = [
            "invocationInput",
            "modelInvocationInput",
            "modelInvocationOutput",
            "observation",
            "rationale",
        ]
        self.trace_values.setdefault(trace_event, dict())
        for event in events:
            if event not in trace_data:
                continue
            self.trace_values[trace_event][event] = trace_data
        self._create_invocation_span(trace_data, trace_event)
        self._create_model_invocation_span(trace_data, trace_event)
        if final_response := trace_data.get("observation", {}).get("finalResponse"):
            orchestration_span = self._initialize_chain_span(trace_event)
            orchestration_span.set_attribute(OUTPUT_VALUE, final_response.get("text"))
            orchestration_span.set_status(Status(StatusCode.OK))
            self.trace_values[trace_event] = {}

    def _initialize_chain_span(self, trace_type: str) -> Span:
        """
        Initializes or retrieves a chain span for the given trace type.

        This function ensures that a group span (chain span) is created for the specified
        trace type if it does not already exist. If the chain span is already created,
        it retrieves and returns the existing span.

        Args:
            trace_type (str): The type of trace for which the chain span is being initialized.

        Returns:
            Span: The initialized or retrieved chain span for the given trace type.
        """
        if trace_type not in self.chain_spans:
            self.chain_spans[trace_type] = self.tracer.start_span(
                name=trace_type,
                attributes={OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value},
                context=trace_api.set_span_in_context(self._span),
            )
        return self.chain_spans[trace_type]

    def _process_trace_event(self, trace_data: dict[str, Any]) -> None:
        """
        Processes a trace event and delegates it to the appropriate handler
        based on the event type.

        This function identifies the type of trace event (e.g., preProcessingTrace,
        postProcessingTrace, orchestrationTrace, etc.) and invokes the corresponding
        method to handle the trace data. It updates the trace values and attributes
        for the event accordingly.

        Args:
            trace_data (dict[str, Any]): The trace data containing information
            about the event to be processed.

        Returns:
            None
        """
        trace_event = self._get_event_type(trace_data)
        if trace_event in ["preProcessingTrace", "postProcessingTrace"]:
            self._get_attributes_from_pre_and_post_processing_trace(
                trace_data.get(trace_event) or {}, trace_event
            )
        elif trace_event == "orchestrationTrace":
            self._get_attributes_from_orchestration_trace(
                trace_data.get(trace_event) or {}, trace_event
            )

    def _finish_tracing(self) -> None:
        if self._is_finished:
            return
        self._finish_chain_spans()
        _finish(self._span, None, self._request_parameters)
        self._is_finished = True

    def _finish_chain_spans(self) -> None:
        for chain_span in self.chain_spans.values():
            chain_span.set_status(Status(StatusCode.OK))
            chain_span.end()


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_NAME = SpanAttributes.TOOL_NAME
TOOL_DESCRIPTION = SpanAttributes.TOOL_DESCRIPTION
TOOL_PARAMETERS = SpanAttributes.TOOL_PARAMETERS

INPUT_VALUE = SpanAttributes.INPUT_VALUE
METADATA = SpanAttributes.METADATA
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
