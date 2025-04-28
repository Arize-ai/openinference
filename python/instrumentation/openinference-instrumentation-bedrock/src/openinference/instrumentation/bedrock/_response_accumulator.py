from __future__ import annotations

import json
import logging
import re
from json import JSONDecodeError
from typing import Any, Dict, Mapping, TypeVar

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span, Status, StatusCode, Tracer
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    Message,
    TokenCount,
    ToolCall,
    ToolCallFunction,
    get_input_attributes,
    get_llm_attributes,
    get_llm_input_message_attributes,
    get_llm_output_message_attributes,
    get_output_attributes,
    get_span_kind_attributes,
    get_tool_attributes,
)
from openinference.instrumentation.bedrock.utils import _finish
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_AnyT = TypeVar("_AnyT")

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def fix_loose_json_string(s: str) -> list[dict[str, Any]]:
    """
    Converts a loosely formatted JSON string into a list of dictionaries.

    Args:
        s (str): The loosely formatted JSON string.

    Returns:
        list[dict[str, Any]]: A list of dictionaries parsed from the string.
    """
    loose_str = s.strip()
    if loose_str.startswith("[") and loose_str.endswith("]"):
        loose_str = loose_str[1:-1]

    obj_strings = re.findall(r"\{.*?\}", loose_str)
    fixed_objects = []

    for obj_str in obj_strings:
        obj_fixed = re.sub(r"(\w+)=", r'"\1":', obj_str)
        obj_fixed = re.sub(r':\s*([^"{},\[\]]+)', r': "\1"', obj_fixed)
        obj_fixed = obj_fixed.replace("'", '"')

        try:
            fixed_obj = json.loads(obj_fixed)
            fixed_objects.append(fixed_obj)
        except json.JSONDecodeError:
            logger.debug(f"Failed to decode JSON object: {obj_fixed}")
            continue

    return fixed_objects


def sanitize_json_input(bad_json_str: str) -> str:
    """
    Cleans a JSON string by escaping invalid backslashes.

    Args:
        bad_json_str (str): The JSON string with potential invalid backslashes.

    Returns:
        str: The sanitized JSON string.
    """

    def escape_bad_backslashes(match: Any) -> Any:
        return match.group(0).replace("\\", "\\\\")

    invalid_escape_re = re.compile(r'\\(?!["\\/bfnrtu])')
    cleaned = invalid_escape_re.sub(escape_bad_backslashes, bad_json_str)
    return cleaned


def safe_json_loads(json_str: str) -> Any:
    """
    Safely loads a JSON string, attempting to sanitize it if initial loading fails.

    Args:
        json_str (str): The JSON string to load.

    Returns:
        Any: The loaded JSON object.
    """
    try:
        return json.loads(json_str)
    except JSONDecodeError as e:
        logger.debug(f"JSONDecodeError encountered: {e}. Attempting to sanitize input.")
        return json.loads(sanitize_json_input(json_str))


class _ResponseAccumulator:
    """
    Accumulates and processes responses from Bedrock service.

    This class handles the processing of trace events, creating spans for different
    types of events, and managing the lifecycle of these spans.
    """

    def __init__(
        self, span: Span, tracer: Tracer, request: Mapping[str, Any], idx: int = 0
    ) -> None:
        """
        Initialize the ResponseAccumulator.

        Args:
            span (Span): The parent span for tracing.
            tracer (Tracer): The tracer instance.
            request (Mapping[str, Any]): The request parameters.
            idx (int, optional): Index for the accumulator. Defaults to 0.
        """
        self._span = span
        self._request_parameters = request
        self.tracer = tracer
        self._is_finished: bool = False
        self.trace_values: dict[str, Any] = dict()
        self.chain_spans: dict[str, Span] = dict()
        self.trace_inputs_flags: dict[str, dict[str, bool]] = dict()

    def __call__(self, obj: _AnyT) -> _AnyT:
        """
        Process an object received from the Bedrock service.

        Args:
            obj (_AnyT): The object to process.

        Returns:
            _AnyT: The processed object.
        """
        try:
            if isinstance(obj, dict):
                self._process_dict_object(obj)
            elif isinstance(obj, (StopIteration, StopAsyncIteration)):
                self._finish_tracing()
            elif isinstance(obj, BaseException):
                self._handle_exception(obj)
        except Exception as e:
            logger.exception(e)
            self._span.record_exception(e)
            self._span.set_status(Status(StatusCode.ERROR))
            self._span.end()
            raise e
        return obj

    def _process_dict_object(self, obj: Dict[str, Any]) -> None:
        """
        Process a dictionary object received from the Bedrock service.

        Args:
            obj (dict): The dictionary object to process.
        """
        if "chunk" in obj:
            if "bytes" in obj["chunk"]:
                output_text = obj["chunk"]["bytes"].decode("utf-8")
                self._span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_text)
        elif "trace" in obj:
            self._process_trace_event(obj["trace"]["trace"])

    def _handle_exception(self, obj: BaseException) -> None:
        """
        Handle an exception object.

        Args:
            obj (BaseException): The exception to handle.
        """
        self._finish_chain_spans()
        self._span.record_exception(obj)
        self._span.set_status(Status(StatusCode.ERROR, str(obj)))
        self._span.end()

    @classmethod
    def _get_messages_object(cls, input_text: str) -> list[Message]:
        """
        Parse input text into a list of Message objects.

        Args:
            input_text (str): The input text to parse.

        Returns:
            list[Message]: A list of parsed Message objects.
        """
        messages = list()
        try:
            input_messages = safe_json_loads(input_text)
            if system_message := input_messages.get("system"):
                messages.append(Message(content=system_message, role="system"))

            for message in input_messages.get("messages", []):
                role = message.get("role", "")
                if content := message.get("content"):
                    parsed_contents = fix_loose_json_string(content) or [content]
                    for parsed_content in parsed_contents:
                        message_content = content
                        if isinstance(parsed_content, dict):
                            if parsed_content_type := parsed_content.get("type"):
                                message_content = parsed_content.get(parsed_content_type, "")
                        messages.append(Message(content=message_content, role=role))
        except Exception:
            return [Message(content=input_text, role="assistant")]
        return messages

    @classmethod
    def _get_attributes_from_message(cls, message: Dict[str, Any], role: str) -> Message | None:
        """
        Extract attributes from a message dictionary.

        Args:
            message (dict[str, Any]): The message dictionary.
            role (str): The role of the message.

        Returns:
            Message | None: A Message object if attributes can be extracted, None otherwise.
        """
        if message.get("type") == "text":
            return Message(content=message.get("text", ""), role=role)
        if message.get("type") == "tool_use":
            tool_call_function = ToolCallFunction(
                name=message.get("name", ""), arguments=message.get("input", {})
            )
            tool_calls = [ToolCall(id=message.get("id", ""), function=tool_call_function)]
            return Message(tool_call_id=message.get("id", ""), role="tool", tool_calls=tool_calls)
        return None

    def _get_output_messages(self, model_output: dict[str, Any]) -> list[Message] | None:
        """
        Extract output messages from model output.

        Args:
            model_output (dict[str, Any]): The model output dictionary.

        Returns:
            list[Message] | None: A list of Message objects if messages can be extracted,
            None otherwise.
        """
        if raw_response := model_output.get("rawResponse"):
            if output_text := raw_response.get("content"):
                try:
                    data = json.loads(str(output_text))
                    messages = list()
                    for content in data.get("content") or []:
                        if message := self._get_attributes_from_message(
                            content, content.get("role", "assistant")
                        ):
                            messages.append(message)
                    return messages
                except Exception:
                    message = Message(content=str(output_text), role="assistant")
                    return [message]
        return None

    @classmethod
    def _get_attributes_from_code_interpreter_input(
        cls, code_input: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract attributes from code interpreter input.

        Args:
            code_input (dict[str, Any]): The code interpreter input dictionary.

        Returns:
            dict[str, Any]: A dictionary of extracted attributes.
        """
        tool_call_function = ToolCallFunction(
            name="code_interpreter",
            arguments={"code": code_input.get("code", ""), "files": code_input.get("files", "")},
        )
        tool_calls = [ToolCall(id="default", function=tool_call_function)]
        messages = [Message(tool_call_id="default", role="tool", tool_calls=tool_calls)]
        name = "code_interpreter"
        description = "Executes code and returns results"
        parameters = json.dumps({"code": {"type": "string", "description": "Code to execute"}})
        metadata = json.dumps(
            {
                "invocation_type": "code_execution",
                "execution_context": code_input.get("context", {}),
            }
        )
        return {
            **get_input_attributes(code_input.get("code", "")),
            **get_span_kind_attributes(OpenInferenceSpanKindValues.TOOL),
            **get_llm_input_message_attributes(messages),
            **get_tool_attributes(name=name, description=description, parameters=parameters),
            **{"metadata": metadata},
        }

    @classmethod
    def _get_attributes_from_knowledge_base_lookup_input(
        cls, kb_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract attributes from knowledge base lookup input.

        Args:
            kb_data (dict[str, Any]): The knowledge base lookup input dictionary.

        Returns:
            dict[str, Any]: A dictionary of extracted attributes.
        """
        metadata = {
            "invocation_type": "knowledge_base_lookup",
            "knowledge_base_id": kb_data.get("knowledgeBaseId"),
        }
        return {
            **get_input_attributes(kb_data.get("text", "")),
            **get_span_kind_attributes(OpenInferenceSpanKindValues.RETRIEVER),
            **{"metadata": json.dumps(metadata)},
        }

    @classmethod
    def _get_attributes_from_action_group_invocation_input(
        cls, action_input: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract attributes from action group invocation input.

        Args:
            action_input (dict[str, Any]): The action group invocation input dictionary.

        Returns:
            dict[str, Any]: A dictionary of extracted attributes.
        """
        name = action_input.get("function", "")
        tool_call_function = ToolCallFunction(
            name=name, arguments=action_input.get("parameters", {})
        )
        tool_calls = [ToolCall(id="default", function=tool_call_function)]
        messages = [Message(tool_call_id="default", role="tool", tool_calls=tool_calls)]
        description = action_input.get("description", "")
        parameters = json.dumps(action_input.get("parameters", []))
        llm_invocation_parameters = {
            "invocation_type": "action_group_invocation",
            "action_group_name": action_input.get("actionGroupName"),
            "execution_type": action_input.get("executionType"),
        }
        if invocation_id := action_input.get("invocationId"):
            llm_invocation_parameters["invocation_id"] = invocation_id
        if verb := action_input.get("verb"):
            llm_invocation_parameters["verb"] = verb
        return {
            **get_span_kind_attributes(OpenInferenceSpanKindValues.TOOL),
            **get_llm_input_message_attributes(messages),
            **get_tool_attributes(name=name, description=description, parameters=parameters),
            **{"metadata": json.dumps(llm_invocation_parameters)},
        }

    @classmethod
    def _get_attributes_from_code_interpreter_output(
        cls, code_invocation_output: dict[str, Any]
    ) -> Dict[str, AttributeValue]:
        """
        Extract attributes from code interpreter output.

        Args:
            code_invocation_output (dict[str, Any]): The code interpreter output dictionary.

        Returns:
            Dict[str, AttributeValue]: A dictionary of extracted attributes.
        """
        output_value = None
        files = None

        if output_text := code_invocation_output.get("executionOutput"):
            output_value = output_text
        elif execution_error := code_invocation_output.get("executionError"):
            output_value = execution_error
        elif code_invocation_output.get("executionTimeout"):
            output_value = "Execution Timeout Error"
        elif files := code_invocation_output.get("files"):
            output_value = json.dumps(files)

        content = json.dumps(files) if files else str(output_value) if output_value else ""
        messages = [Message(role="tool", content=content)]
        return {
            **get_output_attributes(output_value),
            **get_llm_output_message_attributes(messages),
        }

    @classmethod
    def _get_attributes_from_knowledge_base_lookup_output(
        cls, knowledge_base_lookup_output: dict[str, Any]
    ) -> Dict[str, AttributeValue]:
        """
        Extract attributes from knowledge base lookup output.

        Args:
            knowledge_base_lookup_output (dict[str, Any]): The knowledge base lookup
            output dictionary.

        Returns:
            Dict[str, AttributeValue]: A dictionary of extracted attributes.
        """
        retrieved_refs = knowledge_base_lookup_output.get("retrievedReferences", [])
        attributes = dict()
        for i, ref in enumerate(retrieved_refs):
            base_key = f"{RETRIEVAL_DOCUMENTS}.{i}"
            if document_id := ref.get("metadata", {}).get("x-amz-bedrock-kb-chunk-id", ""):
                attributes[f"{base_key}.{DOCUMENT_ID}"] = document_id

            if document_content := ref.get("content", {}).get("text"):
                attributes[f"{base_key}.{DOCUMENT_CONTENT}"] = document_content

            if document_score := ref.get("score", 0.0):
                attributes[f"{base_key}.{DOCUMENT_SCORE}"] = document_score
            metadata = json.dumps(
                {
                    "location": ref.get("location", {}),
                    "metadata": ref.get("metadata", {}),
                    "type": ref.get("content", {}).get("type"),
                }
            )
            attributes[f"{base_key}.{DOCUMENT_METADATA}"] = metadata
        return attributes

    @classmethod
    def _get_event_type(cls, trace_data: dict[str, Any]) -> str:
        """
        Identifies the type of trace event from the provided trace data.

        Args:
            trace_data (dict[str, Any]): The trace data containing information
            about the event.

        Returns:
            str: The identified event type if found, otherwise an empty string.
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
    ) -> dict[str, Any] | None:
        """
        Extract attributes from invocation input.

        Args:
            trace_data (dict[str, Any]): The trace data dictionary.

        Returns:
            dict[str, Any] | None: A dictionary of extracted attributes if available,
            None otherwise.
        """
        if invocation_input := trace_data.get("invocationInput"):
            if "actionGroupInvocationInput" in invocation_input:
                return self._get_attributes_from_action_group_invocation_input(
                    invocation_input["actionGroupInvocationInput"]
                )
            if "codeInterpreterInvocationInput" in invocation_input:
                return self._get_attributes_from_code_interpreter_input(
                    invocation_input["codeInterpreterInvocationInput"]
                )
            if "knowledgeBaseLookupInput" in invocation_input:
                return self._get_attributes_from_knowledge_base_lookup_input(
                    invocation_input["knowledgeBaseLookupInput"]
                )
        return None

    def _get_attributes_from_observation(
        self, trace_data: dict[str, Any]
    ) -> Dict[str, AttributeValue]:
        """
        Extract attributes from observation data.

        Args:
            trace_data (dict[str, Any]): The trace data dictionary.

        Returns:
            Dict[str, AttributeValue]: A dictionary of extracted attributes.
        """
        if observation := trace_data.get("observation"):
            if "actionGroupInvocationOutput" in observation:
                tool_output = observation["actionGroupInvocationOutput"]
                return get_output_attributes(tool_output.get("text", ""))
            if "codeInterpreterInvocationOutput" in observation:
                return self._get_attributes_from_code_interpreter_output(
                    observation["codeInterpreterInvocationOutput"]
                )
            if "knowledgeBaseLookupOutput" in observation:
                return self._get_attributes_from_knowledge_base_lookup_output(
                    observation["knowledgeBaseLookupOutput"]
                )
        return {}

    def _get_attributes_from_pre_and_post_processing_trace(
        self, trace_data: dict[str, Any], trace_event: str
    ) -> None:
        """
        Process pre-processing and post-processing trace events.

        Args:
            trace_data (dict[str, Any]): The trace data containing information about
            the event to be processed.
            trace_event (str): The type of trace event (e.g., preProcessingTrace,
            postProcessingTrace).
        """
        self.trace_values.setdefault(trace_event, {})
        if "modelInvocationInput" in trace_data:
            self.trace_values[trace_event]["modelInvocationInput"] = trace_data
        if "modelInvocationOutput" in trace_data:
            self.trace_values[trace_event]["modelInvocationOutput"] = trace_data
            self._create_model_invocation_span(trace_data, trace_event)

        if "rationale" in trace_data:
            preprocessing_span = self._initialize_chain_span(trace_event)
            if rationale_text := trace_data.get("rationale", {}).get("text", ""):
                preprocessing_span.set_attributes(get_output_attributes(rationale_text))

    def _add_model_invocation_attributes_to_parent_span(
        self, trace_event: str, event_type: str
    ) -> None:
        """
        Add model invocation attributes to the parent span.

        Args:
            trace_event (str): The type of trace event.
            event_type (str): The type of event.
        """
        self.trace_inputs_flags.setdefault(trace_event, {})
        model_output = self.trace_values[trace_event].get(event_type, {}).get(event_type)
        if model_output and not self.trace_inputs_flags.get(trace_event, {}).get("has_input_value"):
            parent_trace = self._initialize_chain_span(trace_event)
            try:
                text = model_output.get("text", "")
                for message in self._get_messages_object(text):
                    if message.get("role") == "user" and (input_value := message.get("content")):
                        parent_trace.set_attributes(get_input_attributes(input_value))
                        self.trace_inputs_flags[trace_event]["has_input_value"] = True
                        break
            except Exception:
                parent_trace.set_attributes(get_input_attributes(model_output.get("text", "")))
                self.trace_inputs_flags[trace_event]["has_input_value"] = True

    def _add_invocation_attributes_to_parent_span(
        self, trace_event: str, trace_data: dict[str, Any]
    ) -> None:
        """
        Add invocation attributes to the parent span.

        Args:
            trace_event (str): The type of trace event.
            trace_data (dict[str, Any]): The trace data dictionary.
        """
        self.trace_inputs_flags.setdefault(trace_event, {})
        if self.trace_inputs_flags.get(trace_event, {}).get("has_input_value"):
            return
        parent_trace = self._initialize_chain_span(trace_event)

        if invocation_input := trace_data.get("invocationInput"):
            action_group = invocation_input.get("actionGroupInvocationInput", {})
            if input_value := action_group.get("text", ""):
                parent_trace.set_attributes(get_input_attributes(input_value))
                self.trace_inputs_flags[trace_event]["has_input_value"] = True

            code_interpreter = invocation_input.get("codeInterpreterInvocationInput", {})
            if input_value := code_interpreter.get("code", ""):
                parent_trace.set_attributes(get_input_attributes(input_value))
                self.trace_inputs_flags[trace_event]["has_input_value"] = True

            kb_lookup = invocation_input.get("knowledgeBaseLookupInput", {})
            if input_value := kb_lookup.get("text", ""):
                parent_trace.set_attributes(get_input_attributes(input_value))
                self.trace_inputs_flags[trace_event]["has_input_value"] = True

    def _set_parent_trace_output(self, trace_event: str, event_type: str) -> None:
        """
        Set the output value for the parent trace.

        Args:
            trace_event (str): The type of trace event.
            event_type (str): The type of event.
        """
        parent_trace = self._initialize_chain_span(trace_event)
        model_output = self.trace_values[trace_event][event_type].get(event_type, {})

        parsed_response = model_output.get("parsedResponse", {})
        if output_text := parsed_response.get("text", ""):
            # This block will be executed for Post Processing trace
            parent_trace.set_attributes(get_output_attributes(output_text))

        if output_text := parsed_response.get("rationale", ""):
            # This block will be executed for Pre Processing trace
            parent_trace.set_attributes(get_output_attributes(output_text))

    @classmethod
    def _get_model_name(
        cls, input_params: dict[str, Any], output_params: dict[str, Any]
    ) -> str | None:
        """
        Get the model name from input or output parameters.

        Args:
            input_params (dict[str, Any]): The input parameters.
            output_params (dict[str, Any]): The output parameters.

        Returns:
            str | None: The model name if found, None otherwise.
        """
        if model_name := input_params.get("foundationModel"):
            return str(model_name)
        if raw_response := output_params.get("rawResponse"):
            if output_text := raw_response.get("content"):
                try:
                    data = json.loads(str(output_text))
                    model = data.get("model")
                    if model is not None:
                        return str(model)
                except Exception as e:
                    logger.debug(str(e))
        return None

    @classmethod
    def _get_invocation_parameters(
        cls, input_params: dict[str, Any], output_params: dict[str, Any]
    ) -> str | None:
        """
        Get the invocation parameters from input or output parameters.

        Args:
            input_params (dict[str, Any]): The input parameters.
            output_params (dict[str, Any]): The output parameters.

        Returns:
            str | None: The invocation parameters as a JSON string if found, None otherwise.
        """
        if inference_configuration := input_params.get("inferenceConfiguration"):
            return json.dumps(inference_configuration)
        if inference_configuration := output_params.get("inferenceConfiguration"):
            return json.dumps(inference_configuration)
        return None

    @classmethod
    def _get_token_counts(cls, output_params: dict[str, Any]) -> TokenCount | None:
        """
        Get token counts from output parameters.

        Args:
            output_params (dict[str, Any]): The output parameters.

        Returns:
            TokenCount | None: A TokenCount object if token counts are found, None otherwise.
        """
        if not output_params.get("metadata", {}):
            return None
        if usage := output_params.get("metadata", {}).get("usage"):
            completion, prompt, total = 0, 0, 0

            if input_tokens := usage.get("inputTokens"):
                prompt = input_tokens
            if output_tokens := usage.get("outputTokens"):
                completion = output_tokens
            if (input_tokens := usage.get("inputTokens")) and (
                output_tokens := usage.get("outputTokens")
            ):
                total = input_tokens + output_tokens
            return TokenCount(prompt=prompt, completion=completion, total=total)
        return None

    @classmethod
    def _get_output_value(cls, output_params: dict[str, Any]) -> str | None:
        """
        Get the output value from output parameters.

        Args:
            output_params (dict[str, Any]): The output parameters.

        Returns:
            str | None: The output value if found, None otherwise.
        """
        if raw_response := output_params.get("rawResponse"):
            if output_text := raw_response.get("content"):
                return str(output_text)

        parsed_response = output_params.get("parsedResponse", {})
        if output_text := parsed_response.get("text"):
            # This block will be executed for Post Processing trace
            return str(output_text)
        if output_text := parsed_response.get("rationale"):
            # This block will be executed for Pre Processing trace
            return str(output_text)
        return None

    def _create_model_invocation_span(self, trace_data: dict[str, Any], trace_event: str) -> None:
        """
        Create a model invocation span.

        Args:
            trace_data (dict[str, Any]): The trace data containing information about the
            model invocation.
            trace_event (str): The type of trace event.
        """
        if "modelInvocationOutput" not in trace_data:
            return
        parent_trace = self._initialize_chain_span(trace_event)
        with self.tracer.start_as_current_span(
            name="LLM", context=trace_api.set_span_in_context(parent_trace)
        ) as model_invocation_span:
            trace_values = self.trace_values.get(trace_event, {})

            input_parameters = trace_values.get("modelInvocationInput") or {}
            output_parameters = trace_values.get("modelInvocationOutput") or {}
            model_invocation_input_parameters = input_parameters.get("modelInvocationInput") or {}
            model_invocation_output_parameters = (
                output_parameters.get("modelInvocationOutput") or {}
            )

            input_text = model_invocation_input_parameters.get("text", "")
            request_attributes = get_llm_attributes(
                provider=trace_data.get("provider"),
                model_name=self._get_model_name(
                    model_invocation_input_parameters, model_invocation_output_parameters
                ),
                invocation_parameters=self._get_invocation_parameters(
                    model_invocation_input_parameters, model_invocation_output_parameters
                ),
                token_count=self._get_token_counts(model_invocation_output_parameters),
                input_messages=self._get_messages_object(input_text),
                output_messages=self._get_output_messages(model_invocation_output_parameters),
            )
            model_invocation_span.set_attributes(
                get_span_kind_attributes(OpenInferenceSpanKindValues.LLM)
            )
            model_invocation_span.set_attributes(request_attributes)
            model_invocation_span.set_attributes(get_input_attributes(input_text))

            if output_value := self._get_output_value(model_invocation_output_parameters):
                model_invocation_span.set_attributes(get_output_attributes(output_value))
            model_invocation_span.set_status(Status(StatusCode.OK))
            self._add_model_invocation_attributes_to_parent_span(
                trace_event, "modelInvocationInput"
            )
            self._set_parent_trace_output(trace_event, "modelInvocationOutput")
            self.trace_values[trace_event] = dict()

    def _create_invocation_span(self, trace_data: dict[str, Any], trace_event: str) -> None:
        """
        Create an invocation span.

        Args:
            trace_data (dict[str, Any]): The trace data containing information about the invocation.
            trace_event (str): The type of trace event.
        """
        orchestration_trace_values = self.trace_values[trace_event]
        if "observation" not in trace_data or not orchestration_trace_values.get("invocationInput"):
            return
        invocation_input_trace = orchestration_trace_values["invocationInput"].get(
            "invocationInput"
        )
        if not invocation_input_trace or not invocation_input_trace.get("invocationType"):
            return

        with self.tracer.start_as_current_span(
            name=invocation_input_trace.get("invocationType", "").lower(),
            context=trace_api.set_span_in_context(self._initialize_chain_span(trace_event)),
        ) as invocation_span:
            request_attributes = self._get_attributes_from_invocation_input(
                orchestration_trace_values["invocationInput"]
            )
            if request_attributes:
                invocation_span.set_attributes(request_attributes)
            response_attributes = self._get_attributes_from_observation(
                orchestration_trace_values["observation"]
            )
            invocation_span.set_attributes(response_attributes)
            self._add_invocation_attributes_to_parent_span(
                trace_event, orchestration_trace_values["invocationInput"]
            )
            invocation_span.set_status(Status(StatusCode.OK))

    def _get_attributes_from_orchestration_trace(
        self, trace_data: dict[str, Any], trace_event: str
    ) -> None:
        """
        Process orchestration trace events.

        Args:
            trace_data (dict[str, Any]): The trace data containing information about the event.
            trace_event (str): The type of trace event.
        """
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
            orchestration_span.set_attributes(get_output_attributes(final_response.get("text", "")))
            orchestration_span.set_status(Status(StatusCode.OK))
            self.trace_values[trace_event] = {}

    def _initialize_chain_span(self, trace_type: str) -> Span:
        """
        Initialize or retrieve a chain span for the given trace type.

        Args:
            trace_type (str): The type of trace for which the chain span is being initialized.

        Returns:
            Span: The initialized or retrieved chain span.
        """
        if trace_type not in self.chain_spans:
            self.chain_spans[trace_type] = self.tracer.start_span(
                name=trace_type,
                context=trace_api.set_span_in_context(self._span),
                attributes=get_span_kind_attributes(OpenInferenceSpanKindValues.CHAIN),
            )
        return self.chain_spans[trace_type]

    def _process_trace_event(self, trace_data: dict[str, Any]) -> None:
        """
        Process a trace event and delegate it to the appropriate handler.

        Args:
            trace_data (dict[str, Any]): The trace data containing information about the event.
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
        """
        Finish tracing by ending all spans.
        """
        if self._is_finished:
            return
        self._finish_chain_spans()
        _finish(self._span, None, self._request_parameters)
        self._is_finished = True

    def _finish_chain_spans(self) -> None:
        """
        Finish all chain spans.
        """
        for chain_span in self.chain_spans.values():
            chain_span.set_status(Status(StatusCode.OK))
            chain_span.end()


# Constants
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_SCORE = DocumentAttributes.DOCUMENT_SCORE
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
