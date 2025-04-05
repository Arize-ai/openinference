import json
import logging
from datetime import datetime
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from openinference.instrumentation.bedrock.utils import (
    enhance_span_attributes,
    safe_span_operation,
    set_common_attributes,
)
from openinference.semconv.trace import (
    DocumentAttributes,
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_user_input_span(obs: dict[str, Any], current_span: Span, message_callback: Any) -> None:
    """Handle user input as a tool invocation"""
    # Key observation: the original implementation creates a dedicated span
    with message_callback.tracer.start_as_current_span(
        name="askuser",
        attributes={
            OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
            TOOL_NAME: "user::askuser",
            TOOL_DESCRIPTION: "Asks a question to the user",
            TOOL_PARAMETERS: json.dumps(
                {"question": {"type": "string", "description": "Question to ask the user"}}
            ),
            ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON: json.dumps(
                {"question": obs.get("finalResponse", {}).get("text", "")}
            ),
            SpanAttributes.INPUT_VALUE: obs.get("finalResponse", {}).get("text", ""),
            SpanAttributes.METADATA: json.dumps(
                {
                    "tool_type": "user_interaction",
                    "invocation_type": "ASK_USER",
                    "timestamp": datetime.now().isoformat(),
                    "trace_id": obs.get("traceId", ""),
                }
            ),
        },
        context=trace.set_span_in_context(current_span),
    ) as user_input_span:
        enhance_span_attributes(user_input_span, obs)
        user_input_span.set_status(Status(StatusCode.OK))


def handle_file_operations(event: dict[str, Any], current_span: Span, tracer: Tracer) -> None:
    """Handles file operations in the trace"""
    if "files" in event:
        with tracer.start_as_current_span(
            name="file_processing",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: "file_operation",
                "file.count": len(event["files"]["files"]),
                "file.types": json.dumps(
                    [f.get("type", "unknown") for f in event["files"]["files"]]
                ),
            },
        ) as file_span:
            files_event = event["files"]
            files_list = files_event["files"]

            for idx, this_file in enumerate(files_list):
                file_span.set_attribute(f"file.{idx}.name", this_file.get("name", ""))
                file_span.set_attribute(f"file.{idx}.type", this_file.get("type", ""))
                file_span.set_attribute(f"file.{idx}.size", this_file.get("size", 0))

                if "metadata" in this_file:
                    file_span.set_attribute(
                        f"file.{idx}.metadata", json.dumps(this_file["metadata"])
                    )
            file_span.set_status(Status(StatusCode.OK))


def handle_code_interpreter_output(
    obs: dict[str, Any], current_trace_data: dict[str, Any], message_callback: Any
) -> None:
    """Handle code interpreter output processing"""
    with safe_span_operation():
        code_output = obs["codeInterpreterInvocationOutput"]
        code_span = current_trace_data["code_span"]

        if code_span:
            execution_output = code_output.get("executionOutput", "")
            execution_status = code_output.get("executionStatus", "")
            error_message = code_output.get("errorMessage", "")

            output_value = {
                "execution_output": execution_output,
                "status": execution_status,
                "error": error_message if error_message else None,
            }

            result_span = message_callback.tracer.start_span(
                name="code_interpreter_result",
                attributes={
                    OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                    OUTPUT_VALUE: json.dumps(output_value),
                    METADATA: json.dumps(
                        {
                            "execution_status": execution_status,
                            "error_message": error_message,
                            "output_type": "code_execution_result",
                            "execution_time": code_output.get("executionTime", 0),
                        }
                    ),
                },
                context=trace.set_span_in_context(code_span),
            )
            enhance_span_attributes(result_span, code_output)
            # result_span.start()
            # Set status based on execution status
            if execution_status == "FAILED":
                result_span.set_status(Status(StatusCode.ERROR))
                result_span.set_attribute("error.message", error_message)
            else:
                result_span.set_status(Status(StatusCode.OK))
            result_span.end()

            code_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(output_value))

            if execution_status == "FAILED":
                code_span.set_status(Status(StatusCode.ERROR))
                code_span.set_attribute("error.message", error_message)
            else:
                code_span.set_status(Status(StatusCode.OK))

            code_span.end()
            current_trace_data["code_span"] = None


def handle_knowledge_base_output(
    obs: dict[str, Any], current_trace_data: dict[str, Any], message_callback: Any
) -> None:
    """Handle knowledge base output processing"""
    with safe_span_operation():
        kb_output = obs["knowledgeBaseLookupOutput"]
        kb_span = current_trace_data["kb_span"]
        if kb_span:
            retrieved_refs = kb_output.get("retrievedReferences", [])

            # Process each retrieved document
            for i, ref in enumerate(retrieved_refs):
                kb_span.set_attribute(
                    f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_ID}",
                    ref.get("metadata", {}).get("x-amz-bedrock-kb-chunk-id", ""),
                )
                kb_span.set_attribute(
                    f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}",
                    ref.get("content", {}).get("text", ""),
                )
                kb_span.set_attribute(
                    f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_SCORE}",
                    ref.get("score", 0.0),
                )
                kb_span.set_attribute(
                    f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_METADATA}",
                    json.dumps(
                        {
                            "data_source_id": ref.get("metadata", {}).get(
                                "x-amz-bedrock-kb-data-source-id", ""
                            ),
                            "location": ref.get("location", {}),
                            "chunk_size": ref.get("metadata", {}).get("chunk_size", 0),
                            "file_type": ref.get("metadata", {}).get("file_type", ""),
                        }
                    ),
                )

            kb_result_span = message_callback.tracer.start_span(
                name="knowledge_base_result",
                attributes={
                    OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                    OUTPUT_VALUE: json.dumps(retrieved_refs),
                    LLM_OUTPUT_MESSAGES: json.dumps(retrieved_refs),
                    METADATA: json.dumps(
                        {
                            "num_results": len(retrieved_refs),
                            "data_sources": list(
                                set(
                                    ref.get("metadata", {}).get(
                                        "x-amz-bedrock-kb-data-source-id", ""
                                    )
                                    for ref in retrieved_refs
                                )
                            ),
                            "total_tokens": kb_output.get("totalTokens", 0),
                        }
                    ),
                },
                context=trace.set_span_in_context(kb_span),
            )
            enhance_span_attributes(kb_result_span, kb_output)
            # kb_result_span.start()
            # Set status to OK
            kb_result_span.set_status(Status(StatusCode.OK))
            kb_result_span.end()

            # Set status based on whether we got results or not
            if retrieved_refs:
                kb_span.set_status(Status(StatusCode.OK))
            else:
                kb_span.set_status(Status(StatusCode.OK))
                kb_span.set_attribute("retrieval.no_results", True)

            kb_span.end()
            current_trace_data["kb_span"] = None


def handle_action_group_output(
    obs: dict[str, Any], current_trace_data: dict[str, Any], tracer: Tracer
) -> None:
    """Handle action group output processing with timing"""
    with safe_span_operation():
        tool_output = obs["actionGroupInvocationOutput"]
        tool_span = current_trace_data.get("tool_span")

        if tool_span:
            result_span = tracer.start_span(
                name="tool_result",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                    SpanAttributes.OUTPUT_VALUE: json.dumps(tool_output),
                    SpanAttributes.METADATA: json.dumps(
                        {
                            "result_type": "tool_execution_result",
                            "status": tool_output.get("status", "SUCCESS"),
                        }
                    ),
                },
                context=trace.set_span_in_context(tool_span),
            )

            enhance_span_attributes(result_span, tool_output)
            # result_span.start()
            # Set status based on tool output status
            if tool_output.get("status") == "FAILED":
                result_span.set_status(Status(StatusCode.ERROR))
                result_span.set_attribute("error.message", tool_output.get("error", ""))
            else:
                result_span.set_status(Status(StatusCode.OK))
            result_span.end()

            tool_span.set_attribute(OUTPUT_VALUE, json.dumps(tool_output))

            if tool_output.get("status") == "FAILED":
                tool_span.set_status(Status(StatusCode.ERROR))
                tool_span.set_attribute("error.message", tool_output.get("error", ""))
            else:
                tool_span.set_status(Status(StatusCode.OK))

            tool_span.end()

            # Clean up stored data
            current_trace_data["tool_span"] = None


def initialize_process_trace_span(
    message_callback: Any, parent_span: Span, trace_type: str, trace_info: dict[str, Any]
) -> None:
    trace_processing_data = message_callback.trace_processing_data.get(trace_type, {})
    trace_context_key = f"{trace_type}_{trace_info['trace_id']}"
    if trace_info["trace_id"] != trace_processing_data.get("trace_id"):
        # End previous span if it exists
        if trace_processing_data.get("span"):
            # Before ending, set status to OK if not already set
            if trace_processing_data["span"].status.status_code == StatusCode.UNSET:
                trace_processing_data["span"].set_status(Status(StatusCode.OK))
            trace_processing_data["span"].end()

        # Create new post-processing span
        post_processing_span = message_callback.tracer.start_span(
            name=trace_type,
            attributes={
                OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                METADATA: json.dumps(
                    {"trace_id": trace_info["trace_id"], "type": trace_info["type"]}
                ),
            },
            context=trace.set_span_in_context(parent_span),
        )
        enhance_span_attributes(post_processing_span, trace_info)
        # post_processing_span.start()

        trace_processing_data = {"span": post_processing_span, "trace_id": trace_info["trace_id"]}
        message_callback.trace_processing_data[trace_type] = trace_processing_data
        # Initialize trace storage
        message_callback.trace_context.set(
            trace_context_key, {"llm_input": None, "parsed_response": None}
        )


def handle_model_invocation_input(
    message_callback: Any,
    trace_data: dict[str, Any],
    current_trace_data: dict[str, Any],
    trace_context_key: str,
) -> None:
    """Handle model invocation input processing"""
    if "modelInvocationInput" not in trace_data:
        return
    current_trace_data["llm_input"] = trace_data["modelInvocationInput"]
    # No span is created here, so no status to set
    message_callback.trace_context.set(trace_context_key, current_trace_data)


def set_llm_input_values(llm_span: Span, input_text: str) -> None:
    if not input_text:
        return
    idx = 0
    try:
        data = json.loads(input_text)
        if system_message := data.get("system"):
            prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}"
            llm_span.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", system_message)
            llm_span.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", "system")
            idx += 1
        for message in data.get("messages"):
            prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}"
            llm_span.set_attribute(
                f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", message.get("content") or ""
            )
            llm_span.set_attribute(
                f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", message.get("role")
            )
            idx += 1
    except Exception as e:
        logger.error(str(e))


def set_llm_output_values(llm_span: Span, output_text: dict[str, Any]) -> None:
    if not output_text:
        return
    try:
        data = json.loads(str(output_text.get("content")))
        llm_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, data.get("model"))
        idx = 0
        for content in data.get("content") or []:
            prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}"
            if content.get("type") == "text":
                llm_span.set_attribute(
                    f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", content.get("text")
                )
                llm_span.set_attribute(
                    f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", data.get("role")
                )
                idx += 1
            if content.get("type") == "tool_use":
                tool_prefix = f"{prefix}.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
                llm_span.set_attribute(
                    f"{tool_prefix}.{TOOL_CALL_FUNCTION_NAME}",
                    content.get("name"),
                )
                llm_span.set_attribute(
                    f"{tool_prefix}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                    json.dumps(content.get("input")),
                )
                llm_span.set_attribute(
                    f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", data.get("role")
                )
                idx += 1
    except Exception as e:
        logger.error(str(e))


def handle_model_invocation_output(
    message_callback: Any,
    trace_data: dict[str, Any],
    current_trace_data: dict[str, Any],
    current_span: Span,
    trace_context_key: str,
    trace_type: str,
) -> None:
    """Handle model invocation output processing"""
    if "modelInvocationOutput" not in trace_data:
        return
    model_output = trace_data["modelInvocationOutput"]
    with safe_span_operation():
        with message_callback.tracer.start_as_current_span(
            name="llm",
            attributes={
                OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                LLM_PROVIDER: "aws",
                LLM_SYSTEM: "bedrock",
                INPUT_VALUE: current_trace_data["llm_input"].get("text", ""),
                LLM_INVOCATION_PARAMETERS: json.dumps(
                    current_trace_data["llm_input"].get("inferenceConfiguration", {})
                ),
            },
            context=trace.set_span_in_context(current_span),
        ) as llm_span:
            enhance_span_attributes(llm_span, model_output)

            if "metadata" in model_output and "usage" in model_output["metadata"]:
                usage = model_output["metadata"]["usage"]
                set_common_attributes(
                    llm_span,
                    {
                        LLM_TOKEN_COUNT_PROMPT: usage["inputTokens"],
                        LLM_TOKEN_COUNT_COMPLETION: usage["outputTokens"],
                        LLM_TOKEN_COUNT_TOTAL: usage["inputTokens"] + usage["outputTokens"],
                    },
                )
            set_llm_input_values(llm_span, current_trace_data["llm_input"].get("text", ""))
            if "rawResponse" in model_output:
                raw_content = model_output["rawResponse"].get("content", "")
                llm_span.set_attribute(SpanAttributes.OUTPUT_VALUE, raw_content)
                set_llm_output_values(llm_span, model_output["rawResponse"])

            # Create reasoning span as child of LLM span if rationale exists
            handle_rationale(model_output, llm_span, message_callback)

            # Set LLM span status to OK
            llm_span.set_status(Status(StatusCode.OK))
            current_trace_data["llm_input"] = None

        # If there's a parsed response, create a final_response span
        if "parsedResponse" in model_output:
            parsed_text = model_output["parsedResponse"].get("text", "")
            current_span.set_attribute(SpanAttributes.OUTPUT_VALUE, parsed_text)

            with message_callback.tracer.start_as_current_span(
                name="final_response",
                attributes={
                    OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    INPUT_VALUE: model_output.get("rawResponse", {}).get("content", ""),
                    OUTPUT_VALUE: parsed_text,
                },
                context=trace.set_span_in_context(current_span),
            ) as final_response_span:
                final_response_span.set_status(Status(StatusCode.OK))

        if trace_type in ["preProcessingTrace", "postProcessingTrace"]:
            # End the pre & post processing span and reset tracking
            current_span.set_status(Status(StatusCode.OK))
            current_span.end()

            # Clean up
            message_callback.trace_context.delete(trace_context_key)
            message_callback.trace_processing_data[trace_type] = {"span": None, "trace_id": None}


def handle_rationale(orch_trace: dict[str, Any], current_span: Span, message_callback: Any) -> None:
    """Handle rationale processing"""
    if "rationale" in orch_trace:
        with safe_span_operation():
            rational_span = message_callback.tracer.start_span(
                name="rational",
                attributes={
                    OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                    OUTPUT_VALUE: orch_trace["rationale"].get("text", ""),
                },
                context=trace.set_span_in_context(current_span),
            )
            enhance_span_attributes(rational_span, orch_trace["rationale"])
            # rational_span.start()
            rational_span.set_status(Status(StatusCode.OK))
            rational_span.end()


def process_knowledge_base_span(
    kb_input: dict[str, Any], current_span: Span, message_callback: Any
) -> Any:
    """Enhanced knowledge base span with retriever attributes"""
    with safe_span_operation():
        kb_span: Span = message_callback.tracer.start_span(
            name="knowledge_base_lookup",
            attributes={
                OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                INPUT_VALUE: kb_input.get("text", ""),
                METADATA: json.dumps(
                    {
                        "knowledge_base_id": kb_input.get("knowledgeBaseId", ""),
                        "invocation_type": "SEARCH",
                        "retrieval_type": "semantic",
                        "data_source": kb_input.get("dataSource", ""),
                        "filter_criteria": kb_input.get("filters", {}),
                    }
                ),
            },
            context=trace.set_span_in_context(current_span),
        )
        enhance_span_attributes(kb_span, kb_input)
        return kb_span


def process_code_interpreter_span(
    code_input: dict[str, Any], current_span: Span, message_callback: Any
) -> Any:
    """Enhanced code interpreter span with tool attributes"""
    with safe_span_operation():
        with message_callback.tracer.start_as_current_span(
            name="code_interpreter",
            attributes={
                OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                TOOL_NAME: "code_interpreter",
                TOOL_DESCRIPTION: "Executes Python code and returns results",
                TOOL_PARAMETERS: json.dumps(
                    {
                        "code": {"type": "string", "description": "Python code to execute"},
                        "purpose": {
                            "type": "string",
                            "description": "Purpose of code execution",
                        },
                    }
                ),
                TOOL_CALL_FUNCTION_ARGUMENTS_JSON: json.dumps(
                    {
                        "code": code_input.get("code", ""),
                        "purpose": code_input.get("purpose", ""),
                        "language": "python",
                    }
                ),
                SpanAttributes.INPUT_VALUE: code_input.get("code", ""),
                SpanAttributes.METADATA: json.dumps(
                    {
                        "invocation_type": "code_execution",
                        "code_type": "python",
                        "execution_context": code_input.get("context", {}),
                        "tool_version": "1.0",
                    }
                ),
            },
            context=trace.set_span_in_context(current_span),
        ) as code_span:
            enhance_span_attributes(code_span, code_input)
            return code_span


def handle_invocation_input(
    orch_trace: dict[str, Any],
    current_trace_data: dict[str, Any],
    current_span: Span,
    message_callback: Any,
) -> None:
    """Handle different types of invocation inputs without redundant spans"""
    if "invocationInput" in orch_trace:
        inv_input = orch_trace["invocationInput"]

        # Process tools directly under the parent orchestration span
        if "codeInterpreterInvocationInput" in inv_input:
            code_input = inv_input["codeInterpreterInvocationInput"]
            code_span = process_code_interpreter_span(code_input, current_span, message_callback)
            # code_span.start()
            current_trace_data["code_span"] = code_span

        elif "knowledgeBaseLookupInput" in inv_input:
            kb_input = inv_input["knowledgeBaseLookupInput"]
            kb_span = process_knowledge_base_span(kb_input, current_span, message_callback)
            # kb_span.start()
            current_trace_data["kb_span"] = kb_span

        elif "actionGroupInvocationInput" in inv_input:
            handle_action_group_input(
                inv_input, current_trace_data, current_span, message_callback.tracer
            )


def handle_action_group_input(
    inv_input: dict[str, Any],
    current_trace_data: dict[str, Any],
    current_span: Span,
    tracer: Tracer,
) -> None:
    """Handle action group invocation input with timing"""
    with safe_span_operation():
        action_input = inv_input["actionGroupInvocationInput"]

        tool_span = tracer.start_span(
            name="tool_execution",
            attributes={
                OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                TOOL_NAME: action_input.get("function", ""),
                TOOL_DESCRIPTION: action_input.get("description", ""),
                TOOL_PARAMETERS: json.dumps(action_input.get("parameters", [])),
                TOOL_CALL_FUNCTION_ARGUMENTS_JSON: json.dumps(
                    {
                        "name": action_input.get("function", ""),
                        "arguments": action_input.get("parameters", {}),
                    }
                ),
            },
            context=trace.set_span_in_context(current_span),
        )
        enhance_span_attributes(tool_span, action_input)
        current_trace_data["tool_span"] = tool_span


def handle_observation(
    orch_trace: dict[str, Any],
    current_trace_data: dict[str, Any],
    current_span: Span,
    message_callback: Any,
) -> bool:
    """Handle observation processing with user input support"""
    if "observation" in orch_trace:
        obs = orch_trace["observation"]

        # Handle different types of observations
        if obs.get("type") == "ASK_USER":
            handle_user_input_span(obs, current_span, message_callback)

        elif "codeInterpreterInvocationOutput" in obs and "code_span" in current_trace_data:
            handle_code_interpreter_output(obs, current_trace_data, message_callback)

        elif "knowledgeBaseLookupOutput" in obs and "kb_span" in current_trace_data:
            handle_knowledge_base_output(obs, current_trace_data, message_callback)

        elif "actionGroupInvocationOutput" in obs and "tool_span" in current_trace_data:
            handle_action_group_output(obs, current_trace_data, message_callback.tracer)

        # Process final response if present
        return handle_final_response(obs, current_span)

    return False


def handle_final_response(obs: dict[str, Any], current_span: Span) -> bool:
    """Handle final response processing"""
    if "finalResponse" in obs:
        final_response = obs["finalResponse"].get("text", "")
        if current_span:
            current_span.set_attribute(SpanAttributes.OUTPUT_VALUE, final_response)
            enhance_span_attributes(current_span, obs["finalResponse"])
            current_span.set_status(Status(StatusCode.OK))
            current_span.end()
            return True
    return False


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
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
