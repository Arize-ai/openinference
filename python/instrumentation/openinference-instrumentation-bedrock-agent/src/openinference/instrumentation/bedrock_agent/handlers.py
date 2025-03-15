import json
from datetime import datetime
import logging
from openinference.instrumentation.bedrock_agent.utils import (
    ActionGroupTiming,
    enhance_span_attributes,
    safe_span_operation,
    set_common_attributes
)
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    MessageAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def handle_user_input_span(obs: dict, current_span: trace.Span, message_callback) -> None:
    """Handle user input as a tool invocation"""
    # Key observation: the original implementation creates a dedicated span
    with message_callback.timing_metrics.measure("user_input"):
        with message_callback.tracer.start_as_current_span(
                name="askuser",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                    SpanAttributes.TOOL_NAME: "user::askuser",
                    SpanAttributes.TOOL_DESCRIPTION: "Asks a question to the user",
                    SpanAttributes.TOOL_PARAMETERS: json.dumps({
                        "question": {
                            "type": "string",
                            "description": "Question to ask the user"
                        }
                    }),
                    ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON: json.dumps({
                        "question": obs.get('finalResponse', {}).get('text', '')
                    }),
                    SpanAttributes.INPUT_VALUE: obs.get('finalResponse', {}).get('text', ''),
                    SpanAttributes.METADATA: json.dumps({
                        "tool_type": "user_interaction",
                        "invocation_type": "ASK_USER",
                        "timestamp": datetime.now().isoformat(),
                        "trace_id": obs.get('traceId', '')
                    })
                },
                context=trace.set_span_in_context(current_span)
        ) as user_input_span:
            enhance_span_attributes(user_input_span, obs)
            user_input_span.set_status(Status(StatusCode.OK))


def handle_file_operations(event: dict, current_span, tracer):
    """Handles file operations in the trace"""
    if "files" in event:
        with tracer.start_as_current_span(
                name="file_processing",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: "file_operation",
                    "file.count": len(event["files"]["files"]),
                    "file.types": json.dumps([f.get("type", "unknown") for f in event["files"]["files"]])
                }
        ) as file_span:
            files_event = event["files"]
            files_list = files_event["files"]

            for idx, this_file in enumerate(files_list):
                file_span.set_attribute(f"file.{idx}.name", this_file.get("name", ""))
                file_span.set_attribute(f"file.{idx}.type", this_file.get("type", ""))
                file_span.set_attribute(f"file.{idx}.size", this_file.get("size", 0))

                if "metadata" in this_file:
                    file_span.set_attribute(
                        f"file.{idx}.metadata",
                        json.dumps(this_file["metadata"])
                    )
            file_span.set_status(Status(StatusCode.OK))


def handle_code_interpreter_output(obs, current_trace_data, message_callback):
    """Handle code interpreter output processing"""
    with safe_span_operation():
        with message_callback.timing_metrics.measure("code_interpreter_output"):
            code_output = obs['codeInterpreterInvocationOutput']
            code_span = current_trace_data['code_span']

            if code_span:
                execution_output = code_output.get('executionOutput', '')
                execution_status = code_output.get('executionStatus', '')
                error_message = code_output.get('errorMessage', '')

                output_value = {
                    "execution_output": execution_output,
                    "status": execution_status,
                    "error": error_message if error_message else None
                }

                result_span = message_callback.tracer.start_span(
                    name="code_interpreter_result",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                        SpanAttributes.OUTPUT_VALUE: json.dumps(output_value),
                        SpanAttributes.METADATA: json.dumps({
                            "execution_status": execution_status,
                            "error_message": error_message,
                            "output_type": "code_execution_result",
                            "execution_time": code_output.get('executionTime', 0)
                        })
                    },
                    context=trace.set_span_in_context(code_span)
                )
                enhance_span_attributes(result_span, code_output)
                # result_span.start()
                # Set status based on execution status
                if execution_status == 'FAILED':
                    result_span.set_status(Status(StatusCode.ERROR))
                    result_span.set_attribute("error.message", error_message)
                else:
                    result_span.set_status(Status(StatusCode.OK))
                result_span.end()

                code_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(output_value))

                if execution_status == 'FAILED':
                    code_span.set_status(Status(StatusCode.ERROR))
                    code_span.set_attribute("error.message", error_message)
                else:
                    code_span.set_status(Status(StatusCode.OK))

                code_span.end()
                current_trace_data['code_span'] = None


def handle_knowledge_base_output(obs, current_trace_data, message_callback):
    """Handle knowledge base output processing"""
    with safe_span_operation():
        with message_callback.timing_metrics.measure("knowledge_base_output"):
            kb_output = obs['knowledgeBaseLookupOutput']
            kb_span = current_trace_data['kb_span']
            if kb_span:
                retrieved_refs = kb_output.get('retrievedReferences', [])

                # Process each retrieved document
                for i, ref in enumerate(retrieved_refs):
                    kb_span.set_attribute(
                        f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_ID}",
                        ref.get('metadata', {}).get('x-amz-bedrock-kb-chunk-id', '')
                    )
                    kb_span.set_attribute(
                        f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}",
                        ref.get('content', {}).get('text', '')
                    )
                    kb_span.set_attribute(
                        f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_SCORE}",
                        ref.get('score', 0.0)
                    )
                    kb_span.set_attribute(
                        f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_METADATA}",
                        json.dumps({
                            "data_source_id": ref.get('metadata', {}).get('x-amz-bedrock-kb-data-source-id', ''),
                            "location": ref.get('location', {}),
                            "chunk_size": ref.get('metadata', {}).get('chunk_size', 0),
                            "file_type": ref.get('metadata', {}).get('file_type', '')
                        })
                    )

                kb_result_span = message_callback.tracer.start_span(
                    name="knowledge_base_result",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                        SpanAttributes.OUTPUT_VALUE: json.dumps(retrieved_refs),
                        SpanAttributes.LLM_OUTPUT_MESSAGES: json.dumps(retrieved_refs),

                        SpanAttributes.METADATA: json.dumps({
                            "num_results": len(retrieved_refs),
                            "data_sources": list(set(
                                ref.get('metadata', {}).get('x-amz-bedrock-kb-data-source-id', '')
                                for ref in retrieved_refs
                            )),
                            "total_tokens": kb_output.get('totalTokens', 0)
                        })
                    },
                    context=trace.set_span_in_context(kb_span)
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
                current_trace_data['kb_span'] = None


def handle_action_group_output(obs, current_trace_data, tracer):
    """Handle action group output processing with timing"""
    with safe_span_operation():
        tool_output = obs['actionGroupInvocationOutput']
        tool_span = current_trace_data.get('tool_span')
        action_group_timing = current_trace_data.get('action_group_timing')

        if tool_span and action_group_timing:
            # Record timing for this event
            action_group_timing.record_event()

            # Add timing information to the span
            total_duration = action_group_timing.get_total_duration()
            tool_span.set_attribute("duration_ms", total_duration * 1000)  # Convert to milliseconds

            # Add detailed timing information
            timing_details = {
                "events": action_group_timing.timings,
                "total_duration": total_duration,
                "start_time": action_group_timing.start_time,
                "end_time": action_group_timing.last_event_time
            }

            result_span = tracer.start_span(
                name="tool_result",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                    SpanAttributes.OUTPUT_VALUE: json.dumps(tool_output),
                    SpanAttributes.METADATA: json.dumps({
                        "result_type": "tool_execution_result",
                        "status": tool_output.get('status', 'SUCCESS'),
                        "timing_details": timing_details
                    })
                },
                context=trace.set_span_in_context(tool_span)
            )

            enhance_span_attributes(result_span, tool_output)
            # result_span.start()
            # Set status based on tool output status
            if tool_output.get('status') == 'FAILED':
                result_span.set_status(Status(StatusCode.ERROR))
                result_span.set_attribute("error.message", tool_output.get('error', ''))
            else:
                result_span.set_status(Status(StatusCode.OK))
            result_span.end()

            tool_span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(tool_output))

            if tool_output.get('status') == 'FAILED':
                tool_span.set_status(Status(StatusCode.ERROR))
                tool_span.set_attribute("error.message", tool_output.get('error', ''))
            else:
                tool_span.set_status(Status(StatusCode.OK))

            tool_span.end()

            # Clean up stored data
            current_trace_data['tool_span'] = None
            current_trace_data.pop('action_group_timing', None)


def initialize_process_trace_span(message_callback, parent_span, trace_type, trace_info):
    trace_processing_data = message_callback.trace_processing_data.get(trace_type, {})
    trace_context_key = f"{trace_type}_{trace_info['trace_id']}"
    if trace_info['trace_id'] != trace_processing_data.get('trace_id'):
        # End previous span if it exists
        if trace_processing_data.get('span'):
            # Before ending, set status to OK if not already set
            if trace_processing_data['span'].status.status_code == StatusCode.UNSET:
                trace_processing_data['span'].set_status(Status(StatusCode.OK))
            trace_processing_data['span'].end()

        # Create new post-processing span
        post_processing_span = message_callback.tracer.start_span(
            name=trace_type,
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                SpanAttributes.METADATA: json.dumps({
                    "trace_id": trace_info['trace_id'],
                    "type": trace_info['type']
                })
            },
            context=trace.set_span_in_context(parent_span)
        )
        enhance_span_attributes(post_processing_span, trace_info)
        # post_processing_span.start()

        trace_processing_data = {
            'span': post_processing_span,
            'trace_id': trace_info['trace_id']
        }
        message_callback.trace_processing_data[trace_type] = trace_processing_data
        # Initialize trace storage
        message_callback.trace_context.set(trace_context_key, {
            'llm_input': None,
            'parsed_response': None
        })


def handle_model_invocation_input(message_callback, trace_data, current_trace_data, trace_context_key):
    """Handle model invocation input processing"""
    if 'modelInvocationInput' not in trace_data:
        return
    with message_callback.timing_metrics.measure("model_invocation_input"):
        current_trace_data['llm_input'] = trace_data['modelInvocationInput']
        # No span is created here, so no status to set
        message_callback.trace_context.set(trace_context_key, current_trace_data)


def set_llm_input_values(llm_span, input_text):
    if not input_text:
        return
    idx = 0
    try:
        data = json.loads(input_text)

        if system_message := data.get("system"):
            prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}"
            llm_span.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", system_message)
            llm_span.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", 'system')
            idx += 1
        for message in data.get("messages"):
            prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{idx}"
            llm_span.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", message.get("content") or "")
            llm_span.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", message.get("role"))
            idx += 1
    except Exception as e:
        print(str(e))
    return idx


def set_llm_output_values(llm_span, output_text, input_idx):
    if not output_text:
        return
    try:
        data = json.loads(output_text.get('content'))
        llm_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, data.get('model'))
        idxx = 0
        for idx, content in enumerate(data.get('content') or []):
            prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idxx}"
            if content.get('type') == 'text':
                llm_span.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", content.get('text'))
                llm_span.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", data.get('role'))
                idxx += 1
            if content.get('type') == 'tool_use':
                tool_prefix = f"{prefix}.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
                llm_span.set_attribute(f"{tool_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                                       content.get('name'))
                llm_span.set_attribute(f"{tool_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                                       json.dumps(content.get("input")))
                llm_span.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", data.get('role'))
                idxx += 1
    except Exception as e:
        print(str(e))


def handle_model_invocation_output(
        message_callback, trace_data, current_trace_data, current_span,
        trace_context_key, trace_type
):
    """Handle model invocation output processing"""
    if 'modelInvocationOutput' not in trace_data:
        return
    model_output = trace_data['modelInvocationOutput']
    with safe_span_operation():
        with message_callback.tracer.start_as_current_span(
                name="llm",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                    SpanAttributes.LLM_PROVIDER: "aws",
                    SpanAttributes.LLM_SYSTEM: "bedrock",
                    SpanAttributes.INPUT_VALUE: current_trace_data['llm_input'].get('text', ''),
                    SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(
                        current_trace_data['llm_input'].get('inferenceConfiguration', {})
                    )
                },
                context=trace.set_span_in_context(current_span)
        ) as llm_span:
            enhance_span_attributes(llm_span, model_output)

            if 'metadata' in model_output and 'usage' in model_output['metadata']:
                usage = model_output['metadata']['usage']
                set_common_attributes(llm_span, {
                    SpanAttributes.LLM_TOKEN_COUNT_PROMPT: usage['inputTokens'],
                    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: usage['outputTokens'],
                    SpanAttributes.LLM_TOKEN_COUNT_TOTAL: usage['inputTokens'] + usage['outputTokens']
                })

            if 'rawResponse' in model_output:
                raw_content = model_output['rawResponse'].get('content', '')
                llm_span.set_attribute(SpanAttributes.OUTPUT_VALUE, raw_content)
            input_idx = set_llm_input_values(llm_span, current_trace_data['llm_input'].get('text', ''))
            set_llm_output_values(llm_span, model_output['rawResponse'], input_idx)
            # Create reasoning span as child of LLM span if rationale exists
            handle_rationale(model_output, llm_span, message_callback)

            # Set LLM span status to OK
            llm_span.set_status(Status(StatusCode.OK))
            current_trace_data['llm_input'] = None

        # If there's a parsed response, create a final_response span
        if 'parsedResponse' in model_output:
            parsed_text = model_output['parsedResponse'].get('text', '')
            current_span.set_attribute(SpanAttributes.OUTPUT_VALUE, parsed_text)

            with message_callback.tracer.start_as_current_span(
                    name="final_response",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                        SpanAttributes.INPUT_VALUE: model_output.get('rawResponse', {}).get('content', ''),
                        SpanAttributes.OUTPUT_VALUE: parsed_text,
                    },
                    context=trace.set_span_in_context(current_span)
            ) as final_response_span:
                final_response_span.set_status(Status(StatusCode.OK))

        if trace_type in ['preProcessingTrace', 'postProcessingTrace']:
            # End the pre & post processing span and reset tracking
            current_span.set_status(Status(StatusCode.OK))
            current_span.end()

            # Clean up
            message_callback.trace_context.delete(trace_context_key)
            message_callback.trace_processing_data[trace_type] = {
                'span': None,
                'trace_id': None
            }


def handle_rationale(orch_trace, current_span, message_callback):
    """Handle rationale processing"""
    if 'rationale' in orch_trace:
        with safe_span_operation():
            with message_callback.timing_metrics.measure("rationale"):
                rational_span = message_callback.tracer.start_span(
                    name="rational",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                        SpanAttributes.INPUT_VALUE: orch_trace['rationale'].get('text', ''),
                        SpanAttributes.OUTPUT_VALUE: orch_trace['rationale'].get('text', '')
                    },
                    context=trace.set_span_in_context(current_span)
                )
                enhance_span_attributes(rational_span, orch_trace['rationale'])
                # rational_span.start()
                rational_span.set_status(Status(StatusCode.OK))
                rational_span.end()


def process_knowledge_base_span(kb_input, current_span, message_callback):
    """Enhanced knowledge base span with retriever attributes"""
    with safe_span_operation():
        with message_callback.timing_metrics.measure("knowledge_base"):
            kb_span = message_callback.tracer.start_span(
                name="knowledge_base_lookup",
                attributes={
                    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                    SpanAttributes.INPUT_VALUE: kb_input.get('text', ''),
                    SpanAttributes.METADATA: json.dumps({
                        "knowledge_base_id": kb_input.get('knowledgeBaseId', ''),
                        "invocation_type": "SEARCH",
                        "retrieval_type": "semantic",
                        "data_source": kb_input.get('dataSource', ''),
                        "filter_criteria": kb_input.get('filters', {})
                    })
                },
                context=trace.set_span_in_context(current_span)
            )
            enhance_span_attributes(kb_span, kb_input)
            return kb_span


def process_code_interpreter_span(code_input, current_span, message_callback):
    """Enhanced code interpreter span with tool attributes"""
    with safe_span_operation():
        with message_callback.timing_metrics.measure("code_interpreter"):
            with message_callback.tracer.start_as_current_span(
                    name="code_interpreter",
                    attributes={
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                        SpanAttributes.TOOL_NAME: "code_interpreter",
                        SpanAttributes.TOOL_DESCRIPTION: "Executes Python code and returns results",
                        SpanAttributes.TOOL_PARAMETERS: json.dumps({
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            },
                            "purpose": {
                                "type": "string",
                                "description": "Purpose of code execution"
                            }
                        }),
                        ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON: json.dumps({
                            "code": code_input.get('code', ''),
                            "purpose": code_input.get('purpose', ''),
                            "language": "python"
                        }),
                        SpanAttributes.INPUT_VALUE: code_input.get('code', ''),
                        SpanAttributes.METADATA: json.dumps({
                            "invocation_type": "code_execution",
                            "code_type": "python",
                            "execution_context": code_input.get('context', {}),
                            "tool_version": "1.0"
                        })
                    },
                    context=trace.set_span_in_context(current_span)
            ) as code_span:
                enhance_span_attributes(code_span, code_input)
                return code_span


def handle_invocation_input(orch_trace, current_trace_data, current_span, message_callback):
    """Handle different types of invocation inputs without redundant spans"""
    if 'invocationInput' in orch_trace:
        inv_input = orch_trace['invocationInput']

        # Process tools directly under the parent orchestration span
        if 'codeInterpreterInvocationInput' in inv_input:
            code_input = inv_input['codeInterpreterInvocationInput']
            code_span = process_code_interpreter_span(code_input, current_span, message_callback)
            # code_span.start()
            current_trace_data['code_span'] = code_span

        elif 'knowledgeBaseLookupInput' in inv_input:
            kb_input = inv_input['knowledgeBaseLookupInput']
            kb_span = process_knowledge_base_span(kb_input, current_span, message_callback)
            # kb_span.start()
            current_trace_data['kb_span'] = kb_span

        elif 'actionGroupInvocationInput' in inv_input:
            handle_action_group_input(inv_input, current_trace_data, current_span, message_callback.tracer)


def handle_action_group_input(inv_input, current_trace_data, current_span, tracer):
    """Handle action group invocation input with timing"""
    with safe_span_operation():
        action_input = inv_input['actionGroupInvocationInput']

        # Initialize timing tracker
        action_group_timing = ActionGroupTiming()
        # action_group_timing.start()

        # Store timing tracker in trace data
        current_trace_data['action_group_timing'] = action_group_timing

        tool_span = tracer.start_span(
            name="tool_execution",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                SpanAttributes.TOOL_NAME: action_input.get('function', ''),
                SpanAttributes.TOOL_DESCRIPTION: action_input.get('description', ''),
                SpanAttributes.TOOL_PARAMETERS: json.dumps(action_input.get('parameters', [])),
                ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON: json.dumps({
                    'name': action_input.get('function', ''),
                    'arguments': action_input.get('parameters', {})
                }),
                SpanAttributes.METADATA: json.dumps({
                    "action_group": action_input.get('actionGroupName', ''),
                    "execution_type": action_input.get('executionType', ''),
                    "invocation_type": inv_input.get('invocationType', ''),
                    "tool_version": action_input.get('version', '1.0'),
                    "start_time": action_group_timing.start_time
                })
            },
            context=trace.set_span_in_context(current_span)
        )

        enhance_span_attributes(tool_span, action_input)
        # tool_span.start()
        current_trace_data['tool_span'] = tool_span


def handle_observation(orch_trace, current_trace_data, current_span, message_callback):
    """Handle observation processing with user input support"""
    if 'observation' in orch_trace:
        with message_callback.timing_metrics.measure("observation"):
            obs = orch_trace['observation']

            # Handle different types of observations
            if obs.get('type') == 'ASK_USER':
                handle_user_input_span(obs, current_span, message_callback)

            elif 'codeInterpreterInvocationOutput' in obs and 'code_span' in current_trace_data:
                handle_code_interpreter_output(obs, current_trace_data, message_callback)

            elif 'knowledgeBaseLookupOutput' in obs and 'kb_span' in current_trace_data:
                handle_knowledge_base_output(obs, current_trace_data, message_callback)

            elif 'actionGroupInvocationOutput' in obs and 'tool_span' in current_trace_data:
                handle_action_group_output(obs, current_trace_data, message_callback.tracer)

            # Process final response if present
            return handle_final_response(obs, current_span, message_callback)

    return False


def handle_final_response(obs, current_span, message_callback):
    """Handle final response processing"""
    if 'finalResponse' in obs:
        with message_callback.timing_metrics.measure("final_response"):
            final_response = obs['finalResponse'].get('text', '')
            if current_span:
                current_span.set_attribute(SpanAttributes.OUTPUT_VALUE, final_response)
                enhance_span_attributes(current_span, obs['finalResponse'])
                current_span.set_status(Status(StatusCode.OK))
                current_span.end()
                return True
    return False
