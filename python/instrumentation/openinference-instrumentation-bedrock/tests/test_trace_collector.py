"""Unit tests for the collect method of the TraceCollector class."""

from datetime import datetime, timezone

from openinference.instrumentation.bedrock._trace_collector import (
    TraceCollector,
    TraceNode,
    TraceSpan,
)


class TestCollectMethod:
    """Tests for the collect method of the TraceCollector class."""

    def test_handle_chunk_for_current_node_with_span(self) -> None:
        """Test handling a chunk for a node that already has a current span."""
        collector = TraceCollector()
        node = TraceNode("test-trace-id")
        span = TraceSpan("modelInvocationInput")
        node.add_span(span)

        trace_data = {"data": "test_data"}
        chunk_type = "modelInvocationOutput"

        # Handle the chunk
        collector._handle_chunk_for_current_node(node, chunk_type, trace_data)

        # Verify the result
        assert node is not None
        assert node.current_span is not None
        assert len(node.current_span.chunks) == 1
        assert node.current_span.chunks[0] == trace_data

    def test_handle_chunk_for_current_node_without_span(self) -> None:
        """Test handling a chunk for a node that doesn't have a current span."""
        collector = TraceCollector()
        node = TraceNode("test-trace-id")

        trace_data = {"data": "test_data"}
        chunk_type = "modelInvocationInput"

        # Handle the chunk
        collector._handle_chunk_for_current_node(node, chunk_type, trace_data)

        # Verify the result
        assert len(node.spans) == 1
        assert isinstance(node.spans[0], TraceSpan)
        assert node.spans[0].span_type == chunk_type
        assert len(node.spans[0].chunks) == 1
        assert node.spans[0].chunks[0] == trace_data
        assert node.current_span == node.spans[0]

    def test_handle_new_trace_node_scenario_standard(self) -> None:
        """Test handling a new trace node scenario with standard trace data."""
        collector = TraceCollector()

        # Create a parent node
        parent_node = TraceNode("parent-trace-id")
        collector.trace_stack.push(parent_node)

        # Create trace data
        trace_data = {
            "orchestrationTrace": {
                "modelInvocationInput": {
                    "foundationModel": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "traceId": "test-trace-id",
                    "type": "ORCHESTRATION",
                }
            }
        }

        # Set up parameters
        node_trace_id = "orchestrationTrace_test-trace-id"
        event_type = "orchestrationTrace"
        chunk_type = "modelInvocationInput"

        # Handle the new trace node scenario
        collector._handle_new_trace_node_scenario(
            parent_node, node_trace_id, event_type, chunk_type, trace_data
        )

        # Verify the result
        assert node_trace_id in collector.trace_nodes
        assert collector.trace_nodes[node_trace_id].node_type == event_type
        assert len(collector.trace_nodes[node_trace_id].spans) == 1
        assert isinstance(collector.trace_nodes[node_trace_id].spans[0], TraceSpan)
        assert len(collector.trace_nodes[node_trace_id].spans[0].chunks) == 1
        assert collector.trace_nodes[node_trace_id].spans[0].chunks[0] == trace_data
        assert collector.trace_stack.head == collector.trace_nodes[node_trace_id]

    def test_collect_orchestration_trace(self) -> None:
        """Test collecting an orchestration trace."""
        collector = TraceCollector()

        # Create a simple trace
        trace = {
            "orchestrationTrace": {
                "modelInvocationInput": {
                    "foundationModel": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "traceId": "test-trace-id-1",
                    "type": "ORCHESTRATION",
                }
            }
        }

        # Collect the trace
        result = collector.collect(trace)

        # Verify the result
        assert result is not None

        # Check that the trace node was created
        orchestration_trace_id = "orchestrationTrace_test-trace-id-1"
        assert orchestration_trace_id in collector.trace_nodes

        # Check that the node has the correct type
        orchestration_node = collector.trace_nodes[orchestration_trace_id]
        assert orchestration_node.node_type == "orchestrationTrace"

        # Check that the node has a span
        assert len(orchestration_node.spans) == 1
        assert isinstance(orchestration_node.spans[0], TraceSpan)
        assert orchestration_node.spans[0].span_type == "modelInvocationInput"

        # Check that the span has a chunk
        assert len(orchestration_node.spans[0].chunks) == 1
        assert orchestration_node.spans[0].chunks[0] == trace

    def test_collect_preprocessing_trace(self) -> None:
        """Test collecting a preprocessing trace."""
        collector = TraceCollector()

        # Create a preprocessing trace
        trace = {
            "preProcessingTrace": {
                "modelInvocationInput": {
                    "foundationModel": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "traceId": "test-trace-id-1",
                    "type": "PRE_PROCESSING",
                }
            }
        }

        # Collect the trace
        result = collector.collect(trace)

        # Verify the result
        assert result is not None

        # Check that the trace node was created
        preprocessing_trace_id = "preProcessingTrace_test-trace-id-1"
        assert preprocessing_trace_id in collector.trace_nodes

        # Check that the node has the correct type
        preprocessing_node = collector.trace_nodes[preprocessing_trace_id]
        assert preprocessing_node.node_type == "preProcessingTrace"

        # Check that the node has a span
        assert len(preprocessing_node.spans) == 1
        assert isinstance(preprocessing_node.spans[0], TraceSpan)
        assert preprocessing_node.spans[0].span_type == "modelInvocationInput"

        # Check that the span has a chunk
        assert len(preprocessing_node.spans[0].chunks) == 1
        assert preprocessing_node.spans[0].chunks[0] == trace

    def test_collect_postprocessing_trace(self) -> None:
        """Test collecting a postprocessing trace."""
        collector = TraceCollector()

        # Create a postprocessing trace
        trace = {
            "postProcessingTrace": {
                "modelInvocationInput": {
                    "foundationModel": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "traceId": "test-trace-id-1",
                    "type": "POST_PROCESSING",
                }
            }
        }

        # Collect the trace
        result = collector.collect(trace)

        # Verify the result
        assert result is not None

        # Check that the trace node was created
        postprocessing_trace_id = "postProcessingTrace_test-trace-id-1"
        assert postprocessing_trace_id in collector.trace_nodes

        # Check that the node has the correct type
        postprocessing_node = collector.trace_nodes[postprocessing_trace_id]
        assert postprocessing_node.node_type == "postProcessingTrace"

        # Check that the node has a span
        assert len(postprocessing_node.spans) == 1
        assert isinstance(postprocessing_node.spans[0], TraceSpan)
        assert postprocessing_node.spans[0].span_type == "modelInvocationInput"

        # Check that the span has a chunk
        assert len(postprocessing_node.spans[0].chunks) == 1
        assert postprocessing_node.spans[0].chunks[0] == trace

    def test_collect_failure_trace(self) -> None:
        """Test collecting a failure trace."""
        collector = TraceCollector()

        # Create a failure trace
        trace = {
            "failureTrace": {
                "failureCode": 429,
                "failureReason": "Too many requests, please wait before trying again.",
                "metadata": {
                    "clientRequestId": "test-request-id",
                    "endTime": datetime(2025, 5, 21, 5, 27, 14, 358936, tzinfo=timezone.utc),
                    "operationTotalTimeMs": 7796,
                    "startTime": datetime(2025, 5, 21, 5, 27, 14, 290997, tzinfo=timezone.utc),
                    "totalTimeMs": 68,
                },
                "traceId": "test-trace-id-1",
            }
        }

        # Collect the trace
        result = collector.collect(trace)

        # Verify the result
        assert result is not None

        # Check that the trace node was created
        failure_trace_id = "failureTrace_test-trace-id-1"
        assert failure_trace_id in collector.trace_nodes

        # Check that the node has the correct type
        failure_node = collector.trace_nodes[failure_trace_id]
        assert failure_node.node_type == "failureTrace"

        # Check that the node has a chunk
        assert len(failure_node.chunks) == 1
        assert failure_node.chunks[0] == trace

    def test_collect_agent_collaborator_traces(self) -> None:
        """Test collecting agent collaborator traces."""
        collector = TraceCollector()

        # Create traces
        trace1 = {
            "orchestrationTrace": {
                "modelInvocationInput": {
                    "foundationModel": "anthropic.claude-3-5-sonnet-20240620-v1:0",
                    "traceId": "test-trace-id-1",
                    "type": "ORCHESTRATION",
                }
            }
        }

        trace2 = {
            "orchestrationTrace": {
                "invocationInput": {
                    "agentCollaboratorInvocationInput": {
                        "agentCollaboratorName": "TestAgent",
                        "input": {"text": "Test input", "type": "TEXT"},
                    },
                    "invocationType": "AGENT_COLLABORATOR",
                    "traceId": "test-trace-id-1",
                }
            }
        }

        trace3 = {
            "orchestrationTrace": {
                "observation": {
                    "agentCollaboratorInvocationOutput": {
                        "agentCollaboratorName": "TestAgent",
                        "metadata": {
                            "clientRequestId": "test-request-id",
                            "endTime": datetime(
                                2025, 5, 20, 18, 31, 34, 895571, tzinfo=timezone.utc
                            ),
                            "startTime": datetime(
                                2025, 5, 20, 18, 31, 31, 453093, tzinfo=timezone.utc
                            ),
                            "totalTimeMs": 3442,
                        },
                        "output": {
                            "text": "Test output",
                            "type": "TEXT",
                        },
                    },
                    "traceId": "test-trace-id-1",
                    "type": "AGENT_COLLABORATOR",
                }
            }
        }

        # Collect the traces
        collector.collect(trace1)
        collector.collect(trace2)
        collector.collect(trace3)

        # Verify the results
        orchestration_trace_id = "orchestrationTrace_test-trace-id-1"
        assert orchestration_trace_id in collector.trace_nodes

        orchestration_node = collector.trace_nodes[orchestration_trace_id]
        assert orchestration_node.node_type == "orchestrationTrace"

        # Find the agent collaborator node
        agent_node = None
        for span in orchestration_node.spans:
            if isinstance(span, TraceNode) and span.node_type == "agent-collaborator":
                agent_node = span
                break

        assert agent_node is not None
        assert agent_node.parent_trace_node == orchestration_node
        assert len(agent_node.chunks) > 0

    def test_handle_parent_node_match_scenario_agent_collaborator_input(self) -> None:
        """Test handling a parent node match scenario with agent collaborator input."""
        collector = TraceCollector()

        # Create a parent node
        parent_node = TraceNode("parent-trace-id")

        # Create trace data for agent collaborator input
        trace_data = {
            "orchestrationTrace": {
                "invocationInput": {
                    "agentCollaboratorInvocationInput": {
                        "agentCollaboratorName": "TestAgent",
                        "input": {"text": "Test input", "type": "TEXT"},
                    },
                    "invocationType": "AGENT_COLLABORATOR",
                    "traceId": "test-trace-id",
                }
            }
        }

        # Set up parameters
        agent_node_trace_id = "agent-trace-id"
        event_type = "orchestrationTrace"
        chunk_type = "invocationInput"

        # Handle the parent node match scenario
        collector._handle_parent_node_match_scenario(
            parent_node, agent_node_trace_id, event_type, chunk_type, trace_data
        )

        # Verify the result
        assert len(parent_node.spans) == 1
        assert isinstance(parent_node.spans[0], TraceNode)
        assert parent_node.spans[0].node_type == "agent-collaborator"
        assert parent_node.spans[0].node_trace_id == agent_node_trace_id
        assert len(parent_node.spans[0].chunks) == 1
        assert parent_node.spans[0].chunks[0] == trace_data
        assert parent_node.spans[0].parent_trace_node == parent_node
        assert collector.trace_stack.head == parent_node.spans[0]
        assert agent_node_trace_id in collector.trace_nodes
        assert agent_node_trace_id in collector.trace_ids
