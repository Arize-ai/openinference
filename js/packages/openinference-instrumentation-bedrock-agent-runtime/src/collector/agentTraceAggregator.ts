import { AgentTraceNode } from "./agentTraceNode";
import { TraceNodeStack } from "./traceNodeStack";
import { AgentChunkSpan } from "./agentChunkSpan";
import {
  extractTraceId,
  getChunkType,
  getEventType,
} from "../attributes/attributeExtractionUtils";
import { generateUniqueTraceId } from "../attributes/attributeUtils";
import { getObjectDataFromUnknown } from "../utils/jsonUtils";
import { StringKeyedObject } from "../types";
import { diag } from "@opentelemetry/api";

/**
 * Aggregates agent trace data into a hierarchical structure of nodes and spans.
 */
export class AgentTraceAggregator {
  /** Root node that serves as the parent for all trace nodes. */
  readonly rootNode: AgentTraceNode;

  private readonly traceStack = new TraceNodeStack();
  private readonly nodesById: Record<string, AgentTraceNode> = {};
  private readonly seenIds: Set<string> = new Set();

  /**
   * Creates a new instance of AgentTraceAggregator with a default root node.
   */
  constructor() {
    this.rootNode = new AgentTraceNode({
      traceId: "default-parent-node",
      eventType: "bedrock_agent.invoke_agent",
    });
    this.traceStack.push(this.rootNode);
    this.nodesById[this.rootNode.nodeTraceId] = this.rootNode;
    this.seenIds.add(this.rootNode.nodeTraceId);
  }

  /**
   * Processes incoming trace data.
   * @param raw - The raw trace data to be processed
   */
  collect(raw: StringKeyedObject) {
    const traceData = this.unwrapTrace(raw);
    if (!traceData) {
      return;
    }
    const eventType = getEventType(traceData);
    const traceId = extractTraceId(traceData);
    if (!traceId || !eventType) {
      return;
    }
    const nodeTraceId = generateUniqueTraceId(eventType, traceId);
    const chunkType = getChunkType(
      getObjectDataFromUnknown({ data: traceData, key: eventType }) ?? {},
    );
    if (!chunkType) {
      return;
    }
    if (!this.traceStack.head) {
      diag.warn("No trace stack head found");
      return;
    }
    const agentChildId = this.detectAgentCollaboration(
      traceData,
      eventType,
      chunkType,
    )
      ? `${nodeTraceId}-agent`
      : null;
    this.routeTraceData({
      traceData,
      parent: this.traceStack.head,
      nodeTraceId,
      agentChildId,
      eventType,
      chunkType,
    });
  }

  /**
   * Extracts trace data from nested Bedrock trace structure.
   * @param obj - The object potentially containing wrapped trace data
   * @returns The unwrapped trace data
   */
  private unwrapTrace(
    obj: Record<string, unknown>,
  ): Record<string, unknown> | null {
    return getObjectDataFromUnknown({ data: obj, key: "trace" });
  }

  /**
   * Checks if trace data contains agent collaboration markers.
   * @param traceData - The trace data to check
   * @param eventType - The type of event in the trace data
   * @param chunkType - The type of chunk in the event
   * @returns True if agent collaboration is detected, false otherwise
   */
  private detectAgentCollaboration(
    traceData: StringKeyedObject,
    eventType: string,
    chunkType: string,
  ): boolean {
    const eventObj = getObjectDataFromUnknown({
      data: traceData,
      key: eventType,
    });
    if (!eventObj) {
      return false;
    }
    const chunkObj = getObjectDataFromUnknown({
      data: eventObj,
      key: chunkType,
    });
    if (!chunkObj) {
      return false;
    }
    return !!(
      chunkObj &&
      (chunkObj["agentCollaboratorInvocationInput"] ||
        chunkObj["agentCollaboratorInvocationOutput"])
    );
  }

  /**
   * Routes trace data to appropriate processing method based on trace state.
   * @param params
   * @param params.traceData {StringKeyedObject} - The trace data to process
   * @param params.parent {AgentTraceNode} - The parent node to which the trace data might be attached
   * @param params.nodeTraceId {string} - The unique ID for the trace node
   * @param params.agentChildId {string | null} - The ID for a potential agent collaboration child node, or false if not applicable
   * @param params.eventType {string} - The type of event in the trace data
   * @param params.chunkType {string} - The type of chunk in the event
   */
  private routeTraceData({
    traceData,
    parent,
    nodeTraceId,
    agentChildId,
    eventType,
    chunkType,
  }: {
    traceData: StringKeyedObject;
    parent: AgentTraceNode;
    nodeTraceId: string;
    agentChildId: string | null;
    eventType: string;
    chunkType: string;
  }) {
    // New node
    if (!this.seenIds.has(nodeTraceId)) {
      this.addNewTraceNodeIfAbsent(
        parent,
        nodeTraceId,
        eventType,
        chunkType,
        traceData,
      );
    }

    // Current parent is correct target
    else if ([nodeTraceId, agentChildId].includes(parent.nodeTraceId)) {
      this.processChunkOrCreateAgentCollaborationNode({
        parent,
        agentChildId,
        eventType,
        chunkType,
        traceData,
      });
    }

    // Need to pop up to parent node
    else if (parent.parentTraceNode?.nodeTraceId === nodeTraceId) {
      this.traceStack.pop();
      if (!this.traceStack.head) {
        diag.warn(
          "No trace stack head found inside routeTraceData. Skipping appendChunkToCurrentNodeOrStartNewSpan.",
        );
        return;
      }
      this.appendChunkToCurrentNodeOrStartNewSpan({
        node: this.traceStack.head,
        chunkType,
        traceData,
      });
    }

    // Node exists elsewhere in tree
    else if (this.seenIds.has(nodeTraceId)) {
      this.processTraceByExistingId({
        agentChildId,
        eventType,
        chunkType,
        traceData,
      });
    }
  }

  /**
   * Creates a new trace node when encountering an unseen trace ID.
   * @param parent - The potential parent node for the new trace node
   * @param nodeTraceId - The unique ID for the new trace node
   * @param eventType - The type of event for the new node
   * @param chunkType - The type of chunk in the event
   * @param traceData - The trace data to associate with the new node
   */
  private addNewTraceNodeIfAbsent(
    parent: AgentTraceNode,
    nodeTraceId: string,
    eventType: string,
    chunkType: string,
    traceData: StringKeyedObject,
  ): void {
    const excluded = ["bedrock.invoke_agent", "agent-collaborator"];
    let anchor = parent;
    if (
      !excluded.includes(parent.nodeType ?? "") &&
      parent.nodeType !== eventType
    ) {
      this.traceStack.pop();
      anchor = this.traceStack.head ?? this.rootNode;
    }
    this.createAndAttachChildNode(
      anchor,
      nodeTraceId,
      eventType,
      chunkType,
      traceData,
    );
  }

  /**
   * Creates and attaches a new child node to the parent.
   * @param parent - The parent node to attach the new node to
   * @param nodeTraceId - The unique ID for the new node
   * @param eventType - The type of event for the new node
   * @param chunkType - The type of chunk in the event
   * @param traceData - The trace data to associate with the new node
   */
  private createAndAttachChildNode(
    parent: AgentTraceNode,
    nodeTraceId: string,
    eventType: string,
    chunkType: string,
    traceData: StringKeyedObject,
  ): void {
    const eventObj =
      getObjectDataFromUnknown({ data: traceData, key: eventType }) ?? {};
    const chunkObj =
      getObjectDataFromUnknown({ data: eventObj, key: chunkType }) ?? {};
    let newNode: AgentTraceNode;
    if (chunkObj.agentCollaboratorInvocationInput) {
      newNode = new AgentTraceNode({
        traceId: nodeTraceId,
        eventType: "agent-collaborator",
      });
      newNode.addChunk(traceData);
    } else if (eventType === "failureTrace") {
      newNode = new AgentTraceNode({
        traceId: nodeTraceId,
        eventType,
      });
      newNode.addChunk(traceData);
    } else {
      newNode = new AgentTraceNode({
        traceId: nodeTraceId,
        eventType,
      });
      const span = new AgentChunkSpan();
      span.addChunk(traceData);
      span.parentNode = parent;
      newNode.addSpan(span);
    }

    newNode.parentTraceNode = parent;
    parent.addSpan(newNode);
    this.traceStack.push(newNode);

    this.nodesById[nodeTraceId] = newNode;
    this.seenIds.add(nodeTraceId);
  }

  /**
   * Processes chunks for existing nodes or creates agent collaboration nodes.
   * @param params
   * @param params.parent {AgentTraceNode} - The parent node to process
   * @param params.agentChildId {string | null} - The ID for a potential agent collaboration child node, or false if not applicable
   * @param params.eventType {string} - The type of event in the trace data
   * @param params.chunkType {string} - The type of chunk in the event
   * @param params.traceData {StringKeyedObject} - The trace data to process
   */
  private processChunkOrCreateAgentCollaborationNode({
    parent,
    agentChildId,
    eventType,
    chunkType,
    traceData,
  }: {
    parent: AgentTraceNode;
    agentChildId: string | null;
    eventType: string;
    chunkType: string;
    traceData: StringKeyedObject;
  }) {
    const eventObj = traceData[eventType] ?? {};
    const chunkObj =
      getObjectDataFromUnknown({ data: eventObj, key: chunkType }) ?? {};
    if (agentChildId && chunkObj.agentCollaboratorInvocationInput) {
      this.createAndAttachChildNode(
        parent,
        agentChildId,
        eventType,
        chunkType,
        traceData,
      );
    } else {
      this.appendChunkToCurrentNodeOrStartNewSpan({
        node: parent,
        chunkType,
        traceData,
      });
    }
  }

  /**
   * Handles trace data for nodes that exist elsewhere in the trace tree.
   * @param params
   * @param params.agentChildId {string | null} - The ID for a potential agent collaboration child node, or null if not applicable
   * @param params.eventType {string} - The type of event in the trace data
   * @param params.chunkType {string} - The type of chunk in the event
   * @param params.traceData {StringKeyedObject} - The trace data to process
   */
  private processTraceByExistingId({
    agentChildId,
    eventType,
    chunkType,
    traceData,
  }: {
    agentChildId: string | null;
    eventType: string;
    chunkType: string;
    traceData: StringKeyedObject;
  }): void {
    const eventObj =
      getObjectDataFromUnknown({ data: traceData, key: eventType }) ?? {};
    const chunkObj =
      getObjectDataFromUnknown({ data: eventObj, key: chunkType }) ?? {};
    if (agentChildId && chunkObj.agentCollaboratorInvocationOutput) {
      while (
        this.traceStack.head &&
        this.traceStack.head.nodeTraceId !== agentChildId
      ) {
        this.traceStack.pop();
      }
      (this.traceStack.head ?? this.rootNode).addChunk(traceData);
    } else {
      if (this.traceStack.head) {
        this.appendChunkToCurrentNodeOrStartNewSpan({
          node: this.traceStack.head,
          chunkType,
          traceData,
        });
      } else {
        this.appendChunkToCurrentNodeOrStartNewSpan({
          node: this.rootNode,
          chunkType,
          traceData,
        });
      }
    }
  }

  /**
   * Appends chunk to current span or creates a new span if required.
   * @param node {AgentTraceNode} - The node to which the chunk should be associated
   * @param chunkType {string} - The type of chunk in the trace data
   * @param traceData {StringKeyedObject} - The trace data to process
   */
  private appendChunkToCurrentNodeOrStartNewSpan({
    node,
    chunkType,
    traceData,
  }: {
    node: AgentTraceNode;
    chunkType: string;
    traceData: StringKeyedObject;
  }): void {
    const mustStartNew =
      ["invocationInput", "modelInvocationInput"].includes(chunkType) ||
      !node.currentSpan;

    if (mustStartNew) {
      this.startNewSpanWithChunk(node, chunkType, traceData);
    } else {
      node.currentSpan!.addChunk(traceData);
    }
  }

  /**
   * Creates a new span with the given chunk data.
   * @param node - The node to which the new span should be attached
   * @param chunkType - The type of chunk for the new span
   * @param traceData - The trace data to associate with the new span
   */
  private startNewSpanWithChunk(
    node: AgentTraceNode,
    chunkType: string,
    traceData: StringKeyedObject,
  ): void {
    const span = new AgentChunkSpan();
    span.addChunk(traceData);
    span.parentNode = node;
    node.addSpan(span);
  }
}
