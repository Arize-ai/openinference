import { AgentChunkSpan } from "./agent-chunk-span";
import { StringKeyedObject } from "../index";

/**
 * Tree node representing an agent or model invocation. Each node can own:
 * - Multiple spans (AgentChunkSpan)
 * - Child nodes (forming the hierarchy)
 * - "Loose" chunks that do not belong to a span (e.g., failure traces)
 */
export class AgentTraceNode {
  /** Stable identifier for the node (prefixed trace-ID). */
  public readonly nodeTraceId: string;

  /** Semantic type, e.g., 'orchestrationTrace' or 'agent-collaborator'. */
  public readonly nodeType: string | null;

  /** Mixed list of spans and child trace nodes. */
  public readonly spans: (AgentChunkSpan | AgentTraceNode)[] = [];

  /** Back-pointer to parent in the tree. */
  public parentTraceNode: AgentTraceNode | null = null;

  /** The actively filling span (used while streaming). */
  public currentSpan: AgentChunkSpan | null = null;

  /** Chunks attached directly to the node (outside any span). */
  public readonly chunks: StringKeyedObject[] = [];

  /**
   * Creates a new AgentTraceNode instance.
   * @param traceId - Stable identifier for the node (prefixed trace-ID)
   * @param eventType - Semantic type, e.g., 'orchestrationTrace' or 'agent-collaborator'
   */
  constructor(traceId: string, eventType: string | null = null) {
    this.nodeTraceId = traceId;
    this.nodeType = eventType;
  }

  /**
   * Adds a span or child node to this node.
   * If a span is added, it becomes the current active span.
   * @param span - The AgentChunkSpan or AgentTraceNode to add
   */
  addSpan(span: AgentChunkSpan | AgentTraceNode): void {
    this.spans.push(span);
    if (span instanceof AgentChunkSpan) this.currentSpan = span;
  }

  /**
   * Adds a chunk directly to this node (not part of any span).
   * @param chunk - The chunk object to add
   */
  addChunk(chunk: StringKeyedObject): void {
    this.chunks.push(chunk);
  }

  /**
   * Converts the node and its children to a plain object for serialization.
   * @returns An object representation of the node and its hierarchy
   */
  toObject(): object {
    return {
      nodeTraceId: this.nodeTraceId,
      nodeType: this.nodeType,
      chunks: this.chunks,
      spans: this.spans.map((span) =>
        span instanceof AgentChunkSpan
          ? (span.toObject?.() ?? span)
          : (span as AgentTraceNode).toObject(),
      ),
    };
  }
}
