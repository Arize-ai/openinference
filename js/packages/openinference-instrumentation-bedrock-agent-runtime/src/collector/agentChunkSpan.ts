import { StringKeyedObject } from "../types";
import { AgentTraceNode } from "./agentTraceNode";

/**
 * Logical work-unit inside a node. A span groups sequential chunks that share
 * the same spanType. Spans can be nested under nodes but never under each other.
 */
export class AgentChunkSpan {
  /** Raw payloads belonging to this span in order of arrival. */
  public readonly chunks: StringKeyedObject[] = [];

  /** Child nodes created from within this span (e.g., agent collaborators). */
  public readonly childrenNodes: AgentTraceNode[] = [];

  /** Back-pointer to the node that owns this span. */
  public parentNode: AgentTraceNode | null = null;

  /**
   * Create a new AgentChunkSpan.
   */
  constructor() {}

  /**
   * Add a chunk to this span.
   * @param {StringKeyedObject} chunk - Chunk object to add.
   */
  addChunk(chunk: StringKeyedObject) {
    this.chunks.push(chunk);
  }
}
