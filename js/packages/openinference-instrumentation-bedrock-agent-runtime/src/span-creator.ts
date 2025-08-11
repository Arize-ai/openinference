import { AgentTraceNode } from "./collector/agent-trace-node";
import { Span, SpanStatusCode, trace, context } from "@opentelemetry/api";
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";
import { Attributes } from "./attributes/attributes";
import { AttributeExtractor } from "./attributes/attribute-extractor";
import { AgentChunkSpan } from "./collector/agent-chunk-span";
import {
  getInputAttributes,
  getOutputAttributes,
  getSpanKindAttributes,
} from "./attributes/attribute-utils";
import { OITracer } from "@arizeai/openinference-core";

/**
 * SpanCreator creates and manages OpenTelemetry spans from agent trace nodes.
 */
export class SpanCreator {
  private oiTracer: OITracer;

  /**
   * Constructs a SpanCreator instance.
   * @param oiTracer The OpenInference tracer to use for span creation.
   */
  constructor(oiTracer: OITracer) {
    this.oiTracer = oiTracer;
  }

  /**
   * Recursively creates spans from a trace node and its children.
   * @param parentSpan The parent span for the current trace node.
   * @param traceNode The agent trace node to process.
   */
  public createSpans(parentSpan: Span, traceNode: AgentTraceNode) {
    for (const traceSpan of traceNode.spans) {
      const attributes = this.prepareSpanAttributes(traceSpan);

      // Set span kind based on trace span type
      if (traceSpan instanceof AgentTraceNode) {
        attributes.spanKind = OpenInferenceSpanKind.CHAIN; // Placeholder value
        if (traceSpan.nodeType === "agent-collaborator") {
          attributes.spanKind = OpenInferenceSpanKind.AGENT; // Placeholder value
        }
        this.setParentSpanInputAttributes(attributes, traceSpan);
        this.setParentSpanOutputAttributes(attributes, traceSpan);
      }

      // Create span with appropriate timing
      const startTime = this.fetchSpanStartTime(attributes, traceSpan);
      const span = this.createChainSpan(parentSpan, attributes, startTime);
      span.setStatus({ code: SpanStatusCode.OK });

      // Process child spans recursively
      if (traceSpan instanceof AgentTraceNode) {
        this.createSpans(span, traceSpan);
      }

      // End span with appropriate timing
      const endTime = this.fetchSpanEndTime(attributes, traceSpan);
      span.end(endTime ? endTime : undefined);
    }
  }

  /**
   * Prepares span attributes from a trace span.
   * Extracts relevant attributes and metadata from the trace span and its chunks.
   * @param traceSpan The trace span to process.
   * @returns The prepared attributes object.
   */
  private prepareSpanAttributes(
    traceSpan: AgentChunkSpan | AgentTraceNode,
  ): Attributes {
    const attributes = new Attributes();

    // Set name from a node type if it's an AgentTraceNode
    if (traceSpan instanceof AgentTraceNode) {
      attributes.name = traceSpan.nodeType || "";
    }

    // Process each chunk in the trace span
    if (Array.isArray(traceSpan.chunks)) {
      for (const traceData of traceSpan.chunks) {
        const traceEvent = AttributeExtractor.getEventType(traceData);
        const eventData =
          (traceData[traceEvent] as Record<string, Record<string, unknown>>) ??
          {};

        // Process model invocation input
        if ("modelInvocationInput" in eventData) {
          this.processModelInvocationInput(eventData, attributes);
        }
        // Process model invocation output
        if ("modelInvocationOutput" in eventData) {
          this.processModelInvocationOutput(eventData, attributes);
        }
        // Process invocation input
        if ("invocationInput" in eventData) {
          this.processInvocationInput(eventData, attributes);
        }
        // Process observation
        if ("observation" in eventData) {
          this.processObservation(eventData, attributes);
        }
        // Process rationale
        if ("rationale" in eventData) {
          this.processRationale(eventData, attributes);
        }
        // Process failure trace
        if (traceEvent === "failureTrace") {
          this.processFailureTrace(eventData, attributes);
        }
      }
    }
    return attributes;
  }

  /**
   * Processes model invocation input and updates attributes.
   * @param eventData The event data containing model invocation input.
   * @param attributes The attributes object to update.
   */
  private processModelInvocationInput(
    eventData: Record<string, Record<string, unknown>>,
    attributes: Attributes,
  ): void {
    const modelInvocationInput = eventData["modelInvocationInput"] ?? {};
    Object.assign(
      attributes.requestAttributes,
      AttributeExtractor.getAttributesFromModelInvocationInput(
        modelInvocationInput,
      ),
    );
    attributes.name = "LLM";
    attributes.spanKind = OpenInferenceSpanKind.LLM;
  }

  /**
   * Processes model invocation output and updates attributes and metadata.
   * @param eventData The event data containing model invocation output.
   * @param attributes The attributes object to update.
   */
  private processModelInvocationOutput(
    eventData: Record<string, Record<string, unknown>>,
    attributes: Attributes,
  ): void {
    const modelInvocationOutput = eventData["modelInvocationOutput"] ?? {};
    Object.assign(
      attributes.outputAttributes,
      AttributeExtractor.getAttributesFromModelInvocationOutput(
        modelInvocationOutput,
      ),
    );
    Object.assign(
      attributes.metadata,
      AttributeExtractor.getMetadataAttributes(
        modelInvocationOutput["metadata"] as Record<string, unknown>,
      ),
    );
  }

  /**
   * Processes invocation input and updates attributes.
   * @param eventData The event data containing invocation input.
   * @param attributes The attributes object to update.
   */
  private processInvocationInput(
    eventData: Record<string, Record<string, unknown>>,
    attributes: Attributes,
  ): void {
    const invocationInput = eventData["invocationInput"] ?? {};

    const invocationType =
      typeof invocationInput["invocationType"] === "string"
        ? invocationInput["invocationType"]
        : "";

    if (
      "agentCollaboratorInvocationInput" in invocationInput &&
      typeof invocationInput["agentCollaboratorInvocationInput"] === "object" &&
      invocationInput["agentCollaboratorInvocationInput"] !== null
    ) {
      const agentCollaboratorInput = invocationInput[
        "agentCollaboratorInvocationInput"
      ] as Record<string, unknown>;

      const agentCollaboratorName =
        typeof agentCollaboratorInput["agentCollaboratorName"] === "string"
          ? agentCollaboratorInput["agentCollaboratorName"]
          : "";

      attributes.name = `${invocationType.toLowerCase()}[${agentCollaboratorName}]`;
      attributes.spanKind = OpenInferenceSpanKind.AGENT;
      attributes.spanType = "agent_collaborator";
    } else {
      attributes.name = invocationType.toLowerCase();
      attributes.spanKind = OpenInferenceSpanKind.TOOL;
    }

    Object.assign(
      attributes.requestAttributes,
      AttributeExtractor.getAttributesFromInvocationInput(invocationInput),
    );
  }

  /**
   * Processes observation and updates output attributes and metadata.
   * @param eventData The event data containing observation.
   * @param attributes The attributes object to update.
   */
  private processObservation(
    eventData: Record<string, Record<string, unknown>>,
    attributes: Attributes,
  ): void {
    const observation = eventData["observation"] ?? {};
    Object.assign(
      attributes.outputAttributes,
      AttributeExtractor.getAttributesFromObservation(observation),
    );
    Object.assign(
      attributes.metadata,
      AttributeExtractor.extractMetadataAttributesFromObservation(observation),
    );
  }

  /**
   * Processes rationale and updates output attributes.
   * @param eventData The event data containing rationale.
   * @param attributes The attributes object to update.
   */
  private processRationale(
    eventData: Record<string, Record<string, unknown>>,
    attributes: Attributes,
  ): void {
    const rationaleText = eventData["rationale"]?.["text"] ?? "";
    if (rationaleText) {
      Object.assign(
        attributes.outputAttributes,
        getOutputAttributes(rationaleText),
      );
    }
  }

  /**
   * Processes failure trace and updates output attributes and metadata.
   * @param eventData The event data containing failure trace.
   * @param attributes The attributes object to update.
   */
  private processFailureTrace(
    eventData: Record<string, Record<string, unknown>>,
    attributes: Attributes,
  ): void {
    Object.assign(
      attributes.outputAttributes,
      AttributeExtractor.getFailureTraceAttributes(eventData),
    );
    Object.assign(
      attributes.metadata,
      AttributeExtractor.getMetadataAttributes(eventData["metadata"] ?? {}),
    );
  }

  /**
   * Sets parent span input attributes by extracting from nested trace nodes.
   * @param attributes The attributes object to update.
   * @param traceNode The agent trace node or chunk span to process.
   */
  private setParentSpanInputAttributes(
    attributes: Attributes,
    traceNode: AgentTraceNode | AgentChunkSpan,
  ): void {
    // Only process if traceNode is an AgentTraceNode
    if (!(traceNode instanceof AgentTraceNode)) {
      return;
    }

    const inputAttributes: Record<string, unknown> = {};

    for (const span of traceNode.spans) {
      if (!span.chunks) continue;
      for (const traceData of span.chunks) {
        const traceEvent = AttributeExtractor.getEventType(traceData);
        const eventData = traceData[traceEvent] ?? {};

        // Extract from model invocation input
        if ("modelInvocationInput" in eventData) {
          const modelInvocationInput: Record<string, string> =
            (eventData["modelInvocationInput"] as Record<string, string>) ?? {};
          const text = modelInvocationInput["text"] ?? "";
          const messages = AttributeExtractor.getMessagesObject(text);
          for (const message of messages) {
            if (message.role === "user" && message.content) {
              Object.assign(
                inputAttributes,
                getInputAttributes(message.content),
              );
              Object.assign(inputAttributes, attributes.requestAttributes);
              attributes.requestAttributes = inputAttributes;
              return;
            }
          }
        }

        // Extract from invocation input
        if ("invocationInput" in eventData) {
          const invocationInput: Record<string, unknown> =
            (eventData["invocationInput"] as Record<string, unknown>) ?? {};
          const attrs =
            AttributeExtractor.getParentInputAttributesFromInvocationInput(
              invocationInput,
            );
          Object.assign(inputAttributes, attrs);
          Object.assign(inputAttributes, attributes.requestAttributes);
          attributes.requestAttributes = inputAttributes;
          return;
        }
      }
      // Recursively check nested nodes
      if (span instanceof AgentTraceNode) {
        this.setParentSpanInputAttributes(attributes, span);
        return;
      }
    }
  }

  /**
   * Sets parent span output attributes by extracting from nested trace nodes.
   * @param attributes The attributes object to update.
   * @param traceNode The agent trace node or chunk span to process.
   */
  private setParentSpanOutputAttributes(
    attributes: Attributes,
    traceNode: AgentTraceNode | AgentChunkSpan,
  ): void {
    // Only process if traceNode is an AgentTraceNode
    if (!(traceNode instanceof AgentTraceNode)) {
      return;
    }

    // Reverse iterate over spans
    for (let i = traceNode.spans.length - 1; i >= 0; i--) {
      const span = traceNode.spans[i];
      if (!span.chunks) continue;
      // Reverse iterate over chunks
      for (let j = span.chunks.length - 1; j >= 0; j--) {
        const traceData = span.chunks[j];
        const traceEvent = AttributeExtractor.getEventType(traceData);
        const eventData = traceData[traceEvent] ?? {};

        // Extract from model invocation output
        if ("modelInvocationOutput" in eventData) {
          const modelInvocationOutput: Record<string, unknown> =
            (eventData["modelInvocationOutput"] as Record<string, unknown>) ??
            {};
          const parsedResponse =
            (modelInvocationOutput["parsedResponse"] as Record<
              string,
              unknown
            >) ?? {};
          const outputText = parsedResponse["text"] ?? "";
          if (outputText) {
            Object.assign(
              attributes.requestAttributes,
              getOutputAttributes(outputText),
            );
            return;
          }
          const rationaleText = parsedResponse["rationale"] ?? "";
          if (rationaleText) {
            Object.assign(
              attributes.requestAttributes,
              getOutputAttributes(rationaleText),
            );
            return;
          }
        }

        // Extract from observation
        if ("observation" in eventData) {
          const observation: Record<string, unknown> =
            (eventData["observation"] as Record<string, unknown>) ?? {};
          const finalResponse: Record<string, unknown> =
            (observation["finalResponse"] as Record<string, unknown>) ?? {};
          if (finalResponse && finalResponse["text"]) {
            Object.assign(
              attributes.requestAttributes,
              getOutputAttributes(finalResponse["text"]),
            );
            return;
          }
        }
      }
      // Recursively check nested nodes
      if (span instanceof AgentTraceNode) {
        this.setParentSpanOutputAttributes(attributes, span);
        return;
      }
    }
  }

  /**
   * Fetches the start time for a span from attributes or trace span.
   * @param attributes The attributes object that may contain timing metadata.
   * @param traceSpan The trace span to extract time from.
   * @returns The start timestamp in nanoseconds if found, undefined otherwise.
   */
  private fetchSpanStartTime(
    attributes: Attributes,
    traceSpan: AgentTraceNode | AgentChunkSpan,
  ): number | undefined {
    return this.fetchSpanTime(attributes, traceSpan, "start_time", false);
  }

  /**
   * Fetches the end time for a span from attributes or trace span.
   * @param attributes The attributes object that may contain timing metadata.
   * @param traceSpan The trace span to extract time from.
   * @returns The end timestamp in nanoseconds if found, undefined otherwise.
   */
  private fetchSpanEndTime(
    attributes: Attributes,
    traceSpan: AgentTraceNode | AgentChunkSpan,
  ): number | undefined {
    return this.fetchSpanTime(attributes, traceSpan, "end_time", true);
  }

  /**
   * Fetches span time (start or end) from attributes or trace span.
   * Recursively searches for timestamp information in the trace data.
   * @param attributes The attributes object that may contain timing metadata.
   * @param traceSpan The trace span to extract time from.
   * @param timeKey The key to look for ('start_time' or 'end_time').
   * @param reverse Whether to traverse spans and chunks in reverse order.
   * @returns The timestamp in nanoseconds if found, undefined otherwise.
   */
  private fetchSpanTime(
    attributes: Attributes,
    traceSpan: AgentTraceNode | AgentChunkSpan,
    timeKey: string,
    reverse: boolean = false,
  ): number | undefined {
    // First check if time is already in attributes metadata
    if (attributes.metadata && attributes.metadata[timeKey] !== undefined) {
      return Number(attributes.metadata[timeKey]);
    }
    if (!(traceSpan instanceof AgentTraceNode)) {
      return undefined;
    }
    const spans = reverse ? [...traceSpan.spans].reverse() : traceSpan.spans;
    for (const span of spans) {
      const chunks = reverse ? [...span.chunks].reverse() : span.chunks;
      for (const traceData of chunks) {
        const traceEvent = AttributeExtractor.getEventType(traceData);
        const eventData = traceData[traceEvent] ?? {};
        // Check model invocation output
        if ("modelInvocationOutput" in eventData) {
          const modelInvocationOutput: Record<string, unknown> =
            (eventData["modelInvocationOutput"] as Record<string, unknown>) ??
            {};
          const metadata = AttributeExtractor.getMetadataAttributes(
            (modelInvocationOutput["metadata"] as Record<string, unknown>) ??
              {},
          );
          if (metadata && metadata[timeKey] !== undefined) {
            return Number(metadata[timeKey]);
          }
        }
        // Check observation
        if ("observation" in eventData) {
          const observation = eventData["observation"] ?? {};
          const metadata = AttributeExtractor.getObservationMetadataAttributes(
            observation as Record<string, unknown>,
          );
          if (metadata && metadata[timeKey] !== undefined) {
            return Number(metadata[timeKey]);
          }
        }
      }
      // Recursively check nested nodes
      if (span instanceof AgentTraceNode) {
        const nestedTime = this.fetchSpanTime(
          attributes,
          span,
          timeKey,
          reverse,
        );
        if (nestedTime !== undefined) {
          return nestedTime;
        }
      }
    }
    return undefined;
  }

  /**
   * Creates a new span with the given attributes and parent span.
   * Handles setting span attributes, extracting and merging metadata, and setting the appropriate start time.
   * @param parentSpan The parent span for this new span.
   * @param attributes The attributes object containing span properties.
   * @param startTime Optional explicit start time for the span (nanoseconds).
   * @returns The newly created span.
   */
  private createChainSpan(
    parentSpan: Span,
    attributes: Attributes,
    startTime?: number,
  ): Span {
    // Extract start time from attributes metadata if available
    const effectiveStartTime = attributes.metadata?.start_time ?? startTime;

    // Create the span with appropriate context
    const span = this.oiTracer.startSpan(
      attributes.name || "LLM",
      {
        startTime: effectiveStartTime ? Number(effectiveStartTime) : undefined,
        attributes: getSpanKindAttributes(attributes.spanKind),
      },
      trace.setSpan(context.active(), parentSpan),
    );

    // Collect and merge metadata from various sources
    const metadata = { ...(attributes.metadata || {}) };

    // Set request attributes and extract any metadata
    if (attributes?.requestAttributes) {
      if ("metadata" in attributes.requestAttributes) {
        Object.assign(metadata, attributes.requestAttributes["metadata"]);
        delete attributes.requestAttributes["metadata"];
      }
      span.setAttributes(
        attributes.requestAttributes as Record<string, string>,
      );
    }

    // Set output attributes and extract any metadata
    if (attributes.outputAttributes) {
      if ("metadata" in attributes.outputAttributes) {
        Object.assign(metadata, attributes.outputAttributes["metadata"]);
        delete attributes.outputAttributes["metadata"];
      }
      span.setAttributes(attributes.outputAttributes as Record<string, string>);
    }

    // Add collected metadata as a JSON string
    if (Object.keys(metadata).length > 0) {
      span.setAttributes({ metadata: JSON.stringify(metadata) });
    }
    return span;
  }
}
