import { AgentTraceNode } from "./collector/agentTraceNode";
import {
  Span,
  SpanStatusCode,
  trace,
  context,
  Attributes,
} from "@opentelemetry/api";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  extractMetadataAttributesFromObservation,
  getAttributesFromInvocationInput,
  getAttributesFromModelInvocationInput,
  getAttributesFromModelInvocationOutput,
  getAttributesFromObservation,
  getEventType,
  getFailureTraceAttributes,
  getInputMessagesObject,
  getMetadataAttributes,
  getParentInputAttributesFromInvocationInput,
  getStartAndEndTimeFromMetadata,
  getStringAttributeValueFromUnknown,
} from "./attributes/attributeExtractionUtils";
import { AgentChunkSpan } from "./collector/agentChunkSpan";
import {
  getInputAttributes,
  getOutputAttributes,
} from "./attributes/attributeUtils";
import {
  assertUnreachable,
  OITracer,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import { getObjectDataFromUnknown } from "./utils/jsonUtils";
import { StringKeyedObject } from "./types";

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
   * @param params
   * @param params.parentSpan {Span} - The parent span for the current trace node.
   * @param params.traceNode {AgentTraceNode} - The agent trace node to process.
   */
  public createSpans({
    parentSpan,
    traceNode,
  }: {
    parentSpan: Span;
    traceNode: AgentTraceNode;
  }) {
    for (const traceSpan of traceNode.spans) {
      const { attributes, name, rawMetadata } =
        this.prepareSpanAttributes(traceSpan);
      let finalAttributes = { ...attributes };

      // Set span kind based on trace span type if we didn't already set a span kind in the attributes
      if (
        traceSpan instanceof AgentTraceNode &&
        finalAttributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] == null
      ) {
        finalAttributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] =
          OpenInferenceSpanKind.CHAIN;
        if (traceSpan.nodeType === "agent-collaborator") {
          finalAttributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] =
            OpenInferenceSpanKind.AGENT;
        }
        const inputParentAttributes = this.getParentSpanInputAttributes({
          traceNode: traceSpan,
        });
        const { attributes: outputAttributes } =
          this.getParentSpanOutputAttributes({
            traceNode: traceSpan,
          });
        finalAttributes = {
          ...finalAttributes,
          ...outputAttributes,
          ...inputParentAttributes,
        };
      }

      // Create span with appropriate timing
      const startTime = this.fetchSpanStartTime({
        traceSpan,
        metadata: rawMetadata,
      });

      const span = this.createChainSpan({
        parentSpan,
        attributes: finalAttributes,
        name,
        startTime,
      });
      span.setStatus({ code: SpanStatusCode.OK });

      // Process child spans recursively
      if (traceSpan instanceof AgentTraceNode) {
        this.createSpans({
          parentSpan: span,
          traceNode: traceSpan,
        });
      }

      // End span with appropriate timing
      const endTime = this.fetchSpanEndTime({
        traceSpan,
        metadata: rawMetadata,
      });

      span.end(endTime ? endTime : undefined);
    }
  }

  /**
   * Prepares span attributes from a trace span.
   * Extracts relevant attributes and metadata from the trace span and its chunks.
   * @param traceSpan The trace span to process.
   * @returns The prepared attributes object.
   */
  private prepareSpanAttributes(traceSpan: AgentChunkSpan | AgentTraceNode): {
    attributes: Attributes;
    name: string;
    rawMetadata: StringKeyedObject;
  } {
    let attributes: Attributes = {};
    let metadata: StringKeyedObject = {};
    let name: string | null = null;
    // Set name from a node type if it's an AgentTraceNode and has no chunks
    if (
      traceSpan instanceof AgentTraceNode &&
      traceSpan.nodeType != null &&
      traceSpan.chunks.length === 0
    ) {
      name = traceSpan.nodeType;
    }
    // Process each chunk in the trace span
    if (Array.isArray(traceSpan.chunks)) {
      for (const traceData of traceSpan.chunks) {
        const traceEvent = getEventType(traceData);
        if (traceEvent == null) {
          continue;
        }
        const eventData =
          getObjectDataFromUnknown({ data: traceData, key: traceEvent }) ?? {};
        const modelInvocationAttributes =
          this.processModelInvocationInput(eventData);
        const invocationAttributes = this.processInvocationInput(eventData);
        const { attributes: outputAttributes, metadata: outputMetadata } =
          this.processModelInvocationOutput(eventData);
        const {
          attributes: observationAttributes,
          metadata: observationMetadata,
        } = this.processObservation(eventData);
        const rationaleAttributes = this.processRationale(eventData);
        const kindAndName = this.getSpanKindAndNameFromEventData(eventData);

        if (name == null) {
          name = kindAndName.name;
        }

        let failureTraceAttributes: Attributes = {};
        let failureTraceMetadata: StringKeyedObject = {};
        if (traceEvent === "failureTrace") {
          const processFailureTraceResult = this.processFailureTrace(eventData);
          failureTraceAttributes = processFailureTraceResult.attributes;
          failureTraceMetadata = processFailureTraceResult.metadata ?? {};
        }
        attributes = {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: kindAndName.spanKind,
          ...attributes,
          ...modelInvocationAttributes,
          ...outputAttributes,
          ...observationAttributes,
          ...rationaleAttributes,
          ...failureTraceAttributes,
          ...invocationAttributes,
        };
        metadata = {
          ...metadata,
          ...outputMetadata,
          ...observationMetadata,
          ...failureTraceMetadata,
        };
      }

      return {
        attributes: {
          ...attributes,
          metadata: safelyJSONStringify(metadata) ?? undefined,
        },
        rawMetadata: metadata,
        name: name ?? "LLM",
      };
    }
    return {
      attributes: {},
      rawMetadata: {},
      name: name ?? "LLM",
    };
  }

  /**
   * Processes model invocation input and updates attributes.
   * @param eventData The event data containing model invocation input.
   * @param attributes The attributes object to update.
   */
  private processModelInvocationInput(
    eventData: StringKeyedObject,
  ): Attributes {
    const modelInvocationInput =
      getObjectDataFromUnknown({
        data: eventData,
        key: "modelInvocationInput",
      }) ?? {};
    return getAttributesFromModelInvocationInput(modelInvocationInput);
  }

  /**
   * Processes model invocation output and updates attributes and metadata.
   * @param eventData The event data containing model invocation output.
   * @param attributes The attributes object to update.
   */
  private processModelInvocationOutput(eventData: StringKeyedObject): {
    attributes: Attributes;
    metadata: StringKeyedObject | null;
  } {
    const modelInvocationOutput =
      getObjectDataFromUnknown({
        data: eventData,
        key: "modelInvocationOutput",
      }) ?? {};
    const outputAttributes = getAttributesFromModelInvocationOutput(
      modelInvocationOutput,
    );
    return {
      attributes: outputAttributes,
      metadata: getMetadataAttributes(
        getObjectDataFromUnknown({
          data: modelInvocationOutput,
          key: "metadata",
        }) ?? {},
      ),
    };
  }

  private getSpanKindAndNameFromEventData(eventData: StringKeyedObject): {
    spanKind: OpenInferenceSpanKind;
    name: string;
  } {
    const invocationInput = getObjectDataFromUnknown({
      data: eventData,
      key: "invocationInput",
    });

    if (invocationInput == null) {
      return {
        spanKind: OpenInferenceSpanKind.LLM,
        name: "LLM",
      };
    }
    const invocationType =
      typeof invocationInput["invocationType"] === "string"
        ? invocationInput["invocationType"]
        : "";

    if (
      "agentCollaboratorInvocationInput" in invocationInput &&
      typeof invocationInput["agentCollaboratorInvocationInput"] === "object" &&
      invocationInput["agentCollaboratorInvocationInput"] != null
    ) {
      const agentCollaboratorInput =
        getObjectDataFromUnknown({
          data: invocationInput,
          key: "agentCollaboratorInvocationInput",
        }) ?? {};

      const agentCollaboratorName =
        typeof agentCollaboratorInput["agentCollaboratorName"] === "string"
          ? agentCollaboratorInput["agentCollaboratorName"]
          : "";

      return {
        spanKind: OpenInferenceSpanKind.AGENT,
        name: `${invocationType.toLowerCase()}[${agentCollaboratorName}]`,
      };
    } else {
      const maybeActionGroupInvocationInput = getObjectDataFromUnknown({
        data: invocationInput,
        key: "actionGroupInvocationInput",
      });
      const name = invocationType.toLowerCase().trim();
      if (maybeActionGroupInvocationInput) {
        return {
          spanKind: OpenInferenceSpanKind.TOOL,
          name: name ?? "actionGroupInvocationInput",
        };
      }

      const maybeCodeInterpreterInvocationInput = getObjectDataFromUnknown({
        data: invocationInput,
        key: "codeInterpreterInvocationInput",
      });
      if (maybeCodeInterpreterInvocationInput) {
        return {
          spanKind: OpenInferenceSpanKind.TOOL,
          name: name ?? "codeInterpreterInvocationInput",
        };
      }

      const maybeKnowledgeBaseLookupInput = getObjectDataFromUnknown({
        data: invocationInput,
        key: "knowledgeBaseLookupInput",
      });
      if (maybeKnowledgeBaseLookupInput) {
        return {
          spanKind: OpenInferenceSpanKind.RETRIEVER,
          name: name ?? "knowledgeBaseLookupInput",
        };
      }

      const maybeAgentCollaboratorInvocationInput = getObjectDataFromUnknown({
        data: invocationInput,
        key: "agentCollaboratorInvocationInput",
      });
      if (maybeAgentCollaboratorInvocationInput) {
        return {
          spanKind: OpenInferenceSpanKind.AGENT,
          name: name ?? "agentCollaboratorInvocationInput",
        };
      }
      return {
        spanKind: OpenInferenceSpanKind.TOOL,
        name: name ?? "tool",
      };
    }
  }

  /**
   * Processes invocation input and updates attributes.
   * @param eventData The event data containing invocation input.
   * @param attributes The attributes object to update.
   */
  private processInvocationInput(eventData: StringKeyedObject): Attributes {
    const invocationInput =
      getObjectDataFromUnknown({ data: eventData, key: "invocationInput" }) ??
      {};

    return getAttributesFromInvocationInput(invocationInput);
  }

  /**
   * Processes observation and updates output attributes and metadata.
   * @param eventData The event data containing observation.
   * @param attributes The attributes object to update.
   */
  private processObservation(eventData: StringKeyedObject): {
    attributes: Attributes;
    metadata: StringKeyedObject | null;
  } {
    const observation =
      getObjectDataFromUnknown({ data: eventData, key: "observation" }) ?? {};
    return {
      attributes: getAttributesFromObservation(observation),
      metadata: extractMetadataAttributesFromObservation(observation),
    };
  }

  /**
   * Processes rationale and updates output attributes.
   * @param eventData The event data containing rationale.
   * @param attributes The attributes object to update.
   */
  private processRationale(eventData: StringKeyedObject): Attributes {
    const rationaleText = getObjectDataFromUnknown({
      data: eventData,
      key: "rationale",
    })?.text;
    if (rationaleText != null) {
      return getOutputAttributes(rationaleText);
    }
    return {};
  }

  /**
   * Processes failure trace and updates output attributes and metadata.
   * @param eventData The event data containing failure trace.
   * @param attributes The attributes object to update.
   */
  private processFailureTrace(eventData: StringKeyedObject): {
    attributes: Attributes;
    metadata: StringKeyedObject | null;
  } {
    return {
      attributes: getFailureTraceAttributes(eventData),
      metadata: getMetadataAttributes(
        getObjectDataFromUnknown({ data: eventData, key: "metadata" }) ?? {},
      ),
    };
  }

  /**
   * Sets parent span input attributes by extracting from nested trace nodes.
   * Recursively checks nested nodes for input attributes.
   * Mutates the attributes object in place.
   * @param attributes The attributes object to update.
   * @param traceNode The agent trace node or chunk span to process.
   */
  private getParentSpanInputAttributes({
    accumulatedAttributes = {},
    traceNode,
  }: {
    accumulatedAttributes?: Attributes;
    traceNode: AgentTraceNode | AgentChunkSpan;
  }): Attributes {
    let newAttributes = { ...accumulatedAttributes };
    // Only process if traceNode is an AgentTraceNode
    if (!(traceNode instanceof AgentTraceNode)) {
      return { ...newAttributes };
    }

    for (const span of traceNode.spans) {
      if (!span.chunks) continue;
      for (const traceData of span.chunks) {
        const traceEvent = getEventType(traceData);
        if (traceEvent == null) {
          continue;
        }
        const eventData =
          getObjectDataFromUnknown({ data: traceData, key: traceEvent }) ?? {};

        // Extract from model invocation input
        if ("modelInvocationInput" in eventData) {
          const modelInvocationInput: StringKeyedObject =
            getObjectDataFromUnknown({
              data: eventData,
              key: "modelInvocationInput",
            }) ?? {};
          const text = getStringAttributeValueFromUnknown(
            modelInvocationInput?.text,
          );
          if (text == null) {
            continue;
          }
          const messages = getInputMessagesObject(text);
          for (const message of messages) {
            if (message.role === "user" && message.content) {
              newAttributes = {
                ...newAttributes,
                ...getInputAttributes(message.content),
              };
            }
          }
        }

        // Extract from invocation input
        if ("invocationInput" in eventData) {
          const invocationInput =
            getObjectDataFromUnknown({
              data: eventData,
              key: "invocationInput",
            }) ?? {};
          newAttributes = {
            ...newAttributes,
            ...getParentInputAttributesFromInvocationInput(invocationInput),
          };
        }
      }
      // Recursively check nested nodes
      if (span instanceof AgentTraceNode) {
        newAttributes = this.getParentSpanInputAttributes({
          accumulatedAttributes: newAttributes,
          traceNode: span,
        });
      }
    }
    return newAttributes;
  }

  /**
   * Sets parent span output attributes by extracting from nested trace nodes.
   * Recursively checks nested nodes for output attributes.
   * Mutates the attributes object in place.
   * @param attributes The attributes object to update.
   * @param traceNode The agent trace node or chunk span to process.
   */
  private getParentSpanOutputAttributes({
    accumulatedAttributes = {},
    traceNode,
  }: {
    accumulatedAttributes?: Attributes;
    traceNode: AgentTraceNode | AgentChunkSpan;
  }): { attributes: Attributes } {
    let newAttributes = { ...accumulatedAttributes };
    // Only process if traceNode is an AgentTraceNode
    if (!(traceNode instanceof AgentTraceNode)) {
      return { attributes: newAttributes };
    }

    // Reverse iterate over spans
    for (let i = traceNode.spans.length - 1; i >= 0; i--) {
      const span = traceNode.spans[i];
      if (!span.chunks) continue;
      // Reverse iterate over chunks
      for (let j = span.chunks.length - 1; j >= 0; j--) {
        const traceData = span.chunks[j];
        const traceEvent = getEventType(traceData);
        if (traceEvent == null) {
          continue;
        }
        const eventData =
          getObjectDataFromUnknown({ data: traceData, key: traceEvent }) ?? {};

        // Extract from model invocation output
        if ("modelInvocationOutput" in eventData) {
          const modelInvocationOutput =
            getObjectDataFromUnknown({
              data: eventData,
              key: "modelInvocationOutput",
            }) ?? {};
          const parsedResponse =
            getObjectDataFromUnknown({
              data: modelInvocationOutput,
              key: "parsedResponse",
            }) ?? {};
          const outputText = parsedResponse["text"] ?? "";
          if (outputText) {
            newAttributes = {
              ...newAttributes,
              ...getOutputAttributes(outputText),
            };
          }
          const rationaleText = parsedResponse["rationale"] ?? "";
          if (rationaleText) {
            newAttributes = {
              ...newAttributes,
              ...getOutputAttributes(rationaleText),
            };
          }
        }

        // Extract from observation
        if ("observation" in eventData) {
          const observation =
            getObjectDataFromUnknown({ data: eventData, key: "observation" }) ??
            {};
          const finalResponse =
            getObjectDataFromUnknown({
              data: observation,
              key: "finalResponse",
            }) ?? {};
          if (finalResponse?.text) {
            newAttributes = {
              ...newAttributes,
              ...getOutputAttributes(finalResponse.text),
            };
          }
        }
      }
      // Recursively check nested nodes
      if (span instanceof AgentTraceNode) {
        const { attributes } = this.getParentSpanOutputAttributes({
          accumulatedAttributes: newAttributes,
          traceNode: span,
        });
        newAttributes = attributes;
      }
    }
    return {
      attributes: newAttributes,
    };
  }

  /**
   * Fetches the start time for a span from attributes or trace span.
   * @param attributes The attributes object that may contain timing metadata.
   * @param traceSpan The trace span to extract time from.
   * @returns The start timestamp in nanoseconds if found, undefined otherwise.
   */
  private fetchSpanStartTime({
    traceSpan,
    metadata,
  }: {
    traceSpan: AgentTraceNode | AgentChunkSpan;
    metadata: StringKeyedObject;
  }): number | undefined {
    return this.fetchSpanTime({
      traceSpan,
      timeKey: "start",
      reverse: false,
      metadata,
    });
  }

  /**
   * Fetches the end time for a span from attributes or trace span.
   * @param attributes The attributes object that may contain timing metadata.
   * @param traceSpan The trace span to extract time from.
   * @returns The end timestamp in nanoseconds if found, undefined otherwise.
   */
  private fetchSpanEndTime({
    traceSpan,
    metadata,
  }: {
    traceSpan: AgentTraceNode | AgentChunkSpan;
    metadata: StringKeyedObject;
  }): number | undefined {
    return this.fetchSpanTime({
      traceSpan,
      timeKey: "end",
      reverse: true,
      metadata,
    });
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
  private fetchSpanTime({
    traceSpan,
    timeKey,
    reverse,
    metadata,
  }: {
    traceSpan: AgentTraceNode | AgentChunkSpan;
    metadata: StringKeyedObject;
    timeKey: "start" | "end";
    reverse: boolean;
  }): number | undefined {
    // If we've aleady found the time in the metadata, return it
    // We don't really need to store this in the metadata of the span but with all of the traversing of various attributes this does allow us to get it from various places then use it when creating the span
    switch (timeKey) {
      case "start":
        if (typeof metadata["start_time"] === "number") {
          return metadata["start_time"];
        }
        break;
      case "end":
        if (typeof metadata["end_time"] === "number") {
          return metadata["end_time"];
        }
        break;
      default:
        return assertUnreachable(timeKey);
    }
    if (!(traceSpan instanceof AgentTraceNode)) {
      return undefined;
    }
    const spans = reverse ? [...traceSpan.spans].reverse() : traceSpan.spans;
    for (const span of spans) {
      const chunks = reverse ? [...span.chunks].reverse() : span.chunks;
      for (const traceData of chunks) {
        const traceEvent = getEventType(traceData);
        if (traceEvent == null) {
          continue;
        }
        const eventData =
          getObjectDataFromUnknown({ data: traceData, key: traceEvent }) ?? {};
        // Check model invocation output
        if ("modelInvocationOutput" in eventData) {
          const modelInvocationOutput: Record<string, unknown> =
            getObjectDataFromUnknown({
              data: eventData,
              key: "modelInvocationOutput",
            }) ?? {};
          const metadataObject =
            getObjectDataFromUnknown({
              data: modelInvocationOutput,
              key: "metadata",
            }) ?? {};
          const { startTime, endTime } =
            getStartAndEndTimeFromMetadata(metadataObject);
          if (timeKey === "start" && startTime !== undefined) {
            return startTime;
          } else if (timeKey === "end" && endTime !== undefined) {
            return endTime;
          }
        }
        // Check observation
        if ("observation" in eventData) {
          const observation =
            getObjectDataFromUnknown({ data: eventData, key: "observation" }) ??
            {};
          const { startTime, endTime } =
            getStartAndEndTimeFromMetadata(observation);
          if (timeKey === "start" && startTime !== undefined) {
            return startTime;
          } else if (timeKey === "end" && endTime !== undefined) {
            return endTime;
          }
        }
      }
      // Recursively check nested nodes
      if (span instanceof AgentTraceNode) {
        const nestedTime = this.fetchSpanTime({
          traceSpan: span,
          timeKey,
          reverse,
          metadata,
        });
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
  private createChainSpan({
    parentSpan,
    attributes,
    name,
    startTime,
  }: {
    parentSpan: Span;
    attributes: Attributes;
    name: string;
    startTime?: number;
  }): Span {
    // Create the span with appropriate context
    const span = this.oiTracer.startSpan(
      name,
      {
        startTime: startTime ? startTime : undefined,
        attributes,
      },
      trace.setSpan(context.active(), parentSpan),
    );
    return span;
  }
}
