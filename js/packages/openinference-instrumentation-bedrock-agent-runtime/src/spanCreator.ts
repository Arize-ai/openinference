import {
  getAttributesFromModelInvocationInput,
  getAttributesFromModelInvocationOutput,
  getAttributesFromInvocationInput,
  getAttributesFromObservation,
  getAttributesFromRationale,
  getFailureTraceAttributes,
  getGuardrailTraceMetadata,
  getMetadataAttributes,
  getParentInputAttributesFromInvocationInput,
  extractMetadataAttributesFromObservation,
  getStartAndEndTimeFromMetadata,
  getStringAttributeValueFromUnknown,
  getInputMessagesObject,
  isBlockedGuardrail,
  getEventType,
} from "./attributes/attributeExtractionUtils";
import { AgentChunkSpan } from "./collector/agentChunkSpan";
import { AgentTraceNode } from "./collector/agentTraceNode";
import { OITracer } from "@arizeai/openinference-core";
import { Attributes, SpanStatusCode } from "@opentelemetry/api";
import { trace, context, Span } from "@opentelemetry/api";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";
import {
  safelyJSONStringify,
  assertUnreachable,
} from "@arizeai/openinference-core";
import { GuardrailTraceMetadata, StringKeyedObject } from "./types";
import { getObjectDataFromUnknown } from "./utils/jsonUtils";
import {
  getInputAttributes,
  getOutputAttributes,
} from "./attributes/attributeUtils";
import { BaseInvocationInput } from "./attributes/types";
import { TraceEventType } from "./attributes/constants";
import { isArrayOfObjectWithStringKeys } from "./utils/typeUtils";

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

      // Create span with appropriate timing
      const startTime = this.fetchSpanStartTime({
        traceSpan,
        metadata: rawMetadata,
      });

      const span = this.createChainSpan({
        parentSpan,
        attributes,
        name,
        startTime,
      });

      let statusCode = SpanStatusCode.OK;

      // Set error code if guardrail is blocked
      if (
        attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
        OpenInferenceSpanKind.GUARDRAIL
      ) {
        const interveningGuardrails = rawMetadata?.intervening_guardrails || [];
        if (
          isArrayOfObjectWithStringKeys(interveningGuardrails) &&
          isBlockedGuardrail(interveningGuardrails)
        ) {
          statusCode = SpanStatusCode.ERROR;
        }
      }
      span.setStatus({ code: statusCode });

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

    if (
      traceSpan instanceof AgentTraceNode &&
      traceSpan.nodeType != null &&
      traceSpan.chunks.length === 0
    ) {
      let spanKind = OpenInferenceSpanKind.CHAIN;
      if (traceSpan.nodeType === "agent-collaborator") {
        spanKind = OpenInferenceSpanKind.AGENT;
      }

      const inputParentAttributes = this.getParentSpanInputAttributes({
        traceNode: traceSpan,
      });
      const { attributes: outputAttributes } =
        this.getParentSpanOutputAttributes({
          traceNode: traceSpan,
        });

      return {
        attributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: spanKind,
          ...inputParentAttributes,
          ...outputAttributes,
        },
        rawMetadata: {},
        name: traceSpan.nodeType ?? "CHAIN",
      };
    }

    if (Array.isArray(traceSpan.chunks) && traceSpan.chunks.length > 0) {
      // Process the first chunk to determine the span kind and name
      // We should expect that all chunks have the same event type
      const firstChunk = traceSpan.chunks[0];
      const traceEventType = getEventType(firstChunk);
      const eventData = traceEventType
        ? (getObjectDataFromUnknown({
            data: firstChunk,
            key: traceEventType,
          }) ?? {})
        : {};
      let kindAndName = {
        spanKind: OpenInferenceSpanKind.LLM,
        name: "UNKNOWN",
      };

      switch (traceEventType) {
        case "guardrailTrace":
          kindAndName = this.getSpanKindAndNameFromGuardrailEventData();
          break;
        case "failureTrace":
          kindAndName = this.getSpanKindAndNameFromFailureEventData();
          break;
        case "orchestrationTrace":
        case "preProcessingTrace":
        case "postProcessingTrace":
          kindAndName =
            this.getSpanKindAndNameFromOrchestrationEventData(eventData);
          break;
        case undefined:
          kindAndName = {
            spanKind: OpenInferenceSpanKind.LLM,
            name: "UNKNOWN",
          };
          break;
        default: {
          assertUnreachable(traceEventType);
        }
      }

      name = kindAndName.name;
      attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] =
        kindAndName.spanKind;

      for (const traceData of traceSpan.chunks) {
        const traceEventType = getEventType(traceData);
        if (traceEventType == null) {
          continue;
        }
        const eventData =
          getObjectDataFromUnknown({ data: traceData, key: traceEventType }) ??
          {};
        const {
          modelInvocationAttributes,
          invocationAttributes,
          outputAttributes,
          outputMetadata,
          observationAttributes,
          observationMetadata,
          rationaleAttributes,
          failureTraceAttributes,
          failureTraceMetadata,
          guardrailTraceMetadata,
        } = this.processTraceEvent(traceEventType, eventData);

        if (guardrailTraceMetadata) {
          if (!isArrayOfObjectWithStringKeys(metadata.intervening_guardrails)) {
            metadata.intervening_guardrails = [];
          }
          if (
            !isArrayOfObjectWithStringKeys(metadata.non_intervening_guardrails)
          ) {
            metadata.non_intervening_guardrails = [];
          }

          const newIntervening =
            guardrailTraceMetadata.intervening_guardrails || [];
          const newNonIntervening =
            guardrailTraceMetadata.non_intervening_guardrails || [];

          if (newIntervening.length > 0) {
            (metadata.intervening_guardrails as StringKeyedObject[]).push(
              ...newIntervening,
            );
          }
          if (newNonIntervening.length > 0) {
            (metadata.non_intervening_guardrails as StringKeyedObject[]).push(
              ...newNonIntervening,
            );
          }
        }

        // Merge attributes from all chunks
        attributes = {
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

  private processTraceEvent(
    traceEventType: TraceEventType,
    eventData: StringKeyedObject,
  ): {
    modelInvocationAttributes: Attributes;
    invocationAttributes: Attributes;
    outputAttributes: Attributes;
    outputMetadata: StringKeyedObject | null;
    observationAttributes: Attributes;
    observationMetadata: StringKeyedObject | null;
    rationaleAttributes: Attributes;
    failureTraceAttributes: Attributes;
    failureTraceMetadata: StringKeyedObject;
    guardrailTraceMetadata: GuardrailTraceMetadata | null;
  } {
    const defaultResult = {
      modelInvocationAttributes: {} as Attributes,
      invocationAttributes: {} as Attributes,
      outputAttributes: {} as Attributes,
      outputMetadata: null as StringKeyedObject | null,
      observationAttributes: {} as Attributes,
      observationMetadata: null as StringKeyedObject | null,
      rationaleAttributes: {} as Attributes,
      failureTraceAttributes: {} as Attributes,
      failureTraceMetadata: {} as StringKeyedObject,
      guardrailTraceMetadata: null as GuardrailTraceMetadata | null,
    };

    switch (traceEventType) {
      case "orchestrationTrace": {
        const processModelInvocationInputResult =
          this.processModelInvocationOutput(eventData);
        const processObservationResult = this.processObservation(eventData);
        return {
          ...defaultResult,
          modelInvocationAttributes:
            this.processModelInvocationInput(eventData),
          invocationAttributes: this.processInvocationInput(eventData),
          outputAttributes: processModelInvocationInputResult.attributes,
          outputMetadata: processModelInvocationInputResult.metadata,
          observationAttributes: processObservationResult.attributes,
          observationMetadata: processObservationResult.metadata,
          rationaleAttributes: this.processRationale(eventData),
        };
      }

      case "failureTrace": {
        const processFailureTraceResult = this.processFailureTrace(eventData);
        return {
          ...defaultResult,
          failureTraceAttributes: processFailureTraceResult.attributes,
          failureTraceMetadata: processFailureTraceResult.metadata ?? {},
        };
      }

      case "guardrailTrace": {
        const traceResult = this.processGuardrailTrace(eventData);
        return {
          ...defaultResult,
          guardrailTraceMetadata: traceResult.guardrailTraceMetadata,
        };
      }

      case "preProcessingTrace": {
        return {
          ...defaultResult,
          invocationAttributes: this.processInvocationInput(eventData),
          outputAttributes:
            this.processModelInvocationOutput(eventData).attributes,
          modelInvocationAttributes:
            this.processModelInvocationInput(eventData),
        };
      }

      case "postProcessingTrace": {
        return {
          ...defaultResult,
          invocationAttributes: this.processInvocationInput(eventData),
          outputAttributes:
            this.processModelInvocationOutput(eventData).attributes,
          modelInvocationAttributes:
            this.processModelInvocationInput(eventData),
        };
      }
      default: {
        assertUnreachable(traceEventType);
      }
    }
  }

  /**
   * Processes model invocation input and updates attributes.
   * @param eventData The event data containing model invocation input.
   * @param attributes The attributes object to update.
   */
  private processModelInvocationInput(
    eventData: StringKeyedObject,
  ): Attributes {
    const modelInvocationInput = getObjectDataFromUnknown({
      data: eventData,
      key: "modelInvocationInput",
    });
    if (!modelInvocationInput) return {};
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
    const modelInvocationOutput = getObjectDataFromUnknown({
      data: eventData,
      key: "modelInvocationOutput",
    });
    if (!modelInvocationOutput) {
      return { attributes: {}, metadata: null };
    }
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

  /**
   * Gets span kind and name for orchestration event data (preProcessing, orchestration, postProcessing)
   */
  private getSpanKindAndNameFromOrchestrationEventData(
    eventData: StringKeyedObject,
  ): {
    spanKind: OpenInferenceSpanKind;
    name: string;
  } {
    const invocationInput = getObjectDataFromUnknown({
      data: eventData,
      key: "invocationInput",
    });
    if (!invocationInput?.invocationType) {
      return {
        spanKind: OpenInferenceSpanKind.LLM,
        name: "LLM",
      };
    }

    const { invocationType } = invocationInput as BaseInvocationInput;
    switch (invocationType) {
      case "AGENT_COLLABORATOR": {
        const agentCollaboratorName =
          getObjectDataFromUnknown({
            data: invocationInput,
            key: "agentCollaboratorInvocationInput",
          })?.agentCollaboratorName ?? "";
        return {
          spanKind: OpenInferenceSpanKind.AGENT,
          name: `${invocationType.toLowerCase()}[${agentCollaboratorName}]`,
        };
      }

      case "ACTION_GROUP":
        return {
          spanKind: OpenInferenceSpanKind.TOOL,
          name: invocationType.toLowerCase(),
        };

      case "ACTION_GROUP_CODE_INTERPRETER":
        return {
          spanKind: OpenInferenceSpanKind.TOOL,
          name: invocationType.toLowerCase(),
        };

      case "KNOWLEDGE_BASE":
        return {
          spanKind: OpenInferenceSpanKind.RETRIEVER,
          name: invocationType.toLowerCase(),
        };
      default:
        return {
          spanKind: OpenInferenceSpanKind.TOOL,
          name: "TOOL",
        };
    }
  }

  private getSpanKindAndNameFromGuardrailEventData(): {
    spanKind: OpenInferenceSpanKind;
    name: string;
  } {
    return {
      spanKind: OpenInferenceSpanKind.GUARDRAIL,
      name: "Guardrails",
    };
  }

  private getSpanKindAndNameFromFailureEventData(): {
    spanKind: OpenInferenceSpanKind;
    name: string;
  } {
    return {
      spanKind: OpenInferenceSpanKind.CHAIN,
      name: "Failure",
    };
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
    if (!invocationInput) return {};
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
    const observation = getObjectDataFromUnknown({
      data: eventData,
      key: "observation",
    });
    if (!observation) {
      return { attributes: {}, metadata: null };
    }
    return {
      attributes: getAttributesFromObservation(observation),
      metadata: extractMetadataAttributesFromObservation(observation),
    };
  }

  /**
   * Processes rationale data and extracts attributes.
   * @param eventData The event data containing rationale information.
   * @returns Rationale attributes.
   */
  private processRationale(eventData: StringKeyedObject): Attributes {
    const rationale = getObjectDataFromUnknown({
      data: eventData,
      key: "rationale",
    });
    if (!rationale) return {};
    return getAttributesFromRationale(rationale);
  }

  /**
   * Processes failure trace data and extracts attributes and metadata.
   * @param eventData The event data containing failure trace information.
   * @returns Object containing failure trace attributes and metadata.
   */
  private processFailureTrace(eventData: StringKeyedObject): {
    attributes: Attributes;
    metadata: StringKeyedObject | null;
  } {
    const failureTrace = getObjectDataFromUnknown({
      data: eventData,
      key: "failureTrace",
    });
    if (!failureTrace) {
      return { attributes: {}, metadata: null };
    }
    const failureTraceMetadata = getObjectDataFromUnknown({
      data: failureTrace,
      key: "metadata",
    });
    return {
      attributes: getFailureTraceAttributes(failureTrace),
      metadata: getMetadataAttributes(failureTraceMetadata ?? {}),
    };
  }

  /**
   * Processes guardrail trace data and extracts relevant metadata.
   * @param eventData The event data containing guardrail trace information.
   * @returns Object containing guardrail span metadata.
   */
  private processGuardrailTrace(eventData: StringKeyedObject): {
    guardrailTraceMetadata: GuardrailTraceMetadata;
  } {
    const metadata = getMetadataAttributes(
      getObjectDataFromUnknown({ data: eventData, key: "metadata" }) ?? {},
    );
    const guardrailMetadata = {
      ...getGuardrailTraceMetadata(eventData),
      ...metadata,
    };

    const isInterveningGuardrail = guardrailMetadata.action === "INTERVENED";
    const guardrailTraceMetadata: GuardrailTraceMetadata = {
      intervening_guardrails: isInterveningGuardrail ? [guardrailMetadata] : [],
      non_intervening_guardrails: isInterveningGuardrail
        ? []
        : [guardrailMetadata],
    };

    return { guardrailTraceMetadata };
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

    if (!(traceNode instanceof AgentTraceNode)) {
      return { ...newAttributes };
    }

    for (const span of traceNode.spans) {
      if (!span.chunks) continue;
      for (const traceData of span.chunks) {
        const traceEventType = getEventType(traceData);
        if (traceEventType == null) {
          continue;
        }
        const eventData =
          getObjectDataFromUnknown({ data: traceData, key: traceEventType }) ??
          {};

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
        const traceEventType = getEventType(traceData);
        if (traceEventType == null) {
          continue;
        }
        const eventData =
          getObjectDataFromUnknown({ data: traceData, key: traceEventType }) ??
          {};

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
        const traceEventType = getEventType(traceData);
        if (traceEventType == null) {
          continue;
        }
        const eventData =
          getObjectDataFromUnknown({ data: traceData, key: traceEventType }) ??
          {};

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
