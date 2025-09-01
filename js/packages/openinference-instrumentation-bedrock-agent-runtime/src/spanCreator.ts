import {
  getAttributesFromModelInvocationInput,
  getAttributesFromModelInvocationOutput,
  getAttributesFromInvocationInput,
  getAttributesFromObservation,
  getFailureTraceAttributes,
  getMetadataAttributes,
  getParentInputAttributesFromInvocationInput,
  extractMetadataAttributesFromObservation,
  // getStartAndEndTimeFromMetadata,
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
import { InvocationType } from "./attributes/types";
import { TraceEventType } from "./attributes/constants";
import { isArrayOfObjectWithStringKeys } from "./utils/typeUtils";
import { isAttributeValue } from "@opentelemetry/core";

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
      const { attributes, name, timingData, statusCode } =
        this.prepareSpanAttributes(traceSpan);

      const startTime = this.fetchSpanStartTime({
        traceSpan,
        timingData: timingData,
      });

      const span = this.createChainSpan({
        parentSpan,
        attributes,
        name,
        startTime,
      });

      span.setStatus({ code: statusCode });

      // Process child spans recursively
      if (traceSpan instanceof AgentTraceNode) {
        this.createSpans({
          parentSpan: span,
          traceNode: traceSpan,
        });
      }

      const endTime = this.fetchSpanEndTime({
        traceSpan,
        timingData: timingData,
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
    timingData: StringKeyedObject;
    statusCode: SpanStatusCode;
  } {
    let attributes: Attributes = {};
    let metadata: StringKeyedObject = {};
    let name: string | null = null;
    let finalStatusCode = SpanStatusCode.OK;

    if (traceSpan instanceof AgentTraceNode && traceSpan.chunks.length === 0) {
      name = traceSpan.nodeType;
      attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] =
        OpenInferenceSpanKind.CHAIN;
      if (traceSpan.nodeType === "agent-collaborator") {
        attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] =
          OpenInferenceSpanKind.AGENT;
      } else if (traceSpan.nodeType === "guardrailTrace") {
        const parts = traceSpan.nodeTraceId.split("-");
        const preOrPost = parts.at(-1) ?? "unknown";
        name = preOrPost + "GuardrailTrace";
      }
      const inputParentAttributes =
        this.getSpanInputAttributesFromNestedTraceNodes({
          traceNode: traceSpan,
        });
      const { attributes: outputAttributes } =
        this.getSpanOutputAttributesFromNestedTraceNodes({
          traceNode: traceSpan,
        });

      attributes = {
        ...attributes,
        ...outputAttributes,
        ...inputParentAttributes,
      };

      const rawMetadata = getRawMetadataFromTraceSpan(traceSpan);
      metadata = {
        ...metadata,
        ...rawMetadata,
      };
    } else if (Array.isArray(traceSpan.chunks) && traceSpan.chunks.length > 0) {
      for (const traceData of traceSpan.chunks) {
        const traceEventType = getEventType(traceData);
        if (traceEventType == null) {
          continue;
        }

        const {
          spanNameAndKind,
          modelInvocationAttributes,
          invocationAttributes,
          outputAttributes,
          outputMetadata,
          observationAttributes,
          observationMetadata,
          rationaleMetadata,
          failureTraceAttributes,
          failureTraceMetadata,
          guardrailTraceMetadata,
          statusCode,
        } = this.processTraceEvent(traceEventType, traceData, traceSpan);

        if (statusCode === SpanStatusCode.ERROR) {
          // Set final status code to error if any chunk has an error
          finalStatusCode = SpanStatusCode.ERROR;
        }

        name = spanNameAndKind.name;
        attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] =
          spanNameAndKind.spanKind;

        // Merge attributes from all chunks
        attributes = {
          ...attributes,
          ...modelInvocationAttributes,
          ...outputAttributes,
          ...observationAttributes,
          ...failureTraceAttributes,
          ...invocationAttributes,
        };

        metadata = {
          ...metadata,
          ...outputMetadata,
          ...observationMetadata,
          ...failureTraceMetadata,
          ...rationaleMetadata,
          ...guardrailTraceMetadata,
        };
      }

      const { totalTimeMs, startTime, endTime, ...restOfMetadata } = metadata;

      return {
        attributes: {
          ...attributes,
          metadata: safelyJSONStringify(restOfMetadata) ?? undefined,
        },
        timingData: {
          totalTimeMs,
          startTime,
          endTime,
        },
        name: name ?? "LLM",
        statusCode: finalStatusCode,
      };
    }
    return {
      attributes: attributes,
      timingData: metadata,
      name: name ?? "LLM",
      statusCode: finalStatusCode,
    };
  }

  private getSpanKindAndNameFromRoutingClassifierEventData(
    eventData: StringKeyedObject,
  ): { spanKind: OpenInferenceSpanKind; name: string } {
    const invocationInput = getObjectDataFromUnknown({
      data: eventData,
      key: "invocationInput",
    });

    const agentCollaboratorInvocationInput = getObjectDataFromUnknown({
      data: invocationInput,
      key: "agentCollaboratorInvocationInput",
    });
    if (
      agentCollaboratorInvocationInput?.agentCollaboratorName &&
      typeof agentCollaboratorInvocationInput.agentCollaboratorName === "string"
    ) {
      return {
        spanKind: OpenInferenceSpanKind.AGENT,
        name: `agent-collaborator[${agentCollaboratorInvocationInput.agentCollaboratorName}]`,
      };
    } else {
      const observation = getObjectDataFromUnknown({
        data: eventData,
        key: "observation",
      });
      const agentCollaboratorInvocationOutput = getObjectDataFromUnknown({
        data: observation,
        key: "agentCollaboratorInvocationOutput",
      });
      const name =
        typeof agentCollaboratorInvocationOutput?.agentCollaboratorName ===
        "string"
          ? agentCollaboratorInvocationOutput.agentCollaboratorName
          : "UNKNOWN";
      return {
        spanKind: OpenInferenceSpanKind.AGENT,
        name: `agent-collaborator[${name}]`,
      };
    }
  }

  private processTraceEvent(
    traceEventType: TraceEventType,
    traceData: StringKeyedObject,
    traceSpan: AgentTraceNode | AgentChunkSpan,
  ): {
    spanNameAndKind: { spanKind: OpenInferenceSpanKind; name: string };
    modelInvocationAttributes: Attributes;
    invocationAttributes: Attributes;
    outputAttributes: Attributes;
    outputMetadata: StringKeyedObject | null;
    observationAttributes: Attributes;
    observationMetadata: StringKeyedObject | null;
    rationaleMetadata: StringKeyedObject;
    failureTraceAttributes: Attributes;
    failureTraceMetadata: StringKeyedObject;
    guardrailTraceMetadata: GuardrailTraceMetadata | null;
    statusCode: SpanStatusCode;
  } {
    let defaultResult = {
      spanNameAndKind: { spanKind: OpenInferenceSpanKind.LLM, name: "LLM" },
      modelInvocationAttributes: {},
      invocationAttributes: {},
      outputAttributes: {},
      outputMetadata: null,
      observationAttributes: {},
      observationMetadata: null,
      rationaleMetadata: {},
      failureTraceAttributes: {},
      failureTraceMetadata: {},
      guardrailTraceMetadata: null,
      statusCode: SpanStatusCode.OK,
    };
    const eventData = traceEventType
      ? (getObjectDataFromUnknown({
          data: traceData,
          key: traceEventType,
        }) ?? {})
      : {};

    switch (traceEventType) {
      case "orchestrationTrace": {
        const kindAndName =
          this.getSpanKindAndNameFromOrchestrationEventData(eventData);
        const processModelInvocationOutputResult =
          this.processModelInvocationOutput(eventData);
        const processObservationResult = this.processObservation(eventData);
        return {
          ...defaultResult,
          spanNameAndKind: kindAndName,
          modelInvocationAttributes:
            this.processModelInvocationInput(eventData),
          invocationAttributes: this.processInvocationInput(eventData),
          outputAttributes: processModelInvocationOutputResult.attributes,
          outputMetadata: processModelInvocationOutputResult.metadata,
          observationAttributes: processObservationResult.attributes,
          observationMetadata: processObservationResult.metadata,
          rationaleMetadata: this.processRationaleMetadata(eventData),
        };
      }
      case "failureTrace": {
        const kindAndName = this.getSpanKindAndNameForFailureTrace();
        const processFailureTraceResult = this.processFailureTrace(eventData);
        return {
          ...defaultResult,
          spanNameAndKind: kindAndName,
          failureTraceAttributes: processFailureTraceResult.attributes,
          failureTraceMetadata: processFailureTraceResult.metadata ?? {},
        };
      }
      case "guardrailTrace": {
        const traceResult = this.processGuardrailTrace(eventData);
        let statusCode = SpanStatusCode.OK;
        const interveningGuardrails =
          traceResult.guardrailTraceMetadata?.intervening_guardrails || [];
        if (
          isArrayOfObjectWithStringKeys(interveningGuardrails) &&
          isBlockedGuardrail(interveningGuardrails)
        ) {
          statusCode = SpanStatusCode.ERROR;
        }

        return {
          ...defaultResult,
          spanNameAndKind: {
            spanKind: OpenInferenceSpanKind.GUARDRAIL,
            name: "Guardrails",
          },
          guardrailTraceMetadata: traceResult.guardrailTraceMetadata,
          statusCode: statusCode,
        };
      }
      case "preProcessingTrace":
      case "postProcessingTrace": {
        const kindAndName =
          this.getSpanKindAndNameFromOrchestrationEventData(eventData);
        return {
          ...defaultResult,
          spanNameAndKind: kindAndName,
          invocationAttributes: this.processInvocationInput(eventData),
          outputAttributes:
            this.processModelInvocationOutput(eventData).attributes,
          modelInvocationAttributes:
            this.processModelInvocationInput(eventData),
        };
      }
      case "routingClassifierTrace": {
        let kindAndName: { spanKind: OpenInferenceSpanKind; name: string };
        if (
          traceSpan instanceof AgentTraceNode &&
          traceSpan.nodeType === "agent-collaborator"
        ) {
          kindAndName =
            this.getSpanKindAndNameFromRoutingClassifierEventData(eventData);
          const inputParentAttributes =
            this.getSpanInputAttributesFromNestedTraceNodes({
              traceNode: traceSpan,
            });
          const { attributes: outputAttributes } =
            this.getSpanOutputAttributesFromNestedTraceNodes({
              traceNode: traceSpan,
            });
          defaultResult = {
            ...defaultResult,
            ...inputParentAttributes,
            ...outputAttributes,
          };
        } else {
          kindAndName = {
            spanKind: OpenInferenceSpanKind.LLM,
            name: "Routing Classifer",
          };
        }
        const processObservationResult = this.processObservation(eventData);
        const processModelInvocationOutputResult =
          this.processModelInvocationOutput(eventData);
        return {
          ...defaultResult,
          spanNameAndKind: kindAndName,
          modelInvocationAttributes:
            this.processModelInvocationInput(eventData),
          outputAttributes: processModelInvocationOutputResult.attributes,
          outputMetadata: processModelInvocationOutputResult.metadata,
          observationAttributes: processObservationResult.attributes,
          observationMetadata: processObservationResult.metadata,
          invocationAttributes: this.processInvocationInput(eventData),
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
    const input = getObjectDataFromUnknown({
      data: eventData,
      key: "invocationInput",
    });

    let type = input?.invocationType as InvocationType;

    if (!input) {
      const observation = getObjectDataFromUnknown({
        data: eventData,
        key: "observation",
      });

      if (!observation) {
        return { spanKind: OpenInferenceSpanKind.LLM, name: "LLM" };
      }

      const finalResponse = getObjectDataFromUnknown({
        data: observation,
        key: "finalResponse",
      });

      if (finalResponse) {
        return { spanKind: OpenInferenceSpanKind.LLM, name: "LLM" };
      }

      type = observation?.type as InvocationType;
    }

    switch (type) {
      case "AGENT_COLLABORATOR":
        return {
          spanKind: OpenInferenceSpanKind.AGENT,
          name: `agent_collaborator[${input?.agentCollaboratorName}]`,
        };
      case "ACTION_GROUP":
        return {
          spanKind: OpenInferenceSpanKind.TOOL,
          name: "action_group",
        };
      case "ACTION_GROUP_CODE_INTERPRETER":
        return {
          spanKind: OpenInferenceSpanKind.TOOL,
          name: "action_group_code_interpreter",
        };
      case "KNOWLEDGE_BASE":
        return {
          spanKind: OpenInferenceSpanKind.RETRIEVER,
          name: "knowledge_base",
        };
      default:
        return {
          spanKind: OpenInferenceSpanKind.TOOL,
          name: "TOOL",
        };
    }
  }

  private getSpanKindAndNameForGuardrailTrace(): {
    spanKind: OpenInferenceSpanKind;
    name: string;
  } {
    return {
      spanKind: OpenInferenceSpanKind.GUARDRAIL,
      name: "Guardrails",
    };
  }

  private getSpanKindAndNameForFailureTrace(): {
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
   * Processes rationale metadata.
   * @param eventData The event data containing rationale information.
   * @returns Rationale attributes.
   */
  private processRationaleMetadata(
    eventData: StringKeyedObject,
  ): StringKeyedObject {
    const rationaleText = getObjectDataFromUnknown({
      data: eventData,
      key: "rationale",
    });
    if (!rationaleText) return {};
    const rationale = isAttributeValue(rationaleText)
      ? rationaleText
      : (safelyJSONStringify(rationaleText) ?? undefined);

    return { rationale: rationale };
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

    const isInterveningGuardrail = eventData.action === "INTERVENED";
    const finalMetadata: GuardrailTraceMetadata = {
      intervening_guardrails: isInterveningGuardrail ? [eventData] : [],
      non_intervening_guardrails: isInterveningGuardrail ? [] : [eventData],
      ...metadata,
    };

    return { guardrailTraceMetadata: finalMetadata };
  }

  /**
   * Gets parent span input attributes by extracting from nested trace nodes.
   * Recursively checks nested nodes for input attributes.
   * Mutates the attributes object in place.
   * @param attributes The attributes object to update.
   * @param traceNode The agent trace node or chunk span to process.
   */
  private getSpanInputAttributesFromNestedTraceNodes({
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
        newAttributes = this.getSpanInputAttributesFromNestedTraceNodes({
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
  private getSpanOutputAttributesFromNestedTraceNodes({
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
        const { attributes } = this.getSpanOutputAttributesFromNestedTraceNodes(
          {
            accumulatedAttributes: newAttributes,
            traceNode: span,
          },
        );
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
    timingData,
  }: {
    traceSpan: AgentTraceNode | AgentChunkSpan;
    timingData: StringKeyedObject;
  }): number | undefined {
    return this.fetchSpanTime({
      traceSpan,
      timeKey: "start",
      reverse: false,
      timingData,
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
    timingData,
  }: {
    traceSpan: AgentTraceNode | AgentChunkSpan;
    timingData: StringKeyedObject;
  }): number | undefined {
    return this.fetchSpanTime({
      traceSpan,
      timeKey: "end",
      reverse: true,
      timingData,
    });
  }

  /**
   * Fetches span time (start or end) from attributes or trace span.
   * Recursively searches for timestamp information in the trace data.
   * @param attributes The attributes object that may contain timing metadata.
   * @param traceSpan The trace span to extract time from.
   * @param timeKey The key to look for ('startTime' or 'endTIme').
   * @param reverse Whether to traverse spans and chunks in reverse order.
   * @returns The timestamp in nanoseconds if found, undefined otherwise.
   */
  private fetchSpanTime({
    traceSpan,
    timeKey,
    reverse,
    timingData,
  }: {
    traceSpan: AgentTraceNode | AgentChunkSpan;
    timingData: StringKeyedObject;
    timeKey: "start" | "end";
    reverse: boolean;
  }): number | undefined {
    // If we've aleady found the time in the metadata, return it
    // We don't really need to store this in the metadata of the span but with all of the traversing of various attributes this does allow us to get it from various places then use it when creating the span
    switch (timeKey) {
      case "start":
        if (typeof timingData["startTime"] === "number") {
          return timingData["startTime"];
        } else if (timingData["startTime"] instanceof Date) {
          return timingData["startTime"].getTime(); // Convert ms to ns
        }
        break;
      case "end":
        if (typeof timingData["endTime"] === "number") {
          return timingData["endTime"];
        } else if (timingData["endTime"] instanceof Date) {
          return timingData["endTime"].getTime(); // Convert ms to ns
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
      // Always reverse the chunks so we can get the start and end time from the last event which contains start and end time info
      const chunks = [...span.chunks].reverse();
      for (const traceData of chunks) {
        const traceEventType = getEventType(traceData);
        if (traceEventType == null) {
          continue;
        }
      }
      // Recursively check nested nodes
      if (span instanceof AgentTraceNode) {
        const nestedTime = this.fetchSpanTime({
          traceSpan: span,
          timeKey,
          reverse,
          timingData,
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
function getRawMetadataFromTraceSpan(
  traceSpan: AgentTraceNode,
): StringKeyedObject {
  const rawMetadata: StringKeyedObject = {};

  const mergeMetadata = (metadata: StringKeyedObject) => {
    for (const [key, value] of Object.entries(metadata)) {
      rawMetadata[key] = value;
    }
  };

  for (const span of traceSpan.spans) {
    for (const chunk of span.chunks) {
      const traceEventType = getEventType(chunk);
      if (traceEventType == null) {
        continue;
      }

      const eventData = traceEventType
        ? getObjectDataFromUnknown({
            data: chunk,
            key: traceEventType,
          })
        : {};
      switch (traceEventType) {
        case "guardrailTrace":
        case "failureTrace": {
          const metadata =
            getObjectDataFromUnknown({ data: eventData, key: "metadata" }) ??
            {};
          mergeMetadata(metadata);
          break;
        }
        case "routingClassifierTrace": {
          const observation = getObjectDataFromUnknown({
            data: eventData,
            key: "observation",
          });
          if (observation) {
            const finalResponse = getObjectDataFromUnknown({
              data: observation,
              key: "finalResponse",
            });
            if (finalResponse) {
              const agentOutput = getObjectDataFromUnknown({
                data: observation,
                key: "agentCollaboratorInvocationOutput",
              });
              const metadata =
                getObjectDataFromUnknown({
                  data: agentOutput,
                  key: "metadata",
                }) ?? {};
              mergeMetadata(metadata);
            } else {
              const metadata =
                getObjectDataFromUnknown({
                  data: finalResponse,
                  key: "metadata",
                }) ?? {};
              mergeMetadata(metadata);
            }
          }
          break;
        }
        case "preProcessingTrace":
        case "postProcessingTrace":
        case "orchestrationTrace": {
          const observation = getObjectDataFromUnknown({
            data: eventData,
            key: "observation",
          });
          if (observation) {
            const finalResponse = getObjectDataFromUnknown({
              data: observation,
              key: "finalResponse",
            });
            const metadata =
              getObjectDataFromUnknown({
                data: finalResponse,
                key: "metadata",
              }) ?? {};
            mergeMetadata(metadata);
          } else {
            const modelOutput = getObjectDataFromUnknown({
              data: eventData,
              key: "modelInvocationOutput",
            });
            const metadata =
              getObjectDataFromUnknown({
                data: modelOutput,
                key: "metadata",
              }) ?? {};
            mergeMetadata(metadata);
          }
          break;
        }
        default: {
          assertUnreachable(traceEventType);
        }
      }
    }
  }
  return rawMetadata;
}
