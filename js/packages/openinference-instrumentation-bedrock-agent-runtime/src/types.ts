
/**
 * A string keyed object with unknown values.
 */
export type StringKeyedObject = Record<string, unknown>;

export interface GuardrailTraceMetadata extends StringKeyedObject {
    intervening_guardrails?: StringKeyedObject[];
    non_intervening_guardrails?: StringKeyedObject[];
}
