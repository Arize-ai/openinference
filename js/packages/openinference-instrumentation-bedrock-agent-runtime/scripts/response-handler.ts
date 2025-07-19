import * as console from "node:console";
import {SpanStatusCode, Span} from "@opentelemetry/api";
import {MimeType, SemanticConventions,} from "@arizeai/openinference-semantic-conventions";


export class ResponseHandler {
    private outputChunks: string[] = [];
    private span: Span;

    constructor(span: Span) {
        this.span = span;
        console.log("Initialized.....");
        console.log(this.span);
    }

    consumeResponse(chunk: Uint8Array) {
        const text = Buffer.from(chunk).toString("utf8");
        this.outputChunks.push(text);
    }

    onComplete(): void {
        const finalOutput = this.outputChunks.join("");
        this.span.setAttributes({
            [SemanticConventions.OUTPUT_VALUE]: finalOutput,
            [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]: "assistant",
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]: finalOutput,
        });
        this.span.setStatus({code: SpanStatusCode.OK});
        this.span.end();
        console.log("Final output:", finalOutput);
    }

    onError(error: any): void {
        console.error("Stream error:", error);
        this.span.recordException(error);
        this.span.setStatus({code: SpanStatusCode.ERROR, message: error.message});
        this.span.end();
    }
}