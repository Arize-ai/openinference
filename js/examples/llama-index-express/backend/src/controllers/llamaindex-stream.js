var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
import { createCallbacksTransformer, createStreamDataTransformer, experimental_StreamData, trimStartOfStreamHelper, } from "ai";
function createParser(res, data, opts) {
    const it = res[Symbol.asyncIterator]();
    const trimStartOfStream = trimStartOfStreamHelper();
    return new ReadableStream({
        start() {
            // if image_url is provided, send it via the data stream
            if (opts === null || opts === void 0 ? void 0 : opts.image_url) {
                const message = {
                    type: "image_url",
                    image_url: {
                        url: opts.image_url,
                    },
                };
                data.append(message);
            }
            else {
                data.append({}); // send an empty image response for the user's message
            }
        },
        pull(controller) {
            var _a;
            return __awaiter(this, void 0, void 0, function* () {
                const { value, done } = yield it.next();
                if (done) {
                    controller.close();
                    data.append({}); // send an empty image response for the assistant's message
                    data.close();
                    return;
                }
                const text = trimStartOfStream((_a = value.response) !== null && _a !== void 0 ? _a : "");
                if (text) {
                    controller.enqueue(text);
                }
            });
        },
    });
}
export function LlamaIndexStream(res, opts) {
    const data = new experimental_StreamData();
    return {
        stream: createParser(res, data, opts === null || opts === void 0 ? void 0 : opts.parserOptions)
            .pipeThrough(createCallbacksTransformer(opts === null || opts === void 0 ? void 0 : opts.callbacks))
            .pipeThrough(createStreamDataTransformer(true)),
        data,
    };
}
