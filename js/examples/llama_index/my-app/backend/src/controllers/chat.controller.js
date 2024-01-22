// require("./tracer");
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
import { streamToResponse } from "ai";
import { OpenAI } from "llamaindex";
import { createChatEngine } from "./engine";
import { LlamaIndexStream } from "./llamaindex-stream";
const convertMessageContent = (textMessage, imageUrl) => {
    if (!imageUrl)
        return textMessage;
    return [
        {
            type: "text",
            text: textMessage,
        },
        {
            type: "image_url",
            image_url: {
                url: imageUrl,
            },
        },
    ];
};
export const chat = (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    try {
        const { messages, data } = req.body;
        const userMessage = messages.pop();
        if (!messages || !userMessage || userMessage.role !== "user") {
            return res.status(400).json({
                error: "messages are required in the request body and the last message must be from the user",
            });
        }
        const llm = new OpenAI({
            model: process.env.MODEL || "gpt-3.5-turbo",
        });
        const chatEngine = yield createChatEngine(llm);
        // Convert message content from Vercel/AI format to LlamaIndex/OpenAI format
        const userMessageContent = convertMessageContent(userMessage.content, data === null || data === void 0 ? void 0 : data.imageUrl);
        // Calling LlamaIndex's ChatEngine to get a streamed response
        const response = yield chatEngine.chat({
            message: userMessageContent,
            chatHistory: messages,
            stream: true,
        });
        // Return a stream, which can be consumed by the Vercel/AI client
        const { stream, data: streamData } = LlamaIndexStream(response, {
            parserOptions: {
                image_url: data === null || data === void 0 ? void 0 : data.imageUrl,
            },
        });
        // Pipe LlamaIndexStream to response
        const processedStream = stream.pipeThrough(streamData.stream);
        return streamToResponse(processedStream, res, {
            headers: {
                // response MUST have the `X-Experimental-Stream-Data: 'true'` header
                // so that the client uses the correct parsing logic, see
                // https://sdk.vercel.ai/docs/api-reference/stream-data#on-the-server
                "X-Experimental-Stream-Data": "true",
                "Content-Type": "text/plain; charset=utf-8",
                "Access-Control-Expose-Headers": "X-Experimental-Stream-Data",
            },
        });
    }
    catch (error) {
        console.error("[LlamaIndex]", error);
        return res.status(500).json({
            error: error.message,
        });
    }
});
