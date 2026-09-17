import type { Attributes } from "@opentelemetry/api";
import type {
  ImageEditCompletedEvent,
  ImageGenCompletedEvent,
  ImagesResponse,
} from "openai/resources/images";

import type { TraceConfigOptions } from "@arizeai/openinference-core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

function imageAttributeKey(namespace: string, index: number): string {
  return `${namespace}.${index}.${SemanticConventions.IMAGE_URL}`;
}

function mediaTypeFromFormat(format: unknown): string | undefined {
  if (typeof format !== "string") return undefined;
  const normalized = format.toLowerCase();
  if (normalized === "jpg") return "image/jpeg";
  if (["png", "jpeg", "webp", "gif"].includes(normalized)) {
    return `image/${normalized}`;
  }
  return undefined;
}

function mediaTypeFromFilename(filename: string | undefined): string | undefined {
  if (!filename) return undefined;
  const extension = filename.toLowerCase().split(/[?#]/, 1)[0].split(".").pop();
  return mediaTypeFromFormat(extension);
}

function sniffImageMediaType(bytes: Uint8Array): string | undefined {
  if (
    bytes.length >= 8 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a
  ) {
    return "image/png";
  }
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return "image/jpeg";
  }
  if (bytes.length >= 6) {
    const signature = String.fromCharCode(...bytes.subarray(0, 6));
    if (signature === "GIF87a" || signature === "GIF89a") return "image/gif";
  }
  if (
    bytes.length >= 12 &&
    String.fromCharCode(...bytes.subarray(0, 4)) === "RIFF" &&
    String.fromCharCode(...bytes.subarray(8, 12)) === "WEBP"
  ) {
    return "image/webp";
  }
  return undefined;
}

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

export function imageBase64ToDataURL(base64: string, imageFormat?: unknown): string {
  if (base64.startsWith("data:image/")) return base64;
  return `data:${mediaTypeFromFormat(imageFormat) ?? "image/png"};base64,${base64}`;
}

export function getImagesResponseAttributes(
  response: ImagesResponse,
  requestFormat?: unknown,
): Attributes {
  const attributes: Attributes = {};
  const imageFormat = response.output_format ?? requestFormat;
  response.data?.forEach((image, index) => {
    const imageUrl =
      image.url ?? (image.b64_json ? imageBase64ToDataURL(image.b64_json, imageFormat) : undefined);
    if (imageUrl) {
      attributes[imageAttributeKey(SemanticConventions.OUTPUT_IMAGES, index)] = imageUrl;
    }
  });
  return attributes;
}

export function getCompletedImageEventAttributes(
  event: ImageEditCompletedEvent | ImageGenCompletedEvent,
): Attributes {
  return {
    [imageAttributeKey(SemanticConventions.OUTPUT_IMAGES, 0)]: imageBase64ToDataURL(
      event.b64_json,
      event.output_format,
    ),
  };
}

async function uploadToDataURL(upload: unknown): Promise<string | undefined> {
  try {
    let bytes: Uint8Array;
    let declaredMediaType: string | undefined;
    let filename: string | undefined;

    if (typeof Response !== "undefined" && upload instanceof Response) {
      const clone = upload.clone();
      bytes = new Uint8Array(await clone.arrayBuffer());
      declaredMediaType = clone.headers.get("content-type") ?? undefined;
      filename = clone.url;
    } else if (typeof Blob !== "undefined" && upload instanceof Blob) {
      bytes = new Uint8Array(await upload.arrayBuffer());
      declaredMediaType = upload.type || undefined;
      if ("name" in upload && typeof upload.name === "string") filename = upload.name;
    } else if (upload != null && typeof upload === "object" && "path" in upload) {
      const path = Reflect.get(upload, "path");
      if (typeof path !== "string") return undefined;
      // Read from the path separately so the SDK still owns the one-shot stream.
      const { readFile } = await import("node:fs/promises");
      bytes = await readFile(path);
      filename = path;
    } else {
      // Unknown one-shot async iterables cannot be inspected without consuming
      // the user's upload before the OpenAI SDK can build the request.
      return undefined;
    }

    const mediaType =
      (declaredMediaType?.startsWith("image/") ? declaredMediaType : undefined) ??
      sniffImageMediaType(bytes) ??
      mediaTypeFromFilename(filename) ??
      "image/png";
    return `data:${mediaType};base64,${bytesToBase64(bytes)}`;
  } catch {
    return undefined;
  }
}

export async function getInputImageAttributes(
  uploads: ReadonlyArray<unknown>,
  traceConfig: TraceConfigOptions = {},
): Promise<Attributes> {
  if (traceConfig.hideInputs || traceConfig.hideInputImages) return {};
  const flattenedUploads = uploads.flatMap((upload) => (Array.isArray(upload) ? upload : [upload]));
  const dataURLs = await Promise.all(flattenedUploads.map(uploadToDataURL));
  const attributes: Attributes = {};
  let imageIndex = 0;
  for (const dataURL of dataURLs) {
    if (!dataURL) continue;
    attributes[imageAttributeKey(SemanticConventions.INPUT_IMAGES, imageIndex)] = dataURL;
    imageIndex++;
  }
  return attributes;
}
