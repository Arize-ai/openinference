import { type Attributes, diag } from "@opentelemetry/api";
import type {
  ImageEditCompletedEvent,
  ImageGenCompletedEvent,
  ImagesResponse,
} from "openai/resources/images";

import {
  generateTraceConfig,
  REDACTED_VALUE,
  type TraceConfigOptions,
} from "@arizeai/openinference-core";
import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

function getImageAttributeKey({ namespace, index }: { namespace: string; index: number }): string {
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

export function imageBase64ToDataURL({
  base64,
  imageFormat,
}: {
  base64: string;
  imageFormat?: unknown;
}): string {
  if (base64.startsWith("data:image/")) return base64;
  return `data:${mediaTypeFromFormat(imageFormat) ?? "image/png"};base64,${base64}`;
}

export function getImagesResponseAttributes({
  response,
  requestFormat,
}: {
  response: ImagesResponse;
  requestFormat?: unknown;
}): Attributes {
  const attributes: Attributes = {};
  const imageFormat = response.output_format ?? requestFormat;
  response.data?.forEach((image, index) => {
    const imageUrl =
      image.url ??
      (image.b64_json ? imageBase64ToDataURL({ base64: image.b64_json, imageFormat }) : undefined);
    if (imageUrl) {
      attributes[getImageAttributeKey({ namespace: SemanticConventions.OUTPUT_IMAGES, index })] =
        imageUrl;
    }
  });
  return attributes;
}

export function getCompletedImageEventAttributes(
  event: ImageEditCompletedEvent | ImageGenCompletedEvent,
): Attributes {
  return {
    [getImageAttributeKey({ namespace: SemanticConventions.OUTPUT_IMAGES, index: 0 })]:
      imageBase64ToDataURL({ base64: event.b64_json, imageFormat: event.output_format }),
  };
}

/**
 * Serializes an Images response for output.value without its base64 images. They are
 * recorded on output.images.*, where base64ImageMaxLength applies to them.
 */
export function stringifyImagesResponse(response: ImagesResponse): string {
  return JSON.stringify({
    ...response,
    data: response.data?.map(({ b64_json: _b64Json, ...image }) => image),
  });
}

/**
 * Serializes a completed stream event for output.value without its base64 image.
 */
export function stringifyCompletedImageEvent({
  b64_json: _b64Json,
  ...event
}: ImageEditCompletedEvent | ImageGenCompletedEvent): string {
  return JSON.stringify(event);
}

function getBase64Length(byteLength: number): number {
  return Math.ceil(byteLength / 3) * 4;
}

type UploadContents = {
  bytes: Uint8Array;
  declaredMediaType?: string;
  filename?: string;
};
type ReadUploadParams = {
  /** Whether the bytes would be redacted once encoded, so there is no need to read them. */
  isTooLong: (byteLength: number) => boolean;
};
type UploadReadResult = UploadContents | typeof REDACTED_VALUE | undefined;

/**
 * Reads a cloned response body, stopping as soon as it is too long.
 *
 * @param params.body - The body of the cloned response
 * @param params.isTooLong - Whether the bytes read so far would be redacted once encoded
 * @returns The body's bytes, or the redacted value if it is too long
 */
async function readBodyWithinLimit({
  body,
  isTooLong,
}: ReadUploadParams & { body: NonNullable<Response["body"]> }): Promise<
  Uint8Array | typeof REDACTED_VALUE
> {
  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let byteLength = 0;
  let readResult = await reader.read();
  while (!readResult.done) {
    byteLength += readResult.value.length;
    if (isTooLong(byteLength)) {
      // Not awaited: cancelling a clone only settles once the SDK's copy of the body is
      // cancelled too, which may never happen.
      void reader.cancel().catch((error: unknown) => {
        diag.debug("Failed to cancel OpenAI image upload read", error);
      });
      return REDACTED_VALUE;
    }
    chunks.push(readResult.value);
    readResult = await reader.read();
  }
  const bytes = new Uint8Array(byteLength);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

async function readResponseUpload({
  upload,
  isTooLong,
}: ReadUploadParams & { upload: Response }): Promise<UploadReadResult> {
  if (isTooLong(Number(upload.headers.get("content-length")))) return REDACTED_VALUE;
  const clone = upload.clone();
  // Content-Length can be missing, or smaller than the decompressed body.
  const bytes = clone.body
    ? await readBodyWithinLimit({ body: clone.body, isTooLong })
    : new Uint8Array();
  if (bytes === REDACTED_VALUE) return REDACTED_VALUE;
  return {
    bytes,
    declaredMediaType: clone.headers.get("content-type") ?? undefined,
    filename: clone.url,
  };
}

async function readBlobUpload({
  upload,
  isTooLong,
}: ReadUploadParams & { upload: Blob }): Promise<UploadReadResult> {
  if (isTooLong(upload.size)) return REDACTED_VALUE;
  return {
    bytes: new Uint8Array(await upload.arrayBuffer()),
    declaredMediaType: upload.type || undefined,
    filename: "name" in upload && typeof upload.name === "string" ? upload.name : undefined,
  };
}

async function readFileStreamUpload({
  upload,
  isTooLong,
}: ReadUploadParams & { upload: object }): Promise<UploadReadResult> {
  const { promises: fileSystem, ReadStream } = await import("node:fs");
  if (!(upload instanceof ReadStream) || typeof upload.path !== "string") return undefined;
  // A ranged stream sends only part of the file, so the file would not match the request.
  const isRangedStream =
    Reflect.get(upload, "start") !== undefined || Reflect.get(upload, "end") !== Infinity;
  if (isRangedStream) return undefined;
  const fileStats = await fileSystem.stat(upload.path);
  if (!fileStats.isFile()) return undefined;
  if (isTooLong(fileStats.size)) return REDACTED_VALUE;
  // Read from the path separately so the SDK still owns the one-shot stream.
  return { bytes: await fileSystem.readFile(upload.path), filename: upload.path };
}

function readUpload({
  upload,
  isTooLong,
}: ReadUploadParams & { upload: unknown }): Promise<UploadReadResult> {
  if (typeof Response !== "undefined" && upload instanceof Response) {
    return readResponseUpload({ upload, isTooLong });
  }
  if (typeof Blob !== "undefined" && upload instanceof Blob) {
    return readBlobUpload({ upload, isTooLong });
  }
  if (upload != null && typeof upload === "object" && "path" in upload) {
    return readFileStreamUpload({ upload, isTooLong });
  }
  // Unknown one-shot async iterables cannot be inspected without consuming
  // the user's upload before the OpenAI SDK can build the request.
  return Promise.resolve(undefined);
}

async function uploadToDataURL({
  upload,
  base64ImageMaxLength,
}: {
  upload: unknown;
  base64ImageMaxLength: number;
}): Promise<string | undefined> {
  try {
    const contents = await readUpload({
      upload,
      isTooLong: (byteLength) => getBase64Length(byteLength) > base64ImageMaxLength,
    });
    if (contents === undefined || contents === REDACTED_VALUE) return contents;
    const { bytes, declaredMediaType, filename } = contents;
    const mediaType =
      (declaredMediaType?.startsWith("image/") ? declaredMediaType : undefined) ??
      sniffImageMediaType(bytes) ??
      mediaTypeFromFilename(filename) ??
      "image/png";
    const prefix = `data:${mediaType};base64,`;
    if (prefix.length + getBase64Length(bytes.length) > base64ImageMaxLength) {
      return REDACTED_VALUE;
    }
    return `${prefix}${bytesToBase64(bytes)}`;
  } catch (error) {
    diag.debug("Failed to capture OpenAI image upload", error);
    return undefined;
  }
}

export async function getInputImageAttributes({
  uploads,
  traceConfigOptions = {},
}: {
  uploads: ReadonlyArray<unknown>;
  traceConfigOptions?: TraceConfigOptions;
}): Promise<Attributes> {
  const traceConfig = generateTraceConfig(traceConfigOptions);
  if (traceConfig.hideInputs || traceConfig.hideInputImages) return {};
  const flattenedUploads = uploads.flatMap((upload) => (Array.isArray(upload) ? upload : [upload]));
  const dataURLs = await Promise.all(
    flattenedUploads.map((upload) =>
      uploadToDataURL({ upload, base64ImageMaxLength: traceConfig.base64ImageMaxLength }),
    ),
  );
  const attributes: Attributes = {};
  let imageIndex = 0;
  for (const dataURL of dataURLs) {
    if (!dataURL) continue;
    attributes[
      getImageAttributeKey({ namespace: SemanticConventions.INPUT_IMAGES, index: imageIndex })
    ] = dataURL;
    imageIndex++;
  }
  return attributes;
}
