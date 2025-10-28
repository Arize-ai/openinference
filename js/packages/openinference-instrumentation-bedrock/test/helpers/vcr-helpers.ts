import nock from "nock";
import * as fs from "fs";
import * as path from "path";
import { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";
import {
  MOCK_AWS_CREDENTIALS,
  VALID_AWS_CREDENTIALS,
  MOCK_AUTH_HEADERS,
} from "../config/constants";

// Helper function to create nock mock for Bedrock API
export const createNockMock = (
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  mockResponse: any,
  modelId?: string,
  status: number = 200,
  defaultModelId: string = "anthropic.claude-3-5-sonnet-20240620-v1:0",
  isStreaming: boolean = false,
  isConverse: boolean = false,
) => {
  const targetModelId = modelId || defaultModelId;
  let endpoint: string;

  if (isConverse && isStreaming) {
    endpoint = "converse-stream";
  } else if (isConverse) {
    endpoint = "converse";
  } else if (isStreaming) {
    endpoint = "invoke-with-response-stream";
  } else {
    endpoint = "invoke";
  }

  const nockScope = nock(
    "https://bedrock-runtime.us-east-1.amazonaws.com",
  ).post(`/model/${encodeURIComponent(targetModelId)}/${endpoint}`);

  if (isStreaming && typeof mockResponse === "string") {
    // For streaming responses, convert hex string to binary buffer
    const binaryResponse = Buffer.from(mockResponse, "hex");
    nockScope.reply(status, binaryResponse, {
      "Content-Type": "application/vnd.amazon.eventstream",
      "Transfer-Encoding": "chunked",
    });
  } else {
    nockScope.reply(status, mockResponse);
  }
};

// Helper function to sanitize auth headers in recordings
export const sanitizeAuthHeaders = (recordings: unknown[]) => {
  recordings.forEach((recording) => {
    if (
      recording &&
      typeof recording === "object" &&
      "reqheaders" in recording &&
      recording.reqheaders &&
      typeof recording.reqheaders === "object"
    ) {
      Object.assign(recording.reqheaders, MOCK_AUTH_HEADERS);
    }
  });
};

// Generate recording file path based on test name
export const getRecordingPath = (testName: string, testDir: string) => {
  const sanitizedName = testName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return path.join(testDir, "recordings", `${sanitizedName}.json`);
};

// Helper function to create test client with consistent configuration
export const createTestClient = (isRecordingMode: boolean) => {
  return new BedrockRuntimeClient({
    region: "us-east-1",
    credentials: isRecordingMode
      ? {
          // Recording mode: use real credentials from environment
          ...VALID_AWS_CREDENTIALS,
        }
      : {
          // Replay mode: use mock credentials that match sanitized recordings
          ...MOCK_AWS_CREDENTIALS,
        },
    // Disable connection reuse to ensure nock can intercept properly
    requestHandler: {
      connectionTimeout: 5000,
      requestTimeout: 10000,
    },
  });
};

// Helper function to setup test recording for VCR-style testing
// This function handles both recording and replay modes
export const setupTestRecording = (
  testName: string,
  testDir: string,
  isRecordingMode: boolean,
  defaultModelId: string = "anthropic.claude-3-5-sonnet-20240620-v1:0",
) => {
  const recordingsPath = getRecordingPath(testName, testDir);

  if (!isRecordingMode) {
    // Replay mode: create mock from test-specific recording
    if (fs.existsSync(recordingsPath)) {
      const recordingData = JSON.parse(fs.readFileSync(recordingsPath, "utf8"));

      // Create mocks for all recorded requests
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      recordingData.forEach((recording: any) => {
        // Extract model ID from the path
        const invokeMatch = recording.path?.match(/\/model\/([^/]+)\/invoke/);
        const converseMatch = recording.path?.match(
          /\/model\/([^/]+)\/converse/,
        );
        const modelId = invokeMatch
          ? decodeURIComponent(invokeMatch[1])
          : converseMatch
            ? decodeURIComponent(converseMatch[1])
            : null;

        // Determine endpoint type
        const isStreaming =
          recording.path?.includes("invoke-with-response-stream") ||
          recording.path?.includes("converse-stream");
        const isConverse = recording.path?.includes("/converse");

        createNockMock(
          recording.response,
          modelId || undefined,
          recording.status || 200,
          defaultModelId,
          isStreaming,
          isConverse,
        );
      });
    } else {
      // eslint-disable-next-line no-console
      console.log(`No recordings found at ${recordingsPath}`);
    }
  }

  return recordingsPath;
};

// Helper function to initialize nock recording mode in beforeEach hooks
export const initializeRecordingMode = () => {
  nock.recorder.rec({
    output_objects: true,
    enable_reqheaders_recording: true,
  });
};

// Helper function to save recording mode data in afterEach hooks
export const saveRecordingModeData = (
  currentTestName: string,
  recordingsPath: string,
) => {
  const recordings = nock.recorder.play();
  // eslint-disable-next-line no-console
  console.log(
    `Captured ${recordings.length} recordings for test: ${currentTestName}`,
  );
  if (recordings.length > 0) {
    // Sanitize auth headers - replace with mock credentials for replay compatibility
    sanitizeAuthHeaders(recordings);

    const recordingsDir = path.dirname(recordingsPath);
    if (!fs.existsSync(recordingsDir)) {
      fs.mkdirSync(recordingsDir, { recursive: true });
    }
    fs.writeFileSync(recordingsPath, JSON.stringify(recordings, null, 2));
    // eslint-disable-next-line no-console
    console.log(
      `Saved sanitized recordings to ${path.basename(recordingsPath)}`,
    );
  }
};
