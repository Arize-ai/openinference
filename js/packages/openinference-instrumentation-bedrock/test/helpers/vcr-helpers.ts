import nock from "nock";
import * as fs from "fs";
import * as path from "path";
import { BedrockRuntimeClient } from "@aws-sdk/client-bedrock-runtime";
import {
  MOCK_AWS_CREDENTIALS,
  VALID_AWS_CREDENTIALS,
  MOCK_AUTH_HEADERS,
} from "../config/constants";

// Helper function to load mock response from recording file
export const loadRecordingData = (recordingPath: string) => {
  if (!fs.existsSync(recordingPath)) {
    return null;
  }

  try {
    const recordingData = JSON.parse(fs.readFileSync(recordingPath, "utf8"));
    const recording = recordingData[0];
    if (!recording) return null;

    // Extract model ID from the path (handle both invoke and invoke-with-response-stream)
    const pathMatch = recording.path?.match(/\/model\/([^\/]+)\/invoke/);
    const modelId = pathMatch ? decodeURIComponent(pathMatch[1]) : null;
    
    // Determine if this is a streaming endpoint
    const isStreaming = recording.path?.includes('invoke-with-response-stream');

    return {
      response: recording.response,
      modelId,
      recording,
      status: recording.status || 200,
      isStreaming,
    };
  } catch (error) {
    console.warn(`Failed to load recording from ${recordingPath}:`, error);
    return null;
  }
};

// Helper function to create nock mock for Bedrock API
export const createNockMock = (
  mockResponse: any,
  modelId?: string,
  status: number = 200,
  defaultModelId: string = "anthropic.claude-3-5-sonnet-20240620-v1:0",
  isStreaming: boolean = false,
) => {
  const targetModelId = modelId || defaultModelId;
  const endpoint = isStreaming ? 'invoke-with-response-stream' : 'invoke';
  
  const nockScope = nock("https://bedrock-runtime.us-east-1.amazonaws.com")
    .post(`/model/${encodeURIComponent(targetModelId)}/${endpoint}`);
  
  if (isStreaming && typeof mockResponse === 'string') {
    // For streaming responses, convert hex string to binary buffer
    const binaryResponse = Buffer.from(mockResponse, 'hex');
    nockScope.reply(status, binaryResponse, {
      'Content-Type': 'application/vnd.amazon.eventstream',
      'Transfer-Encoding': 'chunked'
    });
  } else {
    nockScope.reply(status, mockResponse);
  }
};

// Helper function to sanitize auth headers in recordings
export const sanitizeAuthHeaders = (recordings: any[]) => {
  recordings.forEach((recording: any) => {
    if (recording.reqheaders) {
      Object.assign(recording.reqheaders, MOCK_AUTH_HEADERS);
    }
  });
};

// Helper function to create sanitized recording path
export const createRecordingPath = (testName: string, testDir: string) => {
  const sanitizedName = testName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return path.join(testDir, "recordings", `${sanitizedName}.json`);
};

// Generate recording file path based on test name
export const getRecordingPath = (testName: string, testDir: string) => {
  return createRecordingPath(testName, testDir);
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
      connectionTimeout: 1000,
      requestTimeout: 5000,
    },
  });
};
