// Test constants
export const TEST_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0";
export const TEST_USER_MESSAGE = "Hello, how are you?";
export const TEST_MAX_TOKENS = 100;

// Clearly fake credentials for VCR testing - these are NEVER real
export const MOCK_AWS_CREDENTIALS = {
  accessKeyId: "AKIATEST1234567890AB",
  secretAccessKey: "FAKE-SECRET-KEY-FOR-VCR-TESTING-ONLY-1234567890",
  sessionToken: "FAKE-SESSION-TOKEN-FOR-VCR-TESTING-ONLY",
};

export const VALID_AWS_CREDENTIALS = {
  accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
  sessionToken: process.env.AWS_SESSION_TOKEN!,
};

export const MOCK_AUTH_HEADERS = {
  authorization:
    "AWS4-HMAC-SHA256 Credential=AKIATEST1234567890AB/20250626/us-east-1/bedrock/aws4_request, SignedHeaders=accept;content-length;content-type;host;x-amz-date, Signature=fake-signature-for-vcr-testing",
  "x-amz-security-token": MOCK_AWS_CREDENTIALS.sessionToken,
  "x-amz-date": "20250626T120000Z",
};
