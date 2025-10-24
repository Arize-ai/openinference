import { redactUrl } from "../src/httpUtils";

describe("httpUtils", () => {
  describe("redactUrl", () => {
    it("should redact credentials from URLs", () => {
      const url = "https://user:pass@api.openai.com/v1/chat/completions";
      const result = redactUrl(url);
      expect(result).toBe(
        "https://REDACTED:REDACTED@api.openai.com/v1/chat/completions",
      );
    });

    it("should redact sensitive query parameters", () => {
      const url =
        "https://api.example.com/chat?AWSAccessKeyId=secret&Signature=secret&model=gpt-4";
      const result = redactUrl(url);
      expect(result).toBe(
        "https://api.example.com/chat?AWSAccessKeyId=REDACTED&Signature=REDACTED&model=gpt-4",
      );
    });

    it("should redact both credentials and query parameters", () => {
      const url =
        "https://user:pass@api.example.com/chat?AWSAccessKeyId=secret&model=gpt-4";
      const result = redactUrl(url);
      expect(result).toBe(
        "https://REDACTED:REDACTED@api.example.com/chat?AWSAccessKeyId=REDACTED&model=gpt-4",
      );
    });

    it("should handle malformed URLs gracefully", () => {
      const malformedUrl = "not-a-valid-url";
      const result = redactUrl(malformedUrl);
      expect(result).toBe(malformedUrl);
    });
  });
});
