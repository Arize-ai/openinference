/**
 * HTTP utilities for OpenTelemetry instrumentation
 * JavaScript equivalent of opentelemetry-python-contrib/util/opentelemetry-util-http
 * Minimal version containing only URL redaction functionality
 */

/**
 * List of query parameters that should be redacted for security
 */
const PARAMS_TO_REDACT = [
  "AWSAccessKeyId",
  "Signature",
  "sig",
  "X-Goog-Signature",
];

/**
 * Replaces the username and password with the keyword `REDACTED` in a URL
 * Only modifies the URL if it is valid and contains credentials
 * @param url The URL string to process
 * @returns The URL with credentials redacted, or original URL if invalid
 */
export function removeUrlCredentials(url: string): string {
  try {
    const parsed = new URL(url);

    // Check if URL has credentials
    if (parsed.username || parsed.password) {
      // Create new URL with redacted credentials
      const newUrl = new URL(url);
      newUrl.username = "REDACTED";
      newUrl.password = "REDACTED";
      return newUrl.toString();
    }

    return url;
  } catch (error) {
    // If URL parsing fails, return original URL
    return url;
  }
}

/**
 * Redacts sensitive query parameter values in a URL
 * @param url The URL string to process
 * @returns The URL with sensitive query parameters redacted, or original URL if no changes needed
 */
export function redactQueryParameters(url: string): string {
  try {
    const parsed = new URL(url);

    if (!parsed.search) {
      // No query parameters to redact
      return url;
    }

    const searchParams = new URLSearchParams(parsed.search);
    let hasRedactedParams = false;

    // Check if any parameters need redaction
    for (const param of PARAMS_TO_REDACT) {
      if (searchParams.has(param)) {
        searchParams.set(param, "REDACTED");
        hasRedactedParams = true;
      }
    }

    if (!hasRedactedParams) {
      return url;
    }

    // Reconstruct URL with redacted parameters
    const newUrl = new URL(url);
    newUrl.search = searchParams.toString();
    return newUrl.toString();
  } catch (error) {
    // If URL parsing fails, return original URL
    return url;
  }
}

/**
 * Redacts sensitive data from the URL, including credentials and query parameters
 * @param url The URL string to process
 * @returns The URL with all sensitive data redacted
 */
export function redactUrl(url: string): string {
  let redactedUrl = removeUrlCredentials(url);
  redactedUrl = redactQueryParameters(redactedUrl);
  return redactedUrl;
}
