export { GoogleGenAIInstrumentation, isPatched } from "./instrumentation";

import { GoogleGenAI, GoogleGenAIOptions } from "@google/genai";
import { GoogleGenAIInstrumentation } from "./instrumentation";

// Singleton instance for the helper function
let _globalInstrumentation: GoogleGenAIInstrumentation | null = null;

/**
 * Creates an instrumented GoogleGenAI instance
 * 
 * @param options - GoogleGenAI configuration options
 * @param instrumentation - Optional instrumentation instance. If not provided, uses a global singleton
 * @returns An instrumented GoogleGenAI instance
 * 
 * @example
 * ```typescript
 * import { createInstrumentedGoogleGenAI } from "@arizeai/openinference-instrumentation-google-genai";
 * 
 * const ai = createInstrumentedGoogleGenAI({
 *   apiKey: process.env.GOOGLE_API_KEY,
 * });
 * 
 * // Use ai as normal - all calls are automatically traced
 * const response = await ai.models.generateContent({
 *   model: "gemini-2.5-flash",
 *   contents: "Hello!",
 * });
 * ```
 */
export function createInstrumentedGoogleGenAI(
  options: GoogleGenAIOptions,
  instrumentation?: GoogleGenAIInstrumentation,
): GoogleGenAI {
  // Create the GoogleGenAI instance
  const ai = new GoogleGenAI(options);

  // Use provided instrumentation or create/reuse global singleton
  const inst = instrumentation || _globalInstrumentation || new GoogleGenAIInstrumentation();
  
  // Store as global if we created it
  if (!instrumentation && !_globalInstrumentation) {
    _globalInstrumentation = inst;
  }

  // Instrument the instance
  inst.instrumentInstance(ai);

  return ai;
}

