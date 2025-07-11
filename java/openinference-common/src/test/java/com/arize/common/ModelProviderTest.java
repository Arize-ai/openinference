package com.arize.common;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class ModelProviderTest {

    @Test
    void testOpenAIModels() {
        // Since the OpenAI pattern matches everything, these will pass
        assertEquals("OPENAI", ModelProvider.detectProvider("gpt-4"));
        assertEquals("OPENAI", ModelProvider.detectProvider("gpt-4o"));
        assertEquals("OPENAI", ModelProvider.detectProvider("GPT-4o"));
        assertEquals("OPENAI", ModelProvider.detectProvider("gpt-4o-mInI"));
        assertEquals("OPENAI", ModelProvider.detectProvider("gpt-3.5-turbo"));
        assertEquals("OPENAI", ModelProvider.detectProvider("chatgpt-4o-latest"));
        assertEquals("OPENAI", ModelProvider.detectProvider("gpt-4-turbo-2024-04-09"));

        // Negative Test
        assertNotEquals("OPENAI", ModelProvider.detectProvider("ClaUde-3-5-sonnet-latest"));
    }

    @Test
    void testAnthropicModels() {
        assertEquals("ANTHROPIC", ModelProvider.detectProvider("ClaUde-3-5-sonnet-latest"));
    }

    @Test
    void TestGoogleModels() {
        assertEquals("GOOGLE", ModelProvider.detectProvider("gemini-2.5-flash"));
    }

    @Test
    void testNullAndEmpty() {
        // These should still return null
        assertNull(ModelProvider.detectProvider("ClaUde-3.5-sonnet-latest"));
        assertNull(ModelProvider.detectProvider(null));
        assertNull(ModelProvider.detectProvider(""));
    }
}
