package io.openinference.instrumentation;

/**
 * Configuration for OpenInference tracing.
 */
public class TraceConfig {
    
    private final boolean hideInputs;
    private final boolean hideOutputs;
    private final boolean hideInputMessages;
    private final boolean hideOutputMessages;
    private final boolean hideInputImages;
    private final boolean hideOutputImages;
    private final boolean hideInputText;
    private final boolean hideOutputText;
    private final boolean hideInputAudio;
    private final boolean hideOutputAudio;
    private final boolean hideInputEmbeddings;
    private final boolean hideOutputEmbeddings;
    private final boolean hidePromptTemplate;
    private final boolean hidePromptTemplateVariables;
    private final boolean hidePromptTemplateVersion;
    private final boolean hideToolParameters;
    private final String base64ImageMaxLength;
    
    private TraceConfig(Builder builder) {
        this.hideInputs = builder.hideInputs;
        this.hideOutputs = builder.hideOutputs;
        this.hideInputMessages = builder.hideInputMessages;
        this.hideOutputMessages = builder.hideOutputMessages;
        this.hideInputImages = builder.hideInputImages;
        this.hideOutputImages = builder.hideOutputImages;
        this.hideInputText = builder.hideInputText;
        this.hideOutputText = builder.hideOutputText;
        this.hideInputAudio = builder.hideInputAudio;
        this.hideOutputAudio = builder.hideOutputAudio;
        this.hideInputEmbeddings = builder.hideInputEmbeddings;
        this.hideOutputEmbeddings = builder.hideOutputEmbeddings;
        this.hidePromptTemplate = builder.hidePromptTemplate;
        this.hidePromptTemplateVariables = builder.hidePromptTemplateVariables;
        this.hidePromptTemplateVersion = builder.hidePromptTemplateVersion;
        this.hideToolParameters = builder.hideToolParameters;
        this.base64ImageMaxLength = builder.base64ImageMaxLength;
    }
    
    public static Builder builder() {
        return new Builder();
    }
    
    public static TraceConfig getDefault() {
        return builder().build();
    }
    
    // Getters
    public boolean isHideInputs() { return hideInputs; }
    public boolean isHideOutputs() { return hideOutputs; }
    public boolean isHideInputMessages() { return hideInputMessages; }
    public boolean isHideOutputMessages() { return hideOutputMessages; }
    public boolean isHideInputImages() { return hideInputImages; }
    public boolean isHideOutputImages() { return hideOutputImages; }
    public boolean isHideInputText() { return hideInputText; }
    public boolean isHideOutputText() { return hideOutputText; }
    public boolean isHideInputAudio() { return hideInputAudio; }
    public boolean isHideOutputAudio() { return hideOutputAudio; }
    public boolean isHideInputEmbeddings() { return hideInputEmbeddings; }
    public boolean isHideOutputEmbeddings() { return hideOutputEmbeddings; }
    public boolean isHidePromptTemplate() { return hidePromptTemplate; }
    public boolean isHidePromptTemplateVariables() { return hidePromptTemplateVariables; }
    public boolean isHidePromptTemplateVersion() { return hidePromptTemplateVersion; }
    public boolean isHideToolParameters() { return hideToolParameters; }
    public String getBase64ImageMaxLength() { return base64ImageMaxLength; }
    
    public static class Builder {
        private boolean hideInputs = false;
        private boolean hideOutputs = false;
        private boolean hideInputMessages = false;
        private boolean hideOutputMessages = false;
        private boolean hideInputImages = false;
        private boolean hideOutputImages = false;
        private boolean hideInputText = false;
        private boolean hideOutputText = false;
        private boolean hideInputAudio = false;
        private boolean hideOutputAudio = false;
        private boolean hideInputEmbeddings = false;
        private boolean hideOutputEmbeddings = false;
        private boolean hidePromptTemplate = false;
        private boolean hidePromptTemplateVariables = false;
        private boolean hidePromptTemplateVersion = false;
        private boolean hideToolParameters = false;
        private String base64ImageMaxLength = "unlimited";
        
        public Builder hideInputs(boolean hideInputs) {
            this.hideInputs = hideInputs;
            return this;
        }
        
        public Builder hideOutputs(boolean hideOutputs) {
            this.hideOutputs = hideOutputs;
            return this;
        }
        
        public Builder hideInputMessages(boolean hideInputMessages) {
            this.hideInputMessages = hideInputMessages;
            return this;
        }
        
        public Builder hideOutputMessages(boolean hideOutputMessages) {
            this.hideOutputMessages = hideOutputMessages;
            return this;
        }
        
        public Builder hideInputImages(boolean hideInputImages) {
            this.hideInputImages = hideInputImages;
            return this;
        }
        
        public Builder hideOutputImages(boolean hideOutputImages) {
            this.hideOutputImages = hideOutputImages;
            return this;
        }
        
        public Builder hideInputText(boolean hideInputText) {
            this.hideInputText = hideInputText;
            return this;
        }
        
        public Builder hideOutputText(boolean hideOutputText) {
            this.hideOutputText = hideOutputText;
            return this;
        }
        
        public Builder hideInputAudio(boolean hideInputAudio) {
            this.hideInputAudio = hideInputAudio;
            return this;
        }
        
        public Builder hideOutputAudio(boolean hideOutputAudio) {
            this.hideOutputAudio = hideOutputAudio;
            return this;
        }
        
        public Builder hideInputEmbeddings(boolean hideInputEmbeddings) {
            this.hideInputEmbeddings = hideInputEmbeddings;
            return this;
        }
        
        public Builder hideOutputEmbeddings(boolean hideOutputEmbeddings) {
            this.hideOutputEmbeddings = hideOutputEmbeddings;
            return this;
        }
        
        public Builder hidePromptTemplate(boolean hidePromptTemplate) {
            this.hidePromptTemplate = hidePromptTemplate;
            return this;
        }
        
        public Builder hidePromptTemplateVariables(boolean hidePromptTemplateVariables) {
            this.hidePromptTemplateVariables = hidePromptTemplateVariables;
            return this;
        }
        
        public Builder hidePromptTemplateVersion(boolean hidePromptTemplateVersion) {
            this.hidePromptTemplateVersion = hidePromptTemplateVersion;
            return this;
        }
        
        public Builder hideToolParameters(boolean hideToolParameters) {
            this.hideToolParameters = hideToolParameters;
            return this;
        }
        
        public Builder base64ImageMaxLength(String base64ImageMaxLength) {
            this.base64ImageMaxLength = base64ImageMaxLength;
            return this;
        }
        
        public TraceConfig build() {
            return new TraceConfig(this);
        }
    }
} 