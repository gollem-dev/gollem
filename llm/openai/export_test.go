package openai

import (
	"github.com/gollem-dev/gollem"
	"github.com/sashabaranov/go-openai"
)

// Export convert functions for testing
var (
	ConvertTool                   = convertTool
	ConvertParameterToSchema      = convertParameterToSchema
	ConvertResponseSchemaToOpenAI = convertResponseSchemaToOpenAI
	TokenLimitErrorOptions        = tokenLimitErrorOptions
	OpenaiMessagesToTraceMessages = openaiMessagesToTraceMessages
	ToMessages                    = toMessages
	NewHistory                    = newHistory
)

// Export for testing
type APIClient = apiClient

// NewSessionWithAPIClient creates a new session with a custom API client for testing
func NewSessionWithAPIClient(client apiClient, cfg gollem.SessionConfig, model string) (*Session, error) {
	tools := make([]openai.Tool, 0, len(cfg.Tools()))
	for _, tool := range cfg.Tools() {
		tools = append(tools, convertTool(tool))
	}

	issuer := gollem.Issuer{Provider: gollem.LLMTypeOpenAI, Model: model}

	// Initialize historyMessages from config
	var historyMessages []openai.ChatCompletionMessage
	if cfg.History() != nil {
		var err error
		historyMessages, err = toMessages(cfg.History(), issuer)
		if err != nil {
			return nil, err
		}
	}

	return &Session{
		apiClient:       client,
		defaultModel:    model,
		systemPrompt:    cfg.SystemPrompt(),
		tools:           tools,
		historyMessages: historyMessages,
		issuer:          issuer,
		params:          generationParameters{},
		cfg:             cfg,
	}, nil
}

// EnableStrictMode turns on strict mode for the response schema of a Chat
// Completions session. No option sets it, so tests that send a strict schema
// to the API set it here.
func EnableStrictMode(session gollem.Session) {
	session.(*Session).strictMode = true
}

// GetBaseURL returns the base URL from an OpenAI client for testing
func GetBaseURL(client *Client) string {
	return client.baseURL
}
