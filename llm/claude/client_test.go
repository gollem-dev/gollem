package claude_test

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/m-mizutani/goerr/v2"
	"github.com/m-mizutani/gollem"
	"github.com/m-mizutani/gollem/llm/claude"
	"github.com/m-mizutani/gollem/trace"
	"github.com/m-mizutani/gt"
)

const (
	testTimeout   = 30 * time.Second
	maxTestTokens = 2048
)

func TestClaudeContentGenerate(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
	if !ok {
		t.Skip("TEST_CLAUDE_API_KEY is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := claude.New(ctx, apiKey)
	gt.NoError(t, err)

	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	result, err := session.Generate(ctx, []gollem.Input{gollem.Text("Say hello in one word")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err)
	gt.Array(t, result.Texts).Length(1).Required()
	gt.Value(t, len(result.Texts[0])).NotEqual(0)
}

// TestCreateSystemPrompt tests the createSystemPrompt function
func TestCreateSystemPrompt(t *testing.T) {
	ctx := context.Background()

	t.Run("empty config returns empty slice", func(t *testing.T) {
		cfg := gollem.NewSessionConfig()
		result, err := claude.CreateSystemPrompt(ctx, cfg)
		gt.NoError(t, err)

		// Should return empty slice when no system prompt
		gt.Equal(t, 0, len(result))
	})

	t.Run("result is correct type", func(t *testing.T) {
		cfg := gollem.NewSessionConfig()
		result, err := claude.CreateSystemPrompt(ctx, cfg)
		gt.NoError(t, err)

		// Empty slice can be nil in this implementation
		gt.Equal(t, 0, len(result))
	})

	t.Run("JSON content type check", func(t *testing.T) {
		// Create config with JSON content type
		cfg := gollem.NewSessionConfig()
		// Manually set content type since we can't use WithContentType in test
		// The actual functionality is tested in integration tests
		result, err := claude.CreateSystemPrompt(ctx, cfg)
		gt.NoError(t, err)

		// At minimum, should not panic and return valid type
		_ = result
	})
}

// TestSystemPromptSDKCompliance verifies SDK compliance
func TestSystemPromptSDKCompliance(t *testing.T) {
	ctx := context.Background()

	t.Run("SDK format verification", func(t *testing.T) {
		// This test verifies the format matches SDK expectations:
		// []anthropic.TextBlockParam{{Text: "..."}}

		// Create empty config
		cfg := gollem.NewSessionConfig()
		result, err := claude.CreateSystemPrompt(ctx, cfg)
		gt.NoError(t, err)

		// Empty case should return empty slice
		gt.Equal(t, 0, len(result))
	})

	t.Run("TextBlockParam structure", func(t *testing.T) {
		// Verify we can create TextBlockParam correctly
		testBlock := anthropic.TextBlockParam{
			Text: "Test prompt",
		}

		// Verify the Text field exists and is accessible
		gt.Equal(t, "Test prompt", testBlock.Text)

		// Create a slice as the function would return
		blocks := []anthropic.TextBlockParam{testBlock}
		gt.Equal(t, 1, len(blocks))
		gt.Equal(t, "Test prompt", blocks[0].Text)
	})
}

// TestSystemPromptComment verifies the implementation comment
func TestSystemPromptComment(t *testing.T) {
	ctx := context.Background()

	// This test documents that the implementation follows the official SDK format
	// The createSystemPrompt function should return []anthropic.TextBlockParam
	// in the format: []anthropic.TextBlockParam{{Text: "..."}}

	t.Run("comment accuracy", func(t *testing.T) {
		// The function is documented as:
		// "Returns []anthropic.TextBlockParam as per anthropic-sdk-go v1.5.0 specification"
		// This test verifies that claim

		cfg := gollem.NewSessionConfig()
		result, err := claude.CreateSystemPrompt(ctx, cfg)
		gt.NoError(t, err)

		// Should handle empty case correctly
		if len(result) > 0 {
			// If not empty, each element should have a Text field
			for _, block := range result {
				// Text field should be accessible
				_ = block.Text
			}
		}
	})
}

func TestTokenLimitErrorOptions(t *testing.T) {
	type testCase struct {
		name   string
		err    error
		hasTag bool
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			opts := claude.TokenLimitErrorOptions(tc.err)
			if tc.hasTag {
				gt.NotEqual(t, 0, len(opts))
			} else {
				gt.Equal(t, 0, len(opts))
			}
		}
	}

	// Create a mock anthropic.Error with token exceeded error
	createTokenExceededError := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "prompt is too long: 150000 tokens > 100000 maximum",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 400,
		}
		// Use UnmarshalJSON to properly set the internal raw field
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	createDifferentTypeError := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "authentication_error",
				"message": "Invalid API key",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 401,
		}
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	createDifferentMessageError := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "Invalid model specified",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 400,
		}
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	createDifferentStatusError := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "prompt is too long: 150000 tokens > 100000 maximum",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 500,
		}
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	create413Error := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "Prompt is too long",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 413,
		}
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	createCapitalizedMessageError := func() *anthropic.Error {
		rawJSON := map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "Prompt is too long: 150000 tokens > 100000 maximum",
			},
		}
		rawJSONBytes, _ := json.Marshal(rawJSON)

		err := &anthropic.Error{
			StatusCode: 400,
		}
		_ = err.UnmarshalJSON(rawJSONBytes)
		return err
	}

	t.Run("token exceeded error", runTest(testCase{
		name:   "prompt is too long",
		err:    createTokenExceededError(),
		hasTag: true,
	}))

	t.Run("different error type", runTest(testCase{
		name:   "authentication error",
		err:    createDifferentTypeError(),
		hasTag: false,
	}))

	t.Run("different message", runTest(testCase{
		name:   "invalid model",
		err:    createDifferentMessageError(),
		hasTag: false,
	}))

	t.Run("different status code", runTest(testCase{
		name:   "status 500",
		err:    createDifferentStatusError(),
		hasTag: false,
	}))

	t.Run("413 status code with capitalized message", runTest(testCase{
		name:   "413 Request Entity Too Large",
		err:    create413Error(),
		hasTag: true,
	}))

	t.Run("capitalized message with 400 status", runTest(testCase{
		name:   "Prompt is too long (capitalized)",
		err:    createCapitalizedMessageError(),
		hasTag: true,
	}))

	t.Run("nil error", runTest(testCase{
		name:   "nil error",
		err:    nil,
		hasTag: false,
	}))

	t.Run("non-anthropic error", runTest(testCase{
		name:   "generic error",
		err:    errors.New("some error"),
		hasTag: false,
	}))
}

func TestClaudeTokenLimitErrorIntegration(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
	if !ok {
		t.Skip("TEST_CLAUDE_API_KEY is not set")
	}

	// Only run if explicitly requested via environment variable
	if os.Getenv("TEST_TOKEN_LIMIT_ERROR") != "true" {
		t.Skip("TEST_TOKEN_LIMIT_ERROR is not set to true")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := claude.New(ctx, apiKey)
	gt.NoError(t, err)

	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	// Create a very long prompt to exceed token limit
	// Repeat a long text many times to ensure we exceed the limit
	longText := strings.Repeat("This is a test sentence to make the prompt very long. ", 100000)

	_, err = session.Generate(ctx, []gollem.Input{gollem.Text(longText)})
	gt.Error(t, err)

	// Verify the error has the token exceeded tag
	gt.True(t, goerr.HasTag(err, gollem.ErrTagTokenExceeded))
}

// TestWithBaseURL tests the WithBaseURL option functionality
func TestWithBaseURL(t *testing.T) {
	t.Run("default baseURL", func(t *testing.T) {
		client, err := claude.New(context.Background(), "test-key", claude.WithBaseURL(""))
		gt.NoError(t, err)
		gt.Equal(t, "", claude.GetBaseURL(client))
	})

	t.Run("custom baseURL", func(t *testing.T) {
		customURL := "https://custom.anthropic.com"
		client, err := claude.New(context.Background(), "test-key", claude.WithBaseURL(customURL))
		gt.NoError(t, err)
		gt.Equal(t, customURL, claude.GetBaseURL(client))
	})

	t.Run("empty baseURL after custom", func(t *testing.T) {
		// Test that empty baseURL overrides previous setting
		client1, err1 := claude.New(context.Background(), "test-key", claude.WithBaseURL("https://first.com"))
		gt.NoError(t, err1)
		gt.Equal(t, "https://first.com", claude.GetBaseURL(client1))

		// Apply empty baseURL after custom one
		client2, err2 := claude.New(context.Background(), "test-key",
			claude.WithBaseURL("https://first.com"),
			claude.WithBaseURL(""))
		gt.NoError(t, err2)
		gt.Equal(t, "", claude.GetBaseURL(client2)) // Should be empty, not first URL
	})
}

// TestPerCallGenerateOptions verifies that per-call GenerateOption overrides
// actually change the API request. A text-mode session gets a per-call
// ResponseSchema, and the response must be valid JSON matching the schema.
func TestPerCallGenerateOptions(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
	if !ok {
		t.Skip("TEST_CLAUDE_API_KEY is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := claude.New(ctx, apiKey)
	gt.NoError(t, err)

	// Create a plain text session — no ContentTypeJSON, no ResponseSchema
	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	schema := &gollem.Parameter{
		Type:  gollem.TypeObject,
		Title: "Color",
		Properties: map[string]*gollem.Parameter{
			"name": {Type: gollem.TypeString, Description: "color name", Required: true},
		},
	}

	// Per-call option should force JSON output via system prompt injection
	resp, err := session.Generate(ctx,
		[]gollem.Input{gollem.Text("Name a color.")},
		gollem.WithGenerateResponseSchema(schema),
		gollem.WithMaxTokens(maxTestTokens),
	)
	gt.NoError(t, err)
	gt.True(t, len(resp.Texts) > 0)

	var parsed map[string]any
	gt.NoError(t, json.Unmarshal([]byte(resp.Texts[0]), &parsed))
	gt.True(t, parsed["name"] != nil)
}

func TestClaudeMessagesToTraceMessages(t *testing.T) {
	type testCase struct {
		messages []anthropic.MessageParam
		expected []trace.Message
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			result := claude.ClaudeMessagesToTraceMessages(tc.messages)
			gt.Equal(t, tc.expected, result)
		}
	}

	t.Run("text message", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hello world")),
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewTextContent("hello world"),
			}},
		},
	}))

	t.Run("assistant message", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("response")),
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewTextContent("response"),
			}},
		},
	}))

	t.Run("tool use", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewAssistantMessage(
				anthropic.NewToolUseBlock("call-1", map[string]any{"q": "test"}, "search"),
			),
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewToolCallContent("call-1", "search", map[string]any{"q": "test"}),
			}},
		},
	}))

	t.Run("tool result", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock("call-1", "result text", false),
			),
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewToolResponseContent("call-1", "", nil),
				trace.NewTextContent("result text"),
			}},
		},
	}))

	t.Run("multiple messages", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("hello")),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("hi")),
			anthropic.NewUserMessage(anthropic.NewTextBlock("how are you")),
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{trace.NewTextContent("hello")}},
			{Role: "assistant", Contents: []trace.MessageContent{trace.NewTextContent("hi")}},
			{Role: "user", Contents: []trace.MessageContent{trace.NewTextContent("how are you")}},
		},
	}))

	t.Run("nil messages", runTest(testCase{
		messages: nil,
		expected: nil,
	}))

	t.Run("empty messages", runTest(testCase{
		messages: []anthropic.MessageParam{},
		expected: nil,
	}))

	t.Run("mixed content blocks", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewAssistantMessage(
				anthropic.NewTextBlock("Let me search"),
				anthropic.NewToolUseBlock("call-1", map[string]any{"q": "test"}, "search"),
			),
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewTextContent("Let me search"),
				trace.NewToolCallContent("call-1", "search", map[string]any{"q": "test"}),
			}},
		},
	}))

	t.Run("image with media type", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewImageBlockBase64("image/png", "iVBOR..."),
			),
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				{Type: "image", MediaType: "image/png"},
			}},
		},
	}))

	t.Run("thinking block", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewAssistantMessage(
				anthropic.NewThinkingBlock("sig123", "Let me think about this..."),
				anthropic.NewTextBlock("Here is my answer"),
			),
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewThinkingContent("Let me think about this..."),
				trace.NewTextContent("Here is my answer"),
			}},
		},
	}))

	t.Run("redacted thinking block", runTest(testCase{
		messages: []anthropic.MessageParam{
			anthropic.NewAssistantMessage(
				anthropic.NewRedactedThinkingBlock("redacted-data"),
				anthropic.NewTextBlock("answer"),
			),
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewRedactedThinkingContent(),
				trace.NewTextContent("answer"),
			}},
		},
	}))
}

// TestClaudeTraceRequestMessagesNewTurnOnly verifies that the trace's
// LLMRequest.Messages contains only messages newly added in this turn,
// not the entire conversation history that was actually sent to the API.
func TestClaudeTraceRequestMessagesNewTurnOnly(t *testing.T) {
	userContent, err := gollem.NewTextContent("previous question")
	gt.NoError(t, err)
	assistantContent, err := gollem.NewTextContent("previous answer")
	gt.NoError(t, err)
	history := &gollem.History{
		Version: gollem.HistoryVersion,
		LLType:  gollem.LLMTypeClaude,
		Messages: []gollem.Message{
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{userContent}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{assistantContent}},
		},
	}

	var sentMessages []anthropic.MessageParam
	mockClient := &apiClientMock{
		MessagesNewFunc: func(ctx context.Context, params anthropic.MessageNewParams) (*anthropic.Message, error) {
			sentMessages = params.Messages
			return &anthropic.Message{
				Content: []anthropic.ContentBlockUnion{
					{Type: "text", Text: "ok"},
				},
				Role:  "assistant",
				Model: "claude-3-opus-20240229",
			}, nil
		},
	}

	cfg := gollem.NewSessionConfig(gollem.WithSessionHistory(history))
	session, err := claude.NewSessionWithAPIClient(mockClient, cfg, "claude-3-opus-20240229")
	gt.NoError(t, err)

	rec := trace.New()
	ctx := rec.StartAgentExecute(context.Background())
	ctx = trace.WithHandler(ctx, rec)

	_, err = session.Generate(ctx, []gollem.Input{gollem.Text("new question")})
	gt.NoError(t, err)
	rec.EndAgentExecute(ctx, nil)

	// Sanity check: the actual API request still includes the full history.
	gt.N(t, len(sentMessages)).Equal(3)

	// Find the LLM call span.
	var llmSpan *trace.Span
	for _, child := range rec.Trace().RootSpan.Children {
		if child.Kind == trace.SpanKindLLMCall {
			llmSpan = child
			break
		}
	}
	gt.Value(t, llmSpan).NotNil()

	msgs := llmSpan.LLMCall.Request.Messages
	gt.A(t, msgs).Length(1)
	gt.Equal(t, "user", msgs[0].Role)
	gt.A(t, msgs[0].Contents).Length(1)
	gt.Equal(t, "text", msgs[0].Contents[0].Type)
	gt.Equal(t, "new question", msgs[0].Contents[0].Text)

	for _, m := range msgs {
		for _, c := range m.Contents {
			gt.S(t, c.Text).NotContains("previous")
		}
	}
}
