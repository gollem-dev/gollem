package gemini_test

import (
	"context"
	"fmt"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/m-mizutani/gt"
	"google.golang.org/genai"
)

func TestMiddlewareHistoryIntervention(t *testing.T) {
	testHistoryAccess := func(t *testing.T) {
		historyReceived := false

		// Create a mock API client that returns a simple response
		mockClient := &apiClientMock{
			GenerateContentFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
				return &genai.GenerateContentResponse{
					Candidates: []*genai.Candidate{
						{
							Content: &genai.Content{
								Parts: []*genai.Part{
									{Text: "Response from API"},
								},
								Role: "model",
							},
						},
					},
				}, nil
			},
			GenerateContentStreamFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) <-chan gemini.StreamResponse {
				ch := make(chan gemini.StreamResponse)
				close(ch)
				return ch
			},
		}

		// Create initial history
		textContent, _ := gollem.NewTextContent("Initial message")
		initialHistory := &gollem.History{
			Version: gollem.HistoryVersion,
			LLType:  gollem.LLMTypeGemini,
			Messages: []gollem.Message{
				{
					Role:     gollem.RoleUser,
					Contents: []gollem.MessageContent{textContent},
				},
			},
		}

		// Create middleware that verifies history is accessible
		historyMiddleware := func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
			return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
				// Verify history is accessible in middleware
				if req.History != nil && len(req.History.Messages) > 0 {
					historyReceived = true
				}

				// Call the next handler
				return next(ctx, req)
			}
		}

		// Create session config with middleware
		cfg := gollem.NewSessionConfig(
			gollem.WithSessionHistory(initialHistory),
			gollem.WithSessionContentBlockMiddleware(historyMiddleware),
		)

		// Create session with mock client
		session, _ := gemini.NewSessionWithAPIClient(mockClient, cfg, "gemini-1.5-pro")

		// Generate content
		ctx := context.Background()
		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("test input")})
		gt.NoError(t, err)
		gt.NotNil(t, resp)

		// Verify middleware received history
		gt.True(t, historyReceived)
	}

	t.Run("history access in middleware", testHistoryAccess)
}

func TestMiddlewareChainExecution(t *testing.T) {
	testChainOrder := func(t *testing.T) {
		executionOrder := []string{}

		// Create a mock API client
		mockClient := &apiClientMock{
			GenerateContentFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
				executionOrder = append(executionOrder, "api_call")
				return &genai.GenerateContentResponse{
					Candidates: []*genai.Candidate{
						{
							Content: &genai.Content{
								Parts: []*genai.Part{
									{Text: "response"},
								},
								Role: "model",
							},
						},
					},
				}, nil
			},
			GenerateContentStreamFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) <-chan gemini.StreamResponse {
				ch := make(chan gemini.StreamResponse)
				close(ch)
				return ch
			},
		}

		// Create middleware 1
		mw1 := func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
			return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
				executionOrder = append(executionOrder, "mw1_before")
				resp, err := next(ctx, req)
				executionOrder = append(executionOrder, "mw1_after")
				return resp, err
			}
		}

		// Create middleware 2
		mw2 := func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
			return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
				executionOrder = append(executionOrder, "mw2_before")
				resp, err := next(ctx, req)
				executionOrder = append(executionOrder, "mw2_after")
				return resp, err
			}
		}

		// Create session config with multiple middleware
		initialHistory := &gollem.History{
			Version: gollem.HistoryVersion,
			LLType:  gollem.LLMTypeGemini,
		}
		cfg := gollem.NewSessionConfig(
			gollem.WithSessionHistory(initialHistory),
			gollem.WithSessionContentBlockMiddleware(mw1),
			gollem.WithSessionContentBlockMiddleware(mw2),
		)

		// Create session with mock client
		session, _ := gemini.NewSessionWithAPIClient(mockClient, cfg, "gemini-1.5-pro")

		// Generate content
		ctx := context.Background()
		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("test")})
		gt.NoError(t, err)

		// Verify execution order
		expected := []string{"mw1_before", "mw2_before", "api_call", "mw2_after", "mw1_after"}
		gt.Equal(t, expected, executionOrder)
	}

	t.Run("chain execution order", testChainOrder)
}

func TestMiddlewareSameAddressModifiedContent(t *testing.T) {
	testHistoryAccumulation := func(t *testing.T) {
		callCount := 0

		// Create a mock API client that returns a simple response
		mockClient := &apiClientMock{
			GenerateContentFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*genai.GenerateContentResponse, error) {
				callCount++
				t.Logf("API call %d with %d contents", callCount, len(contents))
				return &genai.GenerateContentResponse{
					Candidates: []*genai.Candidate{
						{
							Content: &genai.Content{
								Parts: []*genai.Part{
									{Text: fmt.Sprintf("Response %d from API", callCount)},
								},
								Role: "model",
							},
						},
					},
				}, nil
			},
			GenerateContentStreamFunc: func(ctx context.Context, model string, contents []*genai.Content, config *genai.GenerateContentConfig) <-chan gemini.StreamResponse {
				ch := make(chan gemini.StreamResponse)
				close(ch)
				return ch
			},
		}

		// Create initial history
		textContent, _ := gollem.NewTextContent("Initial")
		initialHistory := &gollem.History{
			Version: gollem.HistoryVersion,
			LLType:  gollem.LLMTypeGemini,
			Messages: []gollem.Message{
				{
					Role:     gollem.RoleUser,
					Contents: []gollem.MessageContent{textContent},
				},
			},
		}

		// Create session config
		cfg := gollem.NewSessionConfig(
			gollem.WithSessionHistory(initialHistory),
		)

		// Create session with mock client
		session, _ := gemini.NewSessionWithAPIClient(mockClient, cfg, "gemini-1.5-pro")

		// Generate content multiple times
		ctx := context.Background()

		// First call
		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("first input")})
		gt.NoError(t, err)

		// Get history after first call
		history1, err := session.History()
		gt.NoError(t, err)
		firstCallMessageCount := len(history1.Messages)

		// Second call
		_, err = session.Generate(ctx, []gollem.Input{gollem.Text("second input")})
		gt.NoError(t, err)

		// Get history after second call
		history2, err := session.History()
		gt.NoError(t, err)
		secondCallMessageCount := len(history2.Messages)

		// Verify history accumulates correctly
		t.Logf("First call messages: %d, Second call messages: %d", firstCallMessageCount, secondCallMessageCount)
		gt.True(t, secondCallMessageCount > firstCallMessageCount)
	}

	t.Run("history accumulation across calls", testHistoryAccumulation)
}
