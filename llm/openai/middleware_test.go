package openai_test

import (
	"context"
	"fmt"
	"io"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/m-mizutani/gt"
	oai "github.com/sashabaranov/go-openai"
)

func TestMiddlewareHistoryIntervention(t *testing.T) {
	testHistoryAccess := func(t *testing.T) {
		historyReceived := false

		// Create a mock API client that returns a simple response
		mockClient := &apiClientMock{
			CreateChatCompletionFunc: func(ctx context.Context, req oai.ChatCompletionRequest) (oai.ChatCompletionResponse, error) {
				return oai.ChatCompletionResponse{
					Choices: []oai.ChatCompletionChoice{
						{
							Message: oai.ChatCompletionMessage{
								Content: "Response from API",
								Role:    oai.ChatMessageRoleAssistant,
							},
						},
					},
				}, nil
			},
			CreateChatCompletionStreamFunc: func(ctx context.Context, req oai.ChatCompletionRequest) (*oai.ChatCompletionStream, error) {
				return nil, io.EOF // Not used in this test
			},
		}

		// Create initial history
		textContent, _ := gollem.NewTextContent("Initial message")
		initialHistory := &gollem.History{
			Version: gollem.HistoryVersion,
			LLType:  gollem.LLMTypeOpenAI,
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
		session, _ := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-4")

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
			CreateChatCompletionFunc: func(ctx context.Context, req oai.ChatCompletionRequest) (oai.ChatCompletionResponse, error) {
				executionOrder = append(executionOrder, "api_call")
				return oai.ChatCompletionResponse{
					Choices: []oai.ChatCompletionChoice{
						{
							Message: oai.ChatCompletionMessage{
								Content: "response",
								Role:    oai.ChatMessageRoleAssistant,
							},
						},
					},
				}, nil
			},
			CreateChatCompletionStreamFunc: func(ctx context.Context, req oai.ChatCompletionRequest) (*oai.ChatCompletionStream, error) {
				return nil, io.EOF // Not used
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
		cfg := gollem.NewSessionConfig(
			gollem.WithSessionContentBlockMiddleware(mw1),
			gollem.WithSessionContentBlockMiddleware(mw2),
		)

		// Create session with mock client
		session, _ := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-4")

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
			CreateChatCompletionFunc: func(ctx context.Context, req oai.ChatCompletionRequest) (oai.ChatCompletionResponse, error) {
				callCount++
				t.Logf("API call %d with %d messages", callCount, len(req.Messages))
				return oai.ChatCompletionResponse{
					Choices: []oai.ChatCompletionChoice{
						{
							Message: oai.ChatCompletionMessage{
								Content: fmt.Sprintf("Response %d from API", callCount),
								Role:    oai.ChatMessageRoleAssistant,
							},
						},
					},
				}, nil
			},
			CreateChatCompletionStreamFunc: func(ctx context.Context, req oai.ChatCompletionRequest) (*oai.ChatCompletionStream, error) {
				return nil, io.EOF // Not used in this test
			},
		}

		// Create initial history
		textContent, _ := gollem.NewTextContent("Initial")
		initialHistory := &gollem.History{
			Version: gollem.HistoryVersion,
			LLType:  gollem.LLMTypeOpenAI,
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
		session, _ := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-4")

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
