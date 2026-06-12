package compacter_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/gollem-dev/gollem/middleware/compacter"
	"github.com/gollem-dev/gollem/mock"
	"github.com/m-mizutani/goerr/v2"
	"github.com/m-mizutani/gt"
)

func TestCountMessageChars(t *testing.T) {
	testCases := []struct {
		name     string
		messages []gollem.Message
		expected int
	}{
		{
			name:     "empty messages",
			messages: []gollem.Message{},
			expected: 0,
		},
		{
			name: "single message",
			messages: []gollem.Message{
				createMessage(gollem.RoleUser, "Hello"),
			},
			expected: 5,
		},
		{
			name: "multiple messages",
			messages: []gollem.Message{
				createMessage(gollem.RoleUser, "Hello"),
				createMessage(gollem.RoleAssistant, "Hi there!"),
			},
			expected: 14, // "Hello" (5) + "Hi there!" (9)
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := compacter.CountMessageChars(tc.messages)
			gt.Equal(t, tc.expected, actual)
		})
	}
}

func TestExtractMessagesToCompact(t *testing.T) {
	testCases := []struct {
		name                 string
		messages             []gollem.Message
		targetChars          int
		expectedCompactLen   int
		expectedRemainingLen int
	}{
		{
			name:                 "empty messages",
			messages:             []gollem.Message{},
			targetChars:          10,
			expectedCompactLen:   0,
			expectedRemainingLen: 0,
		},
		{
			name: "target exceeds total",
			messages: []gollem.Message{
				createMessage(gollem.RoleUser, "Short"),
			},
			targetChars:          100,
			expectedCompactLen:   0,
			expectedRemainingLen: 1,
		},
		{
			name: "compact first message",
			messages: []gollem.Message{
				createMessage(gollem.RoleUser, "First message"),
				createMessage(gollem.RoleAssistant, "Second"),
			},
			targetChars:          10,
			expectedCompactLen:   1,
			expectedRemainingLen: 1,
		},
		{
			name: "compact multiple messages",
			messages: []gollem.Message{
				createMessage(gollem.RoleUser, "First"),
				createMessage(gollem.RoleAssistant, "Second"),
				createMessage(gollem.RoleUser, "Third"),
			},
			targetChars:          15,
			expectedCompactLen:   2,
			expectedRemainingLen: 1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			compact, remaining := compacter.ExtractMessagesToCompact(tc.messages, tc.targetChars)
			gt.Equal(t, tc.expectedCompactLen, len(compact))
			gt.Equal(t, tc.expectedRemainingLen, len(remaining))
		})
	}
}

func TestContentBlockMiddleware_TokenExceeded(t *testing.T) {
	ctx := context.Background()
	callCount := 0

	// Create mock LLM client for summarization
	mockClient := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return &mock.SessionMock{
				GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
					return &gollem.Response{
						Texts: []string{"This is a summary of the conversation."},
					}, nil
				},
			}, nil
		},
	}

	// Create middleware
	middleware := compacter.NewContentBlockMiddleware(
		mockClient,
		compacter.WithMaxRetries(2),
		compacter.WithCompactRatio(0.5),
	)

	// Create handler that returns token exceeded error on first call, then succeeds
	handler := func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
		callCount++
		if callCount == 1 {
			return nil, goerr.Wrap(gollem.ErrTokenSizeExceeded, "token limit exceeded", goerr.Tag(gollem.ErrTagTokenExceeded))
		}
		return &gollem.ContentResponse{
			Texts: []string{"Success after compaction"},
		}, nil
	}

	// Wrap handler with middleware
	wrappedHandler := middleware(handler)

	// Create request with history
	history := &gollem.History{
		LLType:  gollem.LLMTypeClaude,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			createMessage(gollem.RoleUser, "First message"),
			createMessage(gollem.RoleAssistant, "First response"),
			createMessage(gollem.RoleUser, "Second message"),
			createMessage(gollem.RoleAssistant, "Second response"),
		},
	}

	req := &gollem.ContentRequest{
		Inputs:  []gollem.Input{gollem.Text("New input")},
		History: history,
	}

	// Execute
	resp, err := wrappedHandler(ctx, req)

	// Verify
	gt.NoError(t, err)
	gt.NotNil(t, resp)
	gt.Equal(t, 2, callCount) // Called twice: first failed, second succeeded
	gt.Equal(t, 1, len(resp.Texts))
	gt.Equal(t, "Success after compaction", resp.Texts[0])
}

func TestContentBlockMiddleware_MaxRetriesExceeded(t *testing.T) {
	ctx := context.Background()
	callCount := 0

	// Create mock LLM client
	mockClient := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return &mock.SessionMock{
				GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
					return &gollem.Response{
						Texts: []string{"Summary"},
					}, nil
				},
			}, nil
		},
	}

	// Create middleware with max retries = 2
	middleware := compacter.NewContentBlockMiddleware(
		mockClient,
		compacter.WithMaxRetries(2),
	)

	// Handler that always returns token exceeded error
	handler := func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
		callCount++
		return nil, goerr.Wrap(gollem.ErrTokenSizeExceeded, "always exceed", goerr.Tag(gollem.ErrTagTokenExceeded))
	}

	wrappedHandler := middleware(handler)

	history := &gollem.History{
		LLType:  gollem.LLMTypeClaude,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			createMessage(gollem.RoleUser, "Message"),
		},
	}

	req := &gollem.ContentRequest{
		Inputs:  []gollem.Input{gollem.Text("input")},
		History: history,
	}

	// Execute
	resp, err := wrappedHandler(ctx, req)

	// Verify - should fail after retries
	gt.Error(t, err)
	gt.Nil(t, resp)
	gt.Equal(t, 3, callCount) // Initial + 2 retries
}

func TestContentBlockMiddleware_NoHistory(t *testing.T) {
	ctx := context.Background()

	mockClient := &mock.LLMClientMock{}

	middleware := compacter.NewContentBlockMiddleware(mockClient)

	handler := func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
		return nil, goerr.Wrap(gollem.ErrTokenSizeExceeded, "exceed", goerr.Tag(gollem.ErrTagTokenExceeded))
	}

	wrappedHandler := middleware(handler)

	req := &gollem.ContentRequest{
		Inputs:  []gollem.Input{gollem.Text("input")},
		History: nil, // No history
	}

	// Execute
	resp, err := wrappedHandler(ctx, req)

	// Should return error without retrying
	gt.Error(t, err)
	gt.Nil(t, resp)
}

func TestContentBlockMiddleware_NonTokenError(t *testing.T) {
	ctx := context.Background()
	callCount := 0

	mockClient := &mock.LLMClientMock{}

	middleware := compacter.NewContentBlockMiddleware(mockClient)

	// Handler that returns different error
	handler := func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
		callCount++
		return nil, goerr.New("some other error")
	}

	wrappedHandler := middleware(handler)

	req := &gollem.ContentRequest{
		Inputs: []gollem.Input{gollem.Text("input")},
	}

	// Execute
	resp, err := wrappedHandler(ctx, req)

	// Should return error immediately without retry
	gt.Error(t, err)
	gt.Nil(t, resp)
	gt.Equal(t, 1, callCount) // Only called once
}

func TestContentStreamMiddleware_TokenExceeded(t *testing.T) {
	ctx := context.Background()
	callCount := 0

	// Create mock LLM client
	mockClient := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return &mock.SessionMock{
				GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
					return &gollem.Response{
						Texts: []string{"Summary"},
					}, nil
				},
			}, nil
		},
	}

	middleware := compacter.NewContentStreamMiddleware(
		mockClient,
		compacter.WithMaxRetries(2),
	)

	// Handler that returns token exceeded on first call
	handler := func(ctx context.Context, req *gollem.ContentRequest) (<-chan *gollem.ContentResponse, error) {
		callCount++
		if callCount == 1 {
			return nil, goerr.Wrap(gollem.ErrTokenSizeExceeded, "exceed", goerr.Tag(gollem.ErrTagTokenExceeded))
		}
		ch := make(chan *gollem.ContentResponse, 1)
		ch <- &gollem.ContentResponse{Texts: []string{"Success"}}
		close(ch)
		return ch, nil
	}

	wrappedHandler := middleware(handler)

	history := &gollem.History{
		LLType:  gollem.LLMTypeClaude,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			createMessage(gollem.RoleUser, "Message"),
		},
	}

	req := &gollem.ContentRequest{
		Inputs:  []gollem.Input{gollem.Text("input")},
		History: history,
	}

	// Execute
	respChan, err := wrappedHandler(ctx, req)

	// Verify
	gt.NoError(t, err)
	gt.NotNil(t, respChan)
	gt.Equal(t, 2, callCount)
}

// Helper functions

func TestContentBlockMiddleware_CompactionHook(t *testing.T) {
	ctx := context.Background()
	callCount := 0
	var capturedEvent *compacter.CompactionEvent

	// Create mock LLM client
	mockClient := &mock.LLMClientMock{
		NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
			return &mock.SessionMock{
				GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
					// Return summary for compaction with token usage
					return &gollem.Response{
						Texts:       []string{"Compacted conversation summary"},
						InputToken:  100,
						OutputToken: 20,
					}, nil
				},
			}, nil
		},
	}

	// Create middleware with hook
	middleware := compacter.NewContentBlockMiddleware(
		mockClient,
		compacter.WithMaxRetries(2),
		compacter.WithCompactRatio(0.5),
		compacter.WithCompactionHook(func(ctx context.Context, event *compacter.CompactionEvent) {
			capturedEvent = event
		}),
	)

	// Create handler that returns token exceeded error on first call
	handler := func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
		callCount++
		if callCount == 1 {
			return nil, goerr.Wrap(
				goerr.New("token limit exceeded"),
				"API error",
				goerr.Tag(gollem.ErrTagTokenExceeded),
			)
		}
		return &gollem.ContentResponse{
			Texts: []string{"Success after compaction"},
		}, nil
	}

	// Wrap handler with middleware
	wrappedHandler := middleware(handler)

	// Create request with history
	history := &gollem.History{
		LLType:  gollem.LLMTypeClaude,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			createMessage(gollem.RoleUser, "First message"),
			createMessage(gollem.RoleAssistant, "First response"),
			createMessage(gollem.RoleUser, "Second message"),
			createMessage(gollem.RoleAssistant, "Second response"),
		},
	}

	req := &gollem.ContentRequest{
		Inputs:  []gollem.Input{gollem.Text("New input")},
		History: history,
	}

	// Execute
	resp, err := wrappedHandler(ctx, req)

	// Verify
	gt.NoError(t, err)
	gt.NotNil(t, resp)
	gt.Equal(t, 2, callCount) // Called twice: first failed, second succeeded

	// Verify hook was called with correct event data
	gt.NotNil(t, capturedEvent)
	gt.Equal(t, 1, capturedEvent.Attempt)
	gt.V(t, capturedEvent.OriginalDataSize > 0)
	gt.V(t, capturedEvent.CompactedDataSize > 0)
	gt.V(t, capturedEvent.CompactedDataSize < capturedEvent.OriginalDataSize)
	gt.Equal(t, 100, capturedEvent.InputTokens)
	gt.Equal(t, 20, capturedEvent.OutputTokens)
	gt.V(t, len(capturedEvent.Summary) > 0)
}

func TestContentBlockMiddleware_SummaryRoleAlternation(t *testing.T) {
	ctx := context.Background()

	testCases := []struct {
		name                string
		remainingFirstRole  gollem.MessageRole
		expectedSummaryRole gollem.MessageRole
	}{
		{
			name:                "remaining starts with assistant, summary should be user",
			remainingFirstRole:  gollem.RoleAssistant,
			expectedSummaryRole: gollem.RoleUser,
		},
		{
			name:                "remaining starts with user, summary should be assistant",
			remainingFirstRole:  gollem.RoleUser,
			expectedSummaryRole: gollem.RoleAssistant,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var compactedHistory *gollem.History

			// Create mock LLM client
			mockClient := &mock.LLMClientMock{
				NewSessionFunc: func(ctx context.Context, options ...gollem.SessionOption) (gollem.Session, error) {
					return &mock.SessionMock{
						GenerateFunc: func(ctx context.Context, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, error) {
							return &gollem.Response{
								Texts:       []string{"Summary of conversation"},
								InputToken:  50,
								OutputToken: 10,
							}, nil
						},
					}, nil
				},
			}

			// Create middleware with high compact ratio to compact most messages
			middleware := compacter.NewContentBlockMiddleware(
				mockClient,
				compacter.WithCompactRatio(0.9), // Compact 90% to leave only last message
			)

			// Create handler that captures the compacted history
			callCount := 0
			handler := func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
				callCount++
				if callCount == 1 {
					// First call: return token exceeded error
					return nil, goerr.Wrap(
						goerr.New("token limit exceeded"),
						"API error",
						goerr.Tag(gollem.ErrTagTokenExceeded),
					)
				}
				// Second call after compaction: capture and return success
				compactedHistory = req.History
				return &gollem.ContentResponse{
					Texts: []string{"Success"},
				}, nil
			}

			wrappedHandler := middleware(handler)

			// Create history with messages that will be compacted
			history := &gollem.History{
				LLType:  gollem.LLMTypeClaude,
				Version: gollem.HistoryVersion,
				Messages: []gollem.Message{
					createMessage(gollem.RoleUser, "First message to be compacted"),
					createMessage(gollem.RoleAssistant, "First response to be compacted"),
					createMessage(gollem.RoleUser, "Second message to be compacted"),
					createMessage(gollem.RoleAssistant, "Second response to be compacted"),
					createMessage(tc.remainingFirstRole, "Remaining message"),
				},
			}

			req := &gollem.ContentRequest{
				Inputs:  []gollem.Input{gollem.Text("New input")},
				History: history,
			}

			// Execute
			_, err := wrappedHandler(ctx, req)
			gt.NoError(t, err)

			// Verify compacted history structure
			gt.NotNil(t, compactedHistory)
			gt.V(t, len(compactedHistory.Messages) >= 2) // Summary + at least one remaining

			// Verify summary role maintains alternation
			summaryMessage := compactedHistory.Messages[0]
			gt.Equal(t, tc.expectedSummaryRole, summaryMessage.Role)

			// Verify alternation pattern
			nextMessage := compactedHistory.Messages[1]
			gt.NotEqual(t, summaryMessage.Role, nextMessage.Role)
		})
	}
}

func createMessage(role gollem.MessageRole, text string) gollem.Message {
	content, _ := gollem.NewTextContent(text)
	return gollem.Message{
		Role:     role,
		Contents: []gollem.MessageContent{content},
	}
}

// Integration tests with real LLM clients
func TestCompactionWithRealLLM(t *testing.T) {
	t.Parallel()

	testFn := func(t *testing.T, newClient func(t *testing.T) (gollem.LLMClient, error)) {
		client, err := newClient(t)
		gt.NoError(t, err)

		ctx := context.Background()
		callCount := 0
		var capturedEvent *compacter.CompactionEvent

		// Create middleware with real LLM client
		middleware := compacter.NewContentBlockMiddleware(
			client,
			compacter.WithMaxRetries(2),
			compacter.WithCompactRatio(0.7),
			compacter.WithCompactionHook(func(ctx context.Context, event *compacter.CompactionEvent) {
				capturedEvent = event
			}),
		)

		// Create handler that returns token exceeded error on first call
		handler := func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
			callCount++
			if callCount == 1 {
				return nil, goerr.Wrap(
					goerr.New("token limit exceeded"),
					"API error",
					goerr.Tag(gollem.ErrTagTokenExceeded),
				)
			}
			return &gollem.ContentResponse{
				Texts: []string{"Success after compaction"},
			}, nil
		}

		// Wrap handler with middleware
		wrappedHandler := middleware(handler)

		// Create request with conversation history containing important information
		history := &gollem.History{
			LLType:  gollem.LLMTypeClaude,
			Version: gollem.HistoryVersion,
			Messages: []gollem.Message{
				createMessage(gollem.RoleUser, "My name is Alice and I live in Tokyo."),
				createMessage(gollem.RoleAssistant, "Nice to meet you, Alice from Tokyo!"),
				createMessage(gollem.RoleUser, "I work as a software engineer at a tech company."),
				createMessage(gollem.RoleAssistant, "That's great! Software engineering is an exciting field."),
				createMessage(gollem.RoleUser, "I enjoy playing tennis on weekends."),
				createMessage(gollem.RoleAssistant, "Tennis is a wonderful hobby!"),
			},
		}

		req := &gollem.ContentRequest{
			Inputs:  []gollem.Input{gollem.Text("What do you know about me?")},
			History: history,
		}

		// Execute
		resp, err := wrappedHandler(ctx, req)

		// Verify
		gt.NoError(t, err)
		gt.NotNil(t, resp)
		gt.Equal(t, 2, callCount) // Called twice: first failed, second succeeded

		// Verify compaction hook was called
		gt.NotNil(t, capturedEvent)
		gt.Equal(t, 1, capturedEvent.Attempt)
		gt.V(t, capturedEvent.OriginalDataSize > 0)
		gt.V(t, capturedEvent.CompactedDataSize > 0)
		gt.V(t, capturedEvent.CompactedDataSize < capturedEvent.OriginalDataSize)
		gt.V(t, capturedEvent.InputTokens > 0)
		gt.V(t, capturedEvent.OutputTokens > 0)
		gt.V(t, len(capturedEvent.Summary) > 0)

		// Verify that important information is preserved in the summary
		summary := capturedEvent.Summary
		importantKeywords := []string{"Alice", "Tokyo", "engineer", "tennis"}
		preservedCount := 0
		for _, keyword := range importantKeywords {
			if containsIgnoreCase(summary, keyword) {
				preservedCount++
			}
		}

		// At least 60% of important information should be preserved
		preservationRate := float64(preservedCount) / float64(len(importantKeywords))
		if preservationRate < 0.6 {
			t.Errorf("preservation rate should be >= 60%%, got %.2f%% (%d/%d)", preservationRate*100, preservedCount, len(importantKeywords))
		}
	}

	t.Run("OpenAI", func(t *testing.T) {
		t.Parallel()
		apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
		if !ok {
			t.Skip("TEST_OPENAI_API_KEY is not set")
		}
		testFn(t, func(t *testing.T) (gollem.LLMClient, error) {
			return openai.New(context.Background(), apiKey)
		})
	})

	t.Run("Claude", func(t *testing.T) {
		apiKey, ok := os.LookupEnv("TEST_CLAUDE_API_KEY")
		if !ok {
			t.Skip("TEST_CLAUDE_API_KEY is not set")
		}
		testFn(t, func(t *testing.T) (gollem.LLMClient, error) {
			return claude.New(context.Background(), apiKey)
		})
	})

	t.Run("Gemini", func(t *testing.T) {
		t.Parallel()
		projectID, ok := os.LookupEnv("TEST_GCP_PROJECT_ID")
		if !ok {
			t.Skip("TEST_GCP_PROJECT_ID is not set")
		}
		location, ok := os.LookupEnv("TEST_GCP_LOCATION")
		if !ok {
			t.Skip("TEST_GCP_LOCATION is not set")
		}
		testFn(t, func(t *testing.T) (gollem.LLMClient, error) {
			return gemini.New(context.Background(), projectID, location)
		})
	})

	t.Run("Vertex AI Claude", func(t *testing.T) {
		t.Parallel()
		projectID, ok := os.LookupEnv("TEST_GCP_CLAUDE_PROJECT_ID")
		if !ok {
			t.Skip("TEST_GCP_CLAUDE_PROJECT_ID is not set")
		}
		location, ok := os.LookupEnv("TEST_GCP_CLAUDE_LOCATION")
		if !ok {
			t.Skip("TEST_GCP_CLAUDE_LOCATION is not set")
		}
		testFn(t, func(t *testing.T) (gollem.LLMClient, error) {
			return claude.NewWithVertex(context.Background(), location, projectID)
		})
	})
}

func containsIgnoreCase(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}
