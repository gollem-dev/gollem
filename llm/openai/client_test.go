package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
	"github.com/m-mizutani/gt"
	openaiapi "github.com/sashabaranov/go-openai"
)

const (
	testTimeout   = 30 * time.Second
	maxTestTokens = 2048
)

func TestOpenAIContentGenerate(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := openai.New(ctx, apiKey)
	gt.NoError(t, err)

	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	result, err := session.Generate(ctx, []gollem.Input{gollem.Text("Say hello in one word")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err)
	gt.Array(t, result.Texts).Length(1).Required()
	gt.Value(t, len(result.Texts[0])).NotEqual(0)
}

func TestTokenLimitErrorOptions(t *testing.T) {
	type testCase struct {
		name   string
		err    error
		hasTag bool
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			opts := openai.TokenLimitErrorOptions(tc.err)
			if tc.hasTag {
				gt.NotEqual(t, 0, len(opts))
			} else {
				gt.Equal(t, 0, len(opts))
			}
		}
	}

	t.Run("token exceeded error", runTest(testCase{
		name: "context_length_exceeded",
		err: &openaiapi.APIError{
			Type:    "invalid_request_error",
			Code:    "context_length_exceeded",
			Message: "This model's maximum context length is 128000 tokens. However, your messages resulted in 150000 tokens.",
		},
		hasTag: true,
	}))

	t.Run("different error type", runTest(testCase{
		name: "different type",
		err: &openaiapi.APIError{
			Type:    "authentication_error",
			Code:    "invalid_api_key",
			Message: "Invalid API key",
		},
		hasTag: false,
	}))

	t.Run("different error code", runTest(testCase{
		name: "different code",
		err: &openaiapi.APIError{
			Type:    "invalid_request_error",
			Code:    "invalid_model",
			Message: "The model does not exist",
		},
		hasTag: false,
	}))

	t.Run("code is not string", runTest(testCase{
		name: "code as int",
		err: &openaiapi.APIError{
			Type:    "invalid_request_error",
			Code:    12345,
			Message: "Some error",
		},
		hasTag: false,
	}))

	t.Run("nil error", runTest(testCase{
		name:   "nil error",
		err:    nil,
		hasTag: false,
	}))

	t.Run("non-APIError", runTest(testCase{
		name:   "generic error",
		err:    errors.New("some error"),
		hasTag: false,
	}))
}

func TestOpenAITokenLimitErrorIntegration(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}

	// Only run if explicitly requested via environment variable
	if os.Getenv("TEST_TOKEN_LIMIT_ERROR") != "true" {
		t.Skip("TEST_TOKEN_LIMIT_ERROR is not set to true")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	// Use gpt-5 (default model) which has 128k context limit
	client, err := openai.New(ctx, apiKey)
	gt.NoError(t, err)

	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	// Create a very long prompt to exceed token limit
	// gpt-5 may have larger context limit, so we create much more text
	// Approximately 1 token = 4 characters, aim for ~300k+ tokens
	longText := strings.Repeat("This is a test sentence to make the prompt very long. ", 25000)

	_, err = session.Generate(ctx, []gollem.Input{gollem.Text(longText)})
	gt.Error(t, err)

	// Log error details for debugging
	t.Logf("Error: %+v", err)
	t.Logf("Error tags: %v", goerr.Tags(err))

	// Verify the error has the token exceeded tag
	gt.True(t, goerr.HasTag(err, gollem.ErrTagTokenExceeded))
}

// TestPerCallGenerateOptions verifies that per-call GenerateOption overrides
// actually change the API request. A text-mode session gets a per-call
// ResponseSchema, and the response must be valid JSON matching the schema.
func TestPerCallGenerateOptions(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := openai.New(ctx, apiKey)
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

	// Per-call option should force JSON schema output
	resp, err := session.Generate(ctx,
		[]gollem.Input{gollem.Text("Name a color.")},
		gollem.WithGenerateResponseSchema(schema),
		gollem.WithMaxTokens(maxTestTokens),
	)
	gt.NoError(t, err)
	gt.True(t, len(resp.Texts) > 0)

	// The response must be valid JSON
	var parsed map[string]any
	gt.NoError(t, json.Unmarshal([]byte(resp.Texts[0]), &parsed))
	gt.True(t, parsed["name"] != nil)
}

// TestDefaultOptionsLive verifies that a client created without generation
// options can call models that accept different reasoning_effort and
// verbosity values, and that each model follows the session system prompt.
func TestDefaultOptionsLive(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}

	runTest := func(model string) func(t *testing.T) {
		return func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
			defer cancel()

			client, err := openai.New(ctx, apiKey, openai.WithModel(model))
			gt.NoError(t, err).Required()
			session, err := client.NewSession(ctx, gollem.WithSessionSystemPrompt(
				"Whatever the user says, reply with exactly the single word PINEAPPLE and nothing else."))
			gt.NoError(t, err).Required()

			resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("Say hello in one word")}, gollem.WithMaxTokens(maxTestTokens))
			gt.NoError(t, err).Required()
			gt.A(t, resp.Texts).Longer(0).Required()
			gt.S(t, strings.ToUpper(strings.Join(resp.Texts, ""))).Contains("PINEAPPLE")
		}
	}

	t.Run("default model", runTest(openai.DefaultModel))
	t.Run("gpt-5.6", runTest("gpt-5.6"))
	t.Run("gpt-4.1", runTest("gpt-4.1"))
}

// sentRequest is the part of a chat completion request body that the request
// tests inspect.
type sentRequest struct {
	Messages []struct {
		Role string `json:"role"`
		// Content is a string for system messages and an array of parts for
		// user messages.
		Content json.RawMessage `json:"content"`
	} `json:"messages"`
	ReasoningEffort *string `json:"reasoning_effort"`
	Verbosity       *string `json:"verbosity"`
	Stream          bool    `json:"stream"`
}

// newRecordingServer returns the URL of a server that answers every chat
// completion with "ok", as JSON or as server-sent events for a streaming
// request, and a function returning the request bodies received.
func newRecordingServer(t *testing.T) (string, func() []sentRequest) {
	t.Helper()
	var mu sync.Mutex
	var received []sentRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req sentRequest
		gt.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		mu.Lock()
		received = append(received, req)
		mu.Unlock()

		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			_, err := io.WriteString(w, `data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-test",`+
				`"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":null}`+"\n\n"+
				`data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-test","choices":[],`+
				`"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`+"\n\n"+
				"data: [DONE]\n\n")
			gt.NoError(t, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, err := io.WriteString(w, `{"id":"chatcmpl-1","object":"chat.completion","model":"gpt-test",`+
			`"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`)
		gt.NoError(t, err)
	}))
	t.Cleanup(srv.Close)

	return srv.URL, func() []sentRequest {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(received)
	}
}

func TestSystemPromptIsSent(t *testing.T) {
	type testCase struct {
		clientPrompt  string
		sessionPrompt string
		want          string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			url, received := newRecordingServer(t)
			ctx := context.Background()

			client, err := openai.New(ctx, "test-key", openai.WithBaseURL(url), openai.WithSystemPrompt(tc.clientPrompt))
			gt.NoError(t, err).Required()
			var opts []gollem.SessionOption
			if tc.sessionPrompt != "" {
				opts = append(opts, gollem.WithSessionSystemPrompt(tc.sessionPrompt))
			}
			session, err := client.NewSession(ctx, opts...)
			gt.NoError(t, err).Required()

			_, err = session.Generate(ctx, []gollem.Input{gollem.Text("first")})
			gt.NoError(t, err).Required()
			_, err = session.Generate(ctx, []gollem.Input{gollem.Text("second")})
			gt.NoError(t, err).Required()

			reqs := received()
			gt.A(t, reqs).Length(2).Required()
			for _, req := range reqs {
				var systems []string
				for _, m := range req.Messages {
					if m.Role == "system" {
						var content string
						gt.NoError(t, json.Unmarshal(m.Content, &content))
						systems = append(systems, content)
					}
				}
				if tc.want == "" {
					gt.A(t, systems).Length(0)
					continue
				}
				// Exactly one system message, at the head, on every call.
				gt.Equal(t, []string{tc.want}, systems)
				gt.Equal(t, "system", req.Messages[0].Role)
			}
			// The second request carries first, ok, second after the system message.
			wantLen := 3
			if tc.want != "" {
				wantLen = 4
			}
			gt.A(t, reqs[1].Messages).Length(wantLen)

			history, err := session.History()
			gt.NoError(t, err).Required()
			for _, m := range history.Messages {
				gt.NotEqual(t, gollem.RoleSystem, m.Role)
			}
		}
	}

	t.Run("session prompt", runTest(testCase{sessionPrompt: "session rules", want: "session rules"}))
	t.Run("client prompt when the session sets none", runTest(testCase{clientPrompt: "client rules", want: "client rules"}))
	t.Run("session prompt overrides client prompt", runTest(testCase{clientPrompt: "client rules", sessionPrompt: "session rules", want: "session rules"}))
	t.Run("no prompt", runTest(testCase{}))
}

// TestClientSystemPromptReachesEveryConsumer verifies that the client system
// prompt, used when the session sets none, is the prompt that streaming
// requests send, middleware receives, trace records, and CountToken counts.
func TestClientSystemPromptReachesEveryConsumer(t *testing.T) {
	const prompt = "client rules that are long enough to change the token count"
	url, received := newRecordingServer(t)
	ctx := context.Background()

	newSession := func(t *testing.T, systemPrompt string, opts ...gollem.SessionOption) gollem.Session {
		t.Helper()
		client, err := openai.New(ctx, "test-key", openai.WithBaseURL(url), openai.WithSystemPrompt(systemPrompt))
		gt.NoError(t, err).Required()
		session, err := client.NewSession(ctx, opts...)
		gt.NoError(t, err).Required()
		return session
	}

	t.Run("Stream sends it once at the head of every request", func(t *testing.T) {
		var middlewarePrompts []string
		session := newSession(t, prompt, gollem.WithSessionContentStreamMiddleware(
			func(next gollem.ContentStreamHandler) gollem.ContentStreamHandler {
				return func(ctx context.Context, req *gollem.ContentRequest) (<-chan *gollem.ContentResponse, error) {
					middlewarePrompts = append(middlewarePrompts, req.SystemPrompt)
					return next(ctx, req)
				}
			}))

		rec := trace.New()
		traceCtx := trace.WithHandler(rec.StartAgentExecute(ctx), rec)
		before := len(received())
		for _, text := range []string{"first", "second"} {
			ch, err := session.Stream(traceCtx, []gollem.Input{gollem.Text(text)})
			gt.NoError(t, err).Required()
			for resp := range ch {
				gt.NoError(t, resp.Error)
			}
		}
		rec.EndAgentExecute(traceCtx, nil)

		reqs := received()[before:]
		gt.A(t, reqs).Length(2).Required()
		for _, req := range reqs {
			gt.True(t, req.Stream)
			var systems int
			for _, m := range req.Messages {
				if m.Role == "system" {
					systems++
				}
			}
			gt.Equal(t, 1, systems)
			gt.Equal(t, "system", req.Messages[0].Role)
			var content string
			gt.NoError(t, json.Unmarshal(req.Messages[0].Content, &content))
			gt.Equal(t, prompt, content)
		}
		gt.Equal(t, []string{prompt, prompt}, middlewarePrompts)

		var tracePrompts []string
		for _, span := range rec.Trace().RootSpan.Children {
			if span.Kind == trace.SpanKindLLMCall {
				tracePrompts = append(tracePrompts, span.LLMCall.Request.SystemPrompt)
			}
		}
		gt.Equal(t, []string{prompt, prompt}, tracePrompts)

		history, err := session.History()
		gt.NoError(t, err).Required()
		for _, m := range history.Messages {
			gt.NotEqual(t, gollem.RoleSystem, m.Role)
		}
	})

	t.Run("a middleware replacement is sent and traced instead", func(t *testing.T) {
		const replaced = "rules set by middleware"
		session := newSession(t, prompt, gollem.WithSessionContentBlockMiddleware(
			func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
				return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
					req.SystemPrompt = replaced
					return next(ctx, req)
				}
			}))

		rec := trace.New()
		traceCtx := trace.WithHandler(rec.StartAgentExecute(ctx), rec)
		before := len(received())
		_, err := session.Generate(traceCtx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		rec.EndAgentExecute(traceCtx, nil)

		reqs := received()[before:]
		gt.A(t, reqs).Length(1).Required()
		gt.Equal(t, "system", reqs[0].Messages[0].Role)
		var content string
		gt.NoError(t, json.Unmarshal(reqs[0].Messages[0].Content, &content))
		gt.Equal(t, replaced, content)

		var tracePrompts []string
		for _, span := range rec.Trace().RootSpan.Children {
			if span.Kind == trace.SpanKindLLMCall {
				tracePrompts = append(tracePrompts, span.LLMCall.Request.SystemPrompt)
			}
		}
		gt.Equal(t, []string{replaced}, tracePrompts)
	})

	t.Run("CountToken counts it", func(t *testing.T) {
		input := []gollem.Input{gollem.Text("hi")}
		withPrompt, err := newSession(t, prompt).CountToken(ctx, input...)
		gt.NoError(t, err).Required()
		withoutPrompt, err := newSession(t, "").CountToken(ctx, input...)
		gt.NoError(t, err).Required()
		gt.True(t, withPrompt > withoutPrompt)
	})
}

func TestGenerationParametersAreSentOnlyWhenSet(t *testing.T) {
	type testCase struct {
		options       []openai.Option
		wantReasoning *string
		wantVerbosity *string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			url, received := newRecordingServer(t)
			ctx := context.Background()

			client, err := openai.New(ctx, "test-key", append([]openai.Option{openai.WithBaseURL(url)}, tc.options...)...)
			gt.NoError(t, err).Required()
			session, err := client.NewSession(ctx)
			gt.NoError(t, err).Required()
			_, err = session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
			gt.NoError(t, err).Required()

			reqs := received()
			gt.A(t, reqs).Length(1).Required()
			gt.Equal(t, tc.wantReasoning, reqs[0].ReasoningEffort)
			gt.Equal(t, tc.wantVerbosity, reqs[0].Verbosity)
		}
	}

	low, high := "low", "high"
	t.Run("not sent by default", runTest(testCase{}))
	t.Run("sent when set", runTest(testCase{
		options:       []openai.Option{openai.WithReasoningEffort("high"), openai.WithVerbosity("low")},
		wantReasoning: &high,
		wantVerbosity: &low,
	}))
}

// TestWithBaseURL tests the WithBaseURL option functionality for OpenAI
func TestWithBaseURL(t *testing.T) {
	t.Run("default baseURL", func(t *testing.T) {
		client, err := openai.New(context.Background(), "test-key", openai.WithBaseURL(""))
		gt.NoError(t, err)
		gt.Equal(t, "", openai.GetBaseURL(client))
	})

	t.Run("custom baseURL", func(t *testing.T) {
		customURL := "https://api.custom-openai.com"
		client, err := openai.New(context.Background(), "test-key", openai.WithBaseURL(customURL))
		gt.NoError(t, err)
		gt.Equal(t, customURL, openai.GetBaseURL(client))
	})

	t.Run("empty baseURL after custom", func(t *testing.T) {
		// Test that empty baseURL overrides previous setting
		client1, err1 := openai.New(context.Background(), "test-key", openai.WithBaseURL("https://first.com"))
		gt.NoError(t, err1)
		gt.Equal(t, "https://first.com", openai.GetBaseURL(client1))

		// Apply empty baseURL after custom one
		client2, err2 := openai.New(context.Background(), "test-key",
			openai.WithBaseURL("https://first.com"),
			openai.WithBaseURL(""))
		gt.NoError(t, err2)
		gt.Equal(t, "", openai.GetBaseURL(client2)) // Should be empty, not first URL
	})
}

// TestChatReasoningEffort verifies that the chat path sends reasoning_effort
// only when WithReasoningEffort is given; some models reject every default.
func TestChatReasoningEffort(t *testing.T) {
	runTest := func(options []openai.Option, expected any) func(t *testing.T) {
		return func(t *testing.T) {
			var sent map[string]any
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/chat/completions" {
					http.Error(w, "unexpected path "+r.URL.Path, http.StatusNotFound)
					return
				}
				if err := json.NewDecoder(r.Body).Decode(&sent); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"id":"c1","object":"chat.completion","model":"gpt-test",
					"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
					"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`)
			}))
			defer srv.Close()

			options = append([]openai.Option{openai.WithBaseURL(srv.URL + "/v1")}, options...)
			client, err := openai.New(context.Background(), "test-key", options...)
			gt.NoError(t, err).Required()
			session, err := client.NewSession(context.Background())
			gt.NoError(t, err).Required()
			_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")})
			gt.NoError(t, err).Required()

			effort, ok := sent["reasoning_effort"]
			if expected == nil {
				gt.False(t, ok)
				return
			}
			gt.Equal(t, expected, effort)
		}
	}

	t.Run("not sent without the option", runTest(nil, nil))
	t.Run("sent when set", runTest([]openai.Option{openai.WithReasoningEffort("high")}, any("high")))
}

func TestResponsesAPIRejectsPenalties(t *testing.T) {
	runTest := func(options ...openai.Option) func(t *testing.T) {
		return func(t *testing.T) {
			options = append(options, openai.WithResponsesAPI())
			_, err := openai.New(context.Background(), "test-key", options...)
			gt.True(t, errors.Is(err, gollem.ErrInvalidParameter))
		}
	}

	t.Run("presence penalty", runTest(openai.WithPresencePenalty(0.5)))
	t.Run("frequency penalty", runTest(openai.WithFrequencyPenalty(0.5)))
}

func TestOpenaiMessagesToTraceMessages(t *testing.T) {
	type testCase struct {
		messages []openaiapi.ChatCompletionMessage
		expected []trace.Message
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			result := openai.OpenaiMessagesToTraceMessages(tc.messages)
			gt.Equal(t, tc.expected, result)
		}
	}

	t.Run("user text message", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{Role: openaiapi.ChatMessageRoleUser, Content: "hello world"},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewTextContent("hello world"),
			}},
		},
	}))

	t.Run("system message", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{Role: openaiapi.ChatMessageRoleSystem, Content: "you are helpful"},
		},
		expected: []trace.Message{
			{Role: "system", Contents: []trace.MessageContent{
				trace.NewTextContent("you are helpful"),
			}},
		},
	}))

	t.Run("assistant with tool calls", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{
				Role: openaiapi.ChatMessageRoleAssistant,
				ToolCalls: []openaiapi.ToolCall{
					{
						ID:   "call-1",
						Type: openaiapi.ToolTypeFunction,
						Function: openaiapi.FunctionCall{
							Name:      "search",
							Arguments: `{"q":"test"}`,
						},
					},
				},
			},
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewToolCallContent("call-1", "search", map[string]any{"q": "test"}),
			}},
		},
	}))

	t.Run("tool response", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{
				Role:       openaiapi.ChatMessageRoleTool,
				Content:    "search result",
				ToolCallID: "call-1",
			},
		},
		expected: []trace.Message{
			{Role: "tool", Contents: []trace.MessageContent{
				{Type: "tool_response", ToolCallID: "call-1", Text: "search result"},
			}},
		},
	}))

	t.Run("multi content with image URL", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{
				Role: openaiapi.ChatMessageRoleUser,
				MultiContent: []openaiapi.ChatMessagePart{
					{Type: openaiapi.ChatMessagePartTypeText, Text: "describe this"},
					{Type: openaiapi.ChatMessagePartTypeImageURL, ImageURL: &openaiapi.ChatMessageImageURL{URL: "https://example.com/img.png"}},
				},
			},
		},
		expected: []trace.Message{
			{Role: "user", Contents: []trace.MessageContent{
				trace.NewTextContent("describe this"),
				{Type: "image", URL: "https://example.com/img.png"},
			}},
		},
	}))

	t.Run("assistant text with tool calls", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{
				Role:    openaiapi.ChatMessageRoleAssistant,
				Content: "Let me search",
				ToolCalls: []openaiapi.ToolCall{
					{
						ID:   "call-1",
						Type: openaiapi.ToolTypeFunction,
						Function: openaiapi.FunctionCall{
							Name:      "search",
							Arguments: `{"q":"test"}`,
						},
					},
				},
			},
		},
		expected: []trace.Message{
			{Role: "assistant", Contents: []trace.MessageContent{
				trace.NewTextContent("Let me search"),
				trace.NewToolCallContent("call-1", "search", map[string]any{"q": "test"}),
			}},
		},
	}))

	t.Run("multiple messages", runTest(testCase{
		messages: []openaiapi.ChatCompletionMessage{
			{Role: openaiapi.ChatMessageRoleUser, Content: "hello"},
			{Role: openaiapi.ChatMessageRoleAssistant, Content: "hi"},
			{Role: openaiapi.ChatMessageRoleUser, Content: "how are you"},
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
		messages: []openaiapi.ChatCompletionMessage{},
		expected: nil,
	}))
}

func TestThinkingContentExtraction(t *testing.T) {
	t.Run("non-streaming response with reasoning content", func(t *testing.T) {
		// Create a mock API client that returns reasoning content
		mockClient := &apiClientMock{
			CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
				return openaiapi.ChatCompletionResponse{
					Model: "gpt-5",
					Choices: []openaiapi.ChatCompletionChoice{
						{
							Message: openaiapi.ChatCompletionMessage{
								Role:             openaiapi.ChatMessageRoleAssistant,
								Content:          "This is the final answer",
								ReasoningContent: "Let me think through this step by step...",
							},
							FinishReason: openaiapi.FinishReasonStop,
						},
					},
					Usage: openaiapi.Usage{
						PromptTokens:     10,
						CompletionTokens: 20,
					},
				}, nil
			},
		}

		cfg := gollem.NewSessionConfig()
		session, err := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-5")
		gt.NoError(t, err)

		result, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("Test input")})
		gt.NoError(t, err)
		gt.Equal(t, []string{"This is the final answer"}, result.Texts)
		gt.Equal(t, []string{"Let me think through this step by step..."}, result.Thoughts)
		gt.Equal(t, 10, result.InputToken)
		gt.Equal(t, 20, result.OutputToken)
	})

	t.Run("non-streaming response without reasoning content", func(t *testing.T) {
		// Create a mock API client that returns only text content
		mockClient := &apiClientMock{
			CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
				return openaiapi.ChatCompletionResponse{
					Model: "gpt-5",
					Choices: []openaiapi.ChatCompletionChoice{
						{
							Message: openaiapi.ChatCompletionMessage{
								Role:    openaiapi.ChatMessageRoleAssistant,
								Content: "This is the answer",
							},
							FinishReason: openaiapi.FinishReasonStop,
						},
					},
					Usage: openaiapi.Usage{
						PromptTokens:     10,
						CompletionTokens: 15,
					},
				}, nil
			},
		}

		cfg := gollem.NewSessionConfig()
		session, err := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-5")
		gt.NoError(t, err)

		result, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("Test input")})
		gt.NoError(t, err)
		gt.Equal(t, []string{"This is the answer"}, result.Texts)
		gt.Equal(t, []string{}, result.Thoughts) // Should be empty slice
		gt.Equal(t, 10, result.InputToken)
		gt.Equal(t, 15, result.OutputToken)
	})
}

// TestOpenAITraceRequestMessagesNewTurnOnly verifies that the trace's
// LLMRequest.Messages contains only messages newly added in this turn,
// not the entire conversation history that was actually sent to the API.
func TestOpenAITraceRequestMessagesNewTurnOnly(t *testing.T) {
	makeHistory := func() *gollem.History {
		userContent, err := gollem.NewTextContent("previous question")
		gt.NoError(t, err)
		assistantContent, err := gollem.NewTextContent("previous answer")
		gt.NoError(t, err)
		return &gollem.History{
			Version: gollem.HistoryVersion,
			LLType:  gollem.LLMTypeOpenAI,
			Messages: []gollem.Message{
				{Role: gollem.RoleUser, Contents: []gollem.MessageContent{userContent}},
				{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{assistantContent}},
			},
		}
	}

	findLLMSpan := func(t *testing.T, root *trace.Span) *trace.Span {
		t.Helper()
		for _, child := range root.Children {
			if child.Kind == trace.SpanKindLLMCall {
				return child
			}
		}
		t.Fatal("llm_call span not found")
		return nil
	}

	t.Run("Generate", func(t *testing.T) {
		var sentMessages []openaiapi.ChatCompletionMessage
		mockClient := &apiClientMock{
			CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
				sentMessages = req.Messages
				return openaiapi.ChatCompletionResponse{
					Model: "gpt-5",
					Choices: []openaiapi.ChatCompletionChoice{
						{
							Message: openaiapi.ChatCompletionMessage{
								Role:    openaiapi.ChatMessageRoleAssistant,
								Content: "ok",
							},
							FinishReason: openaiapi.FinishReasonStop,
						},
					},
				}, nil
			},
		}

		cfg := gollem.NewSessionConfig(gollem.WithSessionHistory(makeHistory()))
		session, err := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-5")
		gt.NoError(t, err)

		rec := trace.New()
		ctx := rec.StartAgentExecute(context.Background())
		ctx = trace.WithHandler(ctx, rec)

		_, err = session.Generate(ctx, []gollem.Input{gollem.Text("new question")})
		gt.NoError(t, err)
		rec.EndAgentExecute(ctx, nil)

		// Sanity check: the actual API request includes the full history.
		gt.N(t, len(sentMessages)).Equal(3)

		// But the trace records only the new turn.
		llmSpan := findLLMSpan(t, rec.Trace().RootSpan)
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
	})

	// Note: Stream is not covered by a mock-based test because
	// *openai.ChatCompletionStream cannot be constructed outside the SDK
	// (it has unexported fields). The Stream code path uses the same
	// newMessages variable and the same openaiMessagesToTraceMessages call
	// site as Generate, so the Generate test above is structurally
	// equivalent for the trace-delta invariant.
}

func TestOpenAICacheTokenObservation(t *testing.T) {
	runTest := func(details *openaiapi.PromptTokensDetails, wantCacheRead int) func(t *testing.T) {
		return func(t *testing.T) {
			mockClient := &apiClientMock{
				CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
					return openaiapi.ChatCompletionResponse{
						Choices: []openaiapi.ChatCompletionChoice{
							{Message: openaiapi.ChatCompletionMessage{Content: "ok", Role: openaiapi.ChatMessageRoleAssistant}},
						},
						Usage: openaiapi.Usage{
							PromptTokens:        200,
							CompletionTokens:    10,
							PromptTokensDetails: details,
						},
					}, nil
				},
			}

			cfg := gollem.NewSessionConfig()
			session, err := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-4")
			gt.NoError(t, err)

			resp, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")})
			gt.NoError(t, err)

			// OpenAI's PromptTokens already includes cached tokens, so InputToken stays total.
			gt.Equal(t, 200, resp.InputToken)
			gt.Equal(t, wantCacheRead, resp.CacheReadInputToken)
			// Cache writes are read from the HTTP response body, which the
			// mocked apiClient bypasses.
			gt.Equal(t, 0, resp.CacheCreationInputToken)
		}
	}

	t.Run("reports cached prompt tokens", runTest(&openaiapi.PromptTokensDetails{CachedTokens: 150}, 150))
	t.Run("nil details yields zero", runTest(nil, 0))
}

func TestOpenAIFinishReason(t *testing.T) {
	type testCase struct {
		content string
		reason  openaiapi.FinishReason
	}
	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			mockClient := &apiClientMock{
				CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
					return openaiapi.ChatCompletionResponse{
						Choices: []openaiapi.ChatCompletionChoice{{
							Message:      openaiapi.ChatCompletionMessage{Content: tc.content, Role: openaiapi.ChatMessageRoleAssistant},
							FinishReason: tc.reason,
						}},
						Usage: openaiapi.Usage{PromptTokens: 20, CompletionTokens: 3},
					}, nil
				},
			}
			session, err := openai.NewSessionWithAPIClient(mockClient, gollem.NewSessionConfig(), "gpt-4")
			gt.NoError(t, err).Required()

			rec := trace.New()
			ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
			resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
			gt.NoError(t, err).Required()
			gt.Equal(t, string(tc.reason), resp.FinishReason)
			gt.Equal(t, resp.FinishReason, findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response.FinishReason)
			if tc.content == "" {
				gt.A(t, resp.Texts).Length(0)
			} else {
				gt.Equal(t, []string{tc.content}, resp.Texts)
			}
		}
	}
	t.Run("stop", runTest(testCase{content: "ok", reason: openaiapi.FinishReasonStop}))
	t.Run("length", runTest(testCase{content: "ok", reason: openaiapi.FinishReasonLength}))
	t.Run("content_filter", runTest(testCase{content: "ok", reason: openaiapi.FinishReasonContentFilter}))
	t.Run("content_filter without content", runTest(testCase{content: "", reason: openaiapi.FinishReasonContentFilter}))
	t.Run("no reason", runTest(testCase{content: "ok", reason: ""}))

	t.Run("Stream sets the reason on the chunk that carries it", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, c := range []string{
				`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"partial"}}]}`,
				`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
				`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":3,"total_tokens":23}}`,
				"[DONE]",
			} {
				if _, err := io.WriteString(w, "data: "+c+"\n\n"); err != nil {
					t.Errorf("failed to write chunk: %v", err)
					return
				}
			}
		}))
		t.Cleanup(srv.Close)

		client, err := openai.New(context.Background(), "test-key",
			openai.WithBaseURL(srv.URL+"/v1"), openai.WithModel("gpt-test"))
		gt.NoError(t, err).Required()
		session, err := client.NewSession(context.Background())
		gt.NoError(t, err).Required()

		rec := trace.New()
		ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
		ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		var reasons, texts []string
		for resp := range ch {
			gt.NoError(t, resp.Error)
			if resp.FinishReason != "" {
				reasons = append(reasons, resp.FinishReason)
				gt.A(t, resp.Texts).Length(0)
			}
			texts = append(texts, resp.Texts...)
		}
		gt.Equal(t, []string{"length"}, reasons)
		gt.Equal(t, []string{"partial"}, texts)
		gt.Equal(t, "length", findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response.FinishReason)
	})
}

// TestOpenAIHistoryAfterResponseWithoutText pins the request that follows a
// response without text, which is what gollem.Query sends when it asks the
// model to continue.
func TestOpenAIHistoryAfterResponseWithoutText(t *testing.T) {
	type testCase struct {
		reasoning string
		roles     []string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			var requests []openaiapi.ChatCompletionRequest
			mockClient := &apiClientMock{
				CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
					requests = append(requests, req)
					return openaiapi.ChatCompletionResponse{
						Choices: []openaiapi.ChatCompletionChoice{{
							Message: openaiapi.ChatCompletionMessage{
								Role:             openaiapi.ChatMessageRoleAssistant,
								ReasoningContent: tc.reasoning,
							},
							FinishReason: openaiapi.FinishReasonStop,
						}},
					}, nil
				},
			}
			session, err := openai.NewSessionWithAPIClient(mockClient, gollem.NewSessionConfig(), "gpt-4")
			gt.NoError(t, err).Required()

			resp, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("question")})
			gt.NoError(t, err).Required()
			gt.A(t, resp.Texts).Length(0)
			_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("continue")})
			gt.NoError(t, err).Required()

			gt.A(t, requests).Length(2).Required()
			var roles []string
			for _, m := range requests[1].Messages {
				roles = append(roles, m.Role)
			}
			gt.Equal(t, tc.roles, roles)
		}
	}

	t.Run("reasoning only", runTest(testCase{
		reasoning: "plan",
		roles:     []string{"user", "assistant", "user"},
	}))

	// An assistant message without content is not kept, so the next user
	// message follows the previous one.
	t.Run("no content", runTest(testCase{
		roles: []string{"user", "user"},
	}))
}

func TestOpenAITraceRecordsThoughts(t *testing.T) {
	t.Run("Generate", func(t *testing.T) {
		mockClient := &apiClientMock{
			CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
				return openaiapi.ChatCompletionResponse{
					Choices: []openaiapi.ChatCompletionChoice{{
						Message: openaiapi.ChatCompletionMessage{
							Role:             openaiapi.ChatMessageRoleAssistant,
							Content:          "answer",
							ReasoningContent: "plan",
						},
						FinishReason: openaiapi.FinishReasonStop,
					}},
					Usage: openaiapi.Usage{PromptTokens: 20, CompletionTokens: 3},
				}, nil
			},
		}
		session, err := openai.NewSessionWithAPIClient(mockClient, gollem.NewSessionConfig(), "gpt-4")
		gt.NoError(t, err).Required()

		rec := trace.New()
		ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		gt.Equal(t, []string{"plan"}, resp.Thoughts)

		traced := findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response
		gt.Equal(t, resp.Thoughts, traced.Thoughts)
		gt.Equal(t, []string{"answer"}, traced.Texts)
		gt.A(t, traced.OtherBlockTypes).Length(0)
	})

	t.Run("Stream", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, c := range []string{
				`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"pl"}}]}`,
				`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"reasoning_content":"an"}}]}`,
				`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"answer"}}]}`,
				`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
				"[DONE]",
			} {
				if _, err := io.WriteString(w, "data: "+c+"\n\n"); err != nil {
					t.Errorf("failed to write chunk: %v", err)
					return
				}
			}
		}))
		t.Cleanup(srv.Close)

		client, err := openai.New(context.Background(), "test-key",
			openai.WithBaseURL(srv.URL+"/v1"), openai.WithModel("gpt-test"))
		gt.NoError(t, err).Required()
		session, err := client.NewSession(context.Background())
		gt.NoError(t, err).Required()

		rec := trace.New()
		ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
		ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		var thoughts []string
		for resp := range ch {
			gt.NoError(t, resp.Error)
			thoughts = append(thoughts, resp.Thoughts...)
		}
		gt.Equal(t, []string{"pl", "an"}, thoughts)

		traced := findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response
		gt.Equal(t, []string{"plan"}, traced.Thoughts)
		gt.Equal(t, []string{"answer"}, traced.Texts)
	})
}

func TestOpenAIRefusal(t *testing.T) {
	type testCase struct {
		content  string
		refusal  string
		reason   openaiapi.FinishReason
		expected *gollem.Refusal
	}
	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			mockClient := &apiClientMock{
				CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
					return openaiapi.ChatCompletionResponse{
						Choices: []openaiapi.ChatCompletionChoice{{
							Message: openaiapi.ChatCompletionMessage{
								Content: tc.content,
								Refusal: tc.refusal,
								Role:    openaiapi.ChatMessageRoleAssistant,
							},
							FinishReason: tc.reason,
						}},
						Usage: openaiapi.Usage{PromptTokens: 20, CompletionTokens: 3},
					}, nil
				},
			}
			session, err := openai.NewSessionWithAPIClient(mockClient, gollem.NewSessionConfig(), "gpt-4")
			gt.NoError(t, err).Required()

			rec := trace.New()
			ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
			resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
			gt.NoError(t, err).Required()
			gt.Equal(t, tc.expected, resp.Refusal)

			traced := findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response.Refusal
			if tc.expected == nil {
				gt.Nil(t, traced)
				return
			}
			gt.Equal(t, &trace.Refusal{Reason: tc.expected.Reason, Explanation: tc.expected.Explanation}, traced)
		}
	}

	t.Run("refusal message", runTest(testCase{
		refusal:  "I can't help with that.",
		reason:   openaiapi.FinishReasonStop,
		expected: &gollem.Refusal{Reason: "refusal", Explanation: "I can't help with that."},
	}))
	t.Run("content_filter without a refusal message", runTest(testCase{
		reason:   openaiapi.FinishReasonContentFilter,
		expected: &gollem.Refusal{Reason: "content_filter"},
	}))
	t.Run("content_filter with a refusal message", runTest(testCase{
		refusal:  "I can't help with that.",
		reason:   openaiapi.FinishReasonContentFilter,
		expected: &gollem.Refusal{Reason: "content_filter", Explanation: "I can't help with that."},
	}))
	t.Run("stop without a refusal message", runTest(testCase{
		content: "ok",
		reason:  openaiapi.FinishReasonStop,
	}))

	t.Run("Stream reports the refusal message with the finish reason", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			for _, c := range []string{
				`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","refusal":"I can't "}}]}`,
				`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"refusal":"help with that."}}]}`,
				`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
				`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[],"usage":{"prompt_tokens":20,"completion_tokens":3,"total_tokens":23}}`,
				"[DONE]",
			} {
				if _, err := io.WriteString(w, "data: "+c+"\n\n"); err != nil {
					t.Errorf("failed to write chunk: %v", err)
					return
				}
			}
		}))
		t.Cleanup(srv.Close)

		client, err := openai.New(context.Background(), "test-key",
			openai.WithBaseURL(srv.URL+"/v1"), openai.WithModel("gpt-test"))
		gt.NoError(t, err).Required()
		session, err := client.NewSession(context.Background())
		gt.NoError(t, err).Required()

		rec := trace.New()
		ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
		ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		var refusals []*gollem.Refusal
		for resp := range ch {
			gt.NoError(t, resp.Error)
			if resp.Refusal != nil {
				gt.Equal(t, "stop", resp.FinishReason)
				refusals = append(refusals, resp.Refusal)
			}
		}
		gt.Equal(t, []*gollem.Refusal{{Reason: "refusal", Explanation: "I can't help with that."}}, refusals)
		gt.Equal(t, &trace.Refusal{Reason: "refusal", Explanation: "I can't help with that."},
			findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response.Refusal)
	})
}

// TestOpenAIStreamUsageLive verifies that streaming reports token usage. Before
// the fix the loop broke on the finish reason and never read the trailing usage
// chunk, so streamed responses reported zero tokens.
func TestOpenAIStreamUsageLive(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := openai.New(ctx, apiKey)
	gt.NoError(t, err).Required()
	session, err := client.NewSession(ctx)
	gt.NoError(t, err).Required()

	ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("Say hello in one word")}, gollem.WithMaxTokens(64))
	gt.NoError(t, err).Required()

	var lastInput, lastOutput int
	for resp := range ch {
		gt.NoError(t, resp.Error)
		if resp.InputToken > 0 {
			lastInput = resp.InputToken
		}
		if resp.OutputToken > 0 {
			lastOutput = resp.OutputToken
		}
	}
	t.Logf("stream usage: input=%d output=%d", lastInput, lastOutput)
	gt.Value(t, lastInput > 0).Equal(true)
	gt.Value(t, lastOutput > 0).Equal(true)
}

// TestConvertResponseSchemaToOpenAIIsByteStable pins the response schema JSON to
// be byte-identical between conversions. In strict mode every property name is
// copied into the required array, which is the array most exposed to Go map
// iteration order.
func TestConvertResponseSchemaToOpenAIIsByteStable(t *testing.T) {
	param := &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"zulu":  {Type: gollem.TypeString, Required: true},
			"alpha": {Type: gollem.TypeString, Required: true},
			"mike":  {Type: gollem.TypeString},
			"bravo": {Type: gollem.TypeString},
		},
	}

	runTest := func(strict bool, expectedRequired string) func(t *testing.T) {
		return func(t *testing.T) {
			first, err := openai.ConvertResponseSchemaToOpenAI(param, strict)
			gt.NoError(t, err)
			gt.S(t, string(first.Schema.(json.RawMessage))).Contains(expectedRequired)

			for i := 0; i < 100; i++ {
				actual, err := openai.ConvertResponseSchemaToOpenAI(param, strict)
				gt.NoError(t, err)
				gt.Equal(t,
					string(first.Schema.(json.RawMessage)),
					string(actual.Schema.(json.RawMessage)))
			}
		}
	}

	t.Run("strict mode requires every property", runTest(true, `"required":["alpha","bravo","mike","zulu"]`))
	t.Run("non-strict mode requires the marked properties", runTest(false, `"required":["alpha","zulu"]`))
}

func TestConvertResponseSchemaToOpenAIMap(t *testing.T) {
	param := &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"answer": {Type: gollem.TypeString, Required: true},
			"labels": {
				Type:                 gollem.TypeObject,
				AdditionalProperties: &gollem.Parameter{Type: gollem.TypeString},
			},
		},
	}

	t.Run("non-strict mode sends the map", func(t *testing.T) {
		format, err := openai.ConvertResponseSchemaToOpenAI(param, false)
		gt.NoError(t, err)
		gt.S(t, string(format.Schema.(json.RawMessage))).
			Contains(`"labels":{"additionalProperties":{"type":"string"},"type":"object"}`)
	})

	t.Run("strict mode rejects the map", func(t *testing.T) {
		_, err := openai.ConvertResponseSchemaToOpenAI(param, true)
		gt.True(t, errors.Is(err, gollem.ErrUnsupportedSchema))
		gt.S(t, err.Error()).Contains(`map at "labels" cannot be sent in OpenAI strict mode`)
	})
}

// newBlocksParameter returns an object whose "blocks" array holds a union of
// two object kinds told apart by a single-value "kind" enum.
func newBlocksParameter() *gollem.Parameter {
	return &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"blocks": {
				Type:     gollem.TypeArray,
				Required: true,
				Items: &gollem.Parameter{
					Description: "a block",
					AnyOf: []*gollem.Parameter{
						{
							Type: gollem.TypeObject,
							Properties: map[string]*gollem.Parameter{
								"kind": {Type: gollem.TypeString, Enum: []string{"paragraph"}, Required: true},
								"text": {Type: gollem.TypeString, Required: true},
							},
						},
						{
							Type: gollem.TypeObject,
							Properties: map[string]*gollem.Parameter{
								"kind": {Type: gollem.TypeString, Enum: []string{"callout"}, Required: true},
								"text": {Type: gollem.TypeString, Required: true},
								"tone": {Type: gollem.TypeString, Enum: []string{"info", "warning"}},
							},
						},
					},
				},
			},
		},
	}
}

func TestConvertResponseSchemaToOpenAIAnyOf(t *testing.T) {
	runTest := func(strict bool, expected string) func(t *testing.T) {
		return func(t *testing.T) {
			param := newBlocksParameter()
			for range 20 {
				format, err := openai.ConvertResponseSchemaToOpenAI(param, strict)
				gt.NoError(t, err)
				gt.Equal(t, expected, string(format.Schema.(json.RawMessage)))
			}
		}
	}

	t.Run("strict mode requires every property of each element", runTest(true,
		`{"additionalProperties":false,"properties":{"blocks":{"items":{"anyOf":[`+
			`{"additionalProperties":false,"properties":{"kind":{"enum":["paragraph"],"type":"string"},"text":{"type":"string"}},"required":["kind","text"],"type":"object"},`+
			`{"additionalProperties":false,"properties":{"kind":{"enum":["callout"],"type":"string"},"text":{"type":"string"},"tone":{"enum":["info","warning"],"type":"string"}},"required":["kind","text","tone"],"type":"object"}`+
			`],"description":"a block"},"type":"array"}},"required":["blocks"],"type":"object"}`))

	t.Run("non-strict mode requires the marked properties", runTest(false,
		`{"additionalProperties":false,"properties":{"blocks":{"items":{"anyOf":[`+
			`{"additionalProperties":false,"properties":{"kind":{"enum":["paragraph"],"type":"string"},"text":{"type":"string"}},"required":["kind","text"],"type":"object"},`+
			`{"additionalProperties":false,"properties":{"kind":{"enum":["callout"],"type":"string"},"text":{"type":"string"},"tone":{"enum":["info","warning"],"type":"string"}},"required":["kind","text"],"type":"object"}`+
			`],"description":"a block"},"type":"array"}},"required":["blocks"],"type":"object"}`))

	rootUnion := func() *gollem.Parameter {
		return newBlocksParameter().Properties["blocks"].Items
	}

	t.Run("strict mode rejects a union at the root", func(t *testing.T) {
		_, err := openai.ConvertResponseSchemaToOpenAI(rootUnion(), true)
		gt.True(t, errors.Is(err, gollem.ErrUnsupportedSchema))
		gt.S(t, err.Error()).Contains("anyOf at the root cannot be sent in OpenAI strict mode")
	})

	t.Run("non-strict mode sends a union at the root", func(t *testing.T) {
		format, err := openai.ConvertResponseSchemaToOpenAI(rootUnion(), false)
		gt.NoError(t, err)
		gt.S(t, string(format.Schema.(json.RawMessage))).HasPrefix(`{"anyOf":[`)
	})
}

// blocksPrompt asks for a reply that uses both kinds of newBlocksParameter.
const blocksPrompt = "Write a three-block note about tea: a paragraph, a warning callout, then a paragraph. Reply in JSON."

// assertBlocksReply decodes a reply to newBlocksParameter and checks that each
// block matches exactly one kind and carries only that kind's fields.
func assertBlocksReply(t *testing.T, text string) {
	t.Helper()
	var reply struct {
		Blocks []map[string]any `json:"blocks"`
	}
	gt.NoError(t, json.Unmarshal([]byte(text), &reply)).Required()
	gt.A(t, reply.Blocks).Longer(0)

	kinds := newBlocksParameter().Properties["blocks"].Items.AnyOf
	seen := map[string]bool{}
	for i, block := range reply.Blocks {
		matched := 0
		for _, kind := range kinds {
			if kind.ValidateValue("block", block) != nil {
				continue
			}
			matched++
			for key := range block {
				if _, ok := kind.Properties[key]; !ok {
					t.Errorf("block %d has field %q that its kind does not define: %v", i, key, block)
				}
			}
		}
		if matched != 1 {
			t.Errorf("block %d matches %d kinds, want 1: %v", i, matched, block)
		}
		kind, _ := block["kind"].(string)
		seen[kind] = true
	}
	if !seen["paragraph"] || !seen["callout"] {
		t.Errorf("reply must hold both a paragraph and a callout: %v", reply.Blocks)
	}
}

func TestAnyOfResponseSchemaLive(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}

	runTest := func(strict bool) func(t *testing.T) {
		return func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*testTimeout)
			defer cancel()

			client, err := openai.New(ctx, apiKey)
			gt.NoError(t, err).Required()
			session, err := client.NewSession(ctx,
				gollem.WithSessionContentType(gollem.ContentTypeJSON),
				gollem.WithSessionResponseSchema(newBlocksParameter()))
			gt.NoError(t, err).Required()
			if strict {
				openai.EnableStrictMode(session)
			}

			resp, err := session.Generate(ctx, []gollem.Input{gollem.Text(blocksPrompt)})
			gt.NoError(t, err).Required()
			gt.A(t, resp.Texts).Length(1).Required()
			t.Logf("reply: %s", resp.Texts[0])
			assertBlocksReply(t, resp.Texts[0])
		}
	}

	t.Run("strict", runTest(true))
	t.Run("non-strict", runTest(false))
}

var _ gollem.ModelNamer = (*openai.Client)(nil)

// TestClientModel verifies that the client reports the model name it was
// configured with, without consulting the API.
func TestClientModel(t *testing.T) {
	type testCase struct {
		options  []openai.Option
		expected string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			client, err := openai.New(context.Background(), "test-key", tc.options...)
			gt.NoError(t, err).Required()
			gt.Equal(t, tc.expected, client.Model())
		}
	}

	t.Run("configured model", runTest(testCase{
		options:  []openai.Option{openai.WithModel("gpt-5-mini")},
		expected: "gpt-5-mini",
	}))

	t.Run("default model when no option is given", runTest(testCase{
		expected: openai.DefaultModel,
	}))

	t.Run("last option wins", runTest(testCase{
		options: []openai.Option{
			openai.WithModel("gpt-5-mini"),
			openai.WithModel("gpt-5-nano"),
		},
		expected: "gpt-5-nano",
	}))

	t.Run("empty model is reported as configured", runTest(testCase{
		options:  []openai.Option{openai.WithModel("")},
		expected: "",
	}))
}

// lookupTool is a minimal tool for inspecting how tool settings are sent.
type lookupTool struct{}

func (t *lookupTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "lookup",
		Description: "Look up a value",
		Parameters:  map[string]*gollem.Parameter{"key": {Type: gollem.TypeString}},
	}
}

func (t *lookupTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return map[string]any{"value": "ok"}, nil
}

func TestToolCallsDisabled(t *testing.T) {
	type testCase struct {
		tools    []gollem.Tool
		opts     []gollem.GenerateOption
		expected any
	}

	errStreamNotServed := errors.New("stream is not served by this mock")

	runTest := func(stream bool, tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			var sent *openaiapi.ChatCompletionRequest
			mockClient := &apiClientMock{
				CreateChatCompletionFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (openaiapi.ChatCompletionResponse, error) {
					sent = &req
					return openaiapi.ChatCompletionResponse{
						Choices: []openaiapi.ChatCompletionChoice{{
							Message: openaiapi.ChatCompletionMessage{Content: "ok", Role: openaiapi.ChatMessageRoleAssistant},
						}},
					}, nil
				},
				// The request is inspected before the stream would be read, so the
				// mock records it and fails instead of building a stream.
				CreateChatCompletionStreamFunc: func(ctx context.Context, req openaiapi.ChatCompletionRequest) (*openaiapi.ChatCompletionStream, error) {
					sent = &req
					return nil, errStreamNotServed
				},
			}
			cfg := gollem.NewSessionConfig(gollem.WithSessionTools(tc.tools...))
			session, err := openai.NewSessionWithAPIClient(mockClient, cfg, "gpt-5-nano")
			gt.NoError(t, err)

			input := []gollem.Input{gollem.Text("question")}
			if stream {
				_, err = session.Stream(context.Background(), input, tc.opts...)
				gt.True(t, errors.Is(err, errStreamNotServed))
			} else {
				_, err = session.Generate(context.Background(), input, tc.opts...)
				gt.NoError(t, err)
			}

			var expectedTools []openaiapi.Tool
			for _, tool := range tc.tools {
				expectedTools = append(expectedTools, openai.ConvertTool(tool))
			}
			gt.NotNil(t, sent)
			gt.A(t, sent.Tools).Length(len(expectedTools)).Required()
			for i := range expectedTools {
				gt.Equal(t, expectedTools[i], sent.Tools[i])
			}
			gt.Equal(t, tc.expected, sent.ToolChoice)
		}
	}

	for _, stream := range []bool{false, true} {
		name := "Generate"
		if stream {
			name = "Stream"
		}
		t.Run(name, func(t *testing.T) {
			t.Run("sends tool_choice none with the tools unchanged", runTest(stream, testCase{
				tools:    []gollem.Tool{&lookupTool{}},
				opts:     []gollem.GenerateOption{gollem.WithToolCallsDisabled()},
				expected: "none",
			}))
			t.Run("sends no tool_choice without the option", runTest(stream, testCase{
				tools: []gollem.Tool{&lookupTool{}},
			}))
			t.Run("sends no tool_choice when the session has no tools", runTest(stream, testCase{
				opts: []gollem.GenerateOption{gollem.WithToolCallsDisabled()},
			}))
		})
	}
}

// chatReasoningServer is a Chat Completions endpoint that records the messages
// of every request and answers the first request with reasoning and a tool call,
// and later requests with a plain text.
type chatReasoningServer struct {
	srv      *httptest.Server
	requests [][]map[string]any
}

func newChatReasoningServer(t *testing.T) *chatReasoningServer {
	t.Helper()
	cs := &chatReasoningServer{}
	cs.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Stream   bool             `json:"stream"`
			Messages []map[string]any `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		cs.requests = append(cs.requests, body.Messages)
		first := len(cs.requests) == 1

		if body.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			var chunks []string
			if first {
				chunks = []string{
					`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think "}}]}`,
					`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"reasoning_content":"hard"}}]}`,
					`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"let me check"}}]}`,
					`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"alpha\"}"}}]}}]}`,
					`{"id":"c1","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
				}
			} else {
				chunks = []string{
					`{"id":"c2","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":"done"},"finish_reason":"stop"}]}`,
				}
			}
			for _, c := range chunks {
				_, _ = io.WriteString(w, "data: "+c+"\n\n")
			}
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if first {
			_, _ = io.WriteString(w, `{"id":"c1","object":"chat.completion","model":"m",
				"choices":[{"index":0,"message":{"role":"assistant","content":"let me check","reasoning_content":"think hard",
				"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"key\":\"alpha\"}"}}]},
				"finish_reason":"tool_calls"}],
				"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`)
			return
		}
		_, _ = io.WriteString(w, `{"id":"c2","object":"chat.completion","model":"m",
			"choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],
			"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}`)
	}))
	t.Cleanup(cs.srv.Close)
	return cs
}

// OpenAI-compatible servers that return reasoning_content can reject a request
// with tools that does not send it back, so the session keeps it in its history,
// together with the text that came with the tool calls.
func TestChatReasoningIsSentBack(t *testing.T) {
	runTest := func(stream bool) func(t *testing.T) {
		return func(t *testing.T) {
			cs := newChatReasoningServer(t)
			client, err := openai.New(context.Background(), "test-key",
				openai.WithBaseURL(cs.srv.URL+"/v1"), openai.WithModel("gpt-test"))
			gt.NoError(t, err).Required()
			session, err := client.NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
			gt.NoError(t, err).Required()

			send := func(input []gollem.Input) {
				if !stream {
					_, err := session.Generate(context.Background(), input)
					gt.NoError(t, err).Required()
					return
				}
				ch, err := session.Stream(context.Background(), input)
				gt.NoError(t, err).Required()
				for resp := range ch {
					gt.NoError(t, resp.Error).Required()
				}
			}

			send([]gollem.Input{gollem.Text("what is alpha?")})

			h, err := session.History()
			gt.NoError(t, err).Required()
			gt.A(t, h.Messages).Length(2).Required()
			thinking := h.Messages[1].Contents[0]
			gt.Equal(t, gollem.MessageContentTypeThinking, thinking.Type)
			gt.V(t, thinking.Provider).NotNil().Required()
			gt.Equal(t, gollem.Issuer{Provider: gollem.LLMTypeOpenAI, Model: "gpt-test"}, thinking.Provider.Issuer)

			send([]gollem.Input{gollem.FunctionResponse{ID: "call_1", Name: "lookup", Data: map[string]any{"value": "ok"}}})

			gt.A(t, cs.requests).Length(2).Required()
			assistant := cs.requests[1][1]
			gt.Equal(t, any("assistant"), assistant["role"])
			gt.Equal(t, any("think hard"), assistant["reasoning_content"])
			gt.Equal(t, any("let me check"), assistant["content"])
			gt.A(t, assistant["tool_calls"].([]any)).Length(1)
		}
	}

	t.Run("Generate", runTest(false))
	t.Run("Stream", runTest(true))
}

func TestChatReasoningOnlyMessageIsSentBack(t *testing.T) {
	thinking, err := gollem.NewThinkingContent("only reasoning")
	gt.NoError(t, err).Required()
	thinking.Provider = &gollem.ProviderData{Issuer: gollem.Issuer{Provider: gollem.LLMTypeOpenAI, Model: "gpt-test"}}
	user, err := gollem.NewTextContent("hi")
	gt.NoError(t, err).Required()

	messages, err := openai.ToMessages(&gollem.History{
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{user}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{thinking}},
		},
	}, gollem.Issuer{Provider: gollem.LLMTypeOpenAI, Model: "gpt-test"})
	gt.NoError(t, err).Required()
	gt.A(t, messages).Length(2).Required()
	gt.Equal(t, "assistant", messages[1].Role)
	gt.Equal(t, "only reasoning", messages[1].ReasoningContent)
}
