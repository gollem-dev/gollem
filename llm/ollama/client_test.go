package ollama_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/ollama"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
	"github.com/m-mizutani/gt"
)

// recordedRequest is a request received by fakeServer.
type recordedRequest struct {
	Path   string
	Header http.Header
	Body   map[string]any
}

// fakeServer is an httptest server that records each request and answers with
// the handler given to newFakeServer.
type fakeServer struct {
	URL      string
	mu       sync.Mutex
	requests []recordedRequest
}

func newFakeServer(t *testing.T, handler func(w http.ResponseWriter, req recordedRequest)) *fakeServer {
	t.Helper()
	fs := &fakeServer{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		req := recordedRequest{Path: r.URL.Path, Header: r.Header.Clone(), Body: body}
		fs.mu.Lock()
		fs.requests = append(fs.requests, req)
		fs.mu.Unlock()
		handler(w, req)
	}))
	t.Cleanup(srv.Close)
	fs.URL = srv.URL
	return fs
}

func (fs *fakeServer) Requests() []recordedRequest {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	return append([]recordedRequest(nil), fs.requests...)
}

func (fs *fakeServer) Last(t *testing.T) recordedRequest {
	t.Helper()
	reqs := fs.Requests()
	gt.A(t, reqs).Longer(0).Required()
	return reqs[len(reqs)-1]
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// replyText answers every chat request with an assistant message.
func replyText(content, thinking string) func(w http.ResponseWriter, req recordedRequest) {
	return func(w http.ResponseWriter, req recordedRequest) {
		writeJSON(w, http.StatusOK, map[string]any{
			"model":             "test-model",
			"message":           map[string]any{"role": "assistant", "content": content, "thinking": thinking},
			"done":              true,
			"prompt_eval_count": 10,
			"eval_count":        3,
		})
	}
}

func newTestClient(t *testing.T, fs *fakeServer, opts ...ollama.Option) *ollama.Client {
	t.Helper()
	client, err := ollama.New(context.Background(), "test-model", append([]ollama.Option{ollama.WithBaseURL(fs.URL)}, opts...)...)
	gt.NoError(t, err).Required()
	return client
}

func newTestSession(t *testing.T, client *ollama.Client, opts ...gollem.SessionOption) gollem.Session {
	t.Helper()
	session, err := client.NewSession(context.Background(), opts...)
	gt.NoError(t, err).Required()
	return session
}

func messagesOf(t *testing.T, req recordedRequest) []map[string]any {
	t.Helper()
	raw, ok := req.Body["messages"].([]any)
	gt.True(t, ok)
	result := make([]map[string]any, len(raw))
	for i, m := range raw {
		result[i] = m.(map[string]any)
	}
	return result
}

func optionsOf(req recordedRequest) map[string]any {
	options, _ := req.Body["options"].(map[string]any)
	return options
}

type weatherTool struct{}

func (weatherTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "get_weather",
		Description: "Get the weather of a city",
		Parameters: map[string]*gollem.Parameter{
			"city": {Type: gollem.TypeString, Description: "City name", Required: true},
			"days": {Type: gollem.TypeInteger, Description: "Number of days"},
		},
	}
}

func (weatherTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return map[string]any{"weather": "sunny"}, nil
}

func TestNew(t *testing.T) {
	ctx := context.Background()

	t.Run("empty model", func(t *testing.T) {
		_, err := ollama.New(ctx, "")
		gt.Error(t, err).Is(gollem.ErrInvalidParameter)
	})

	runInvalidBaseURL := func(baseURL string) func(t *testing.T) {
		return func(t *testing.T) {
			_, err := ollama.New(ctx, "m", ollama.WithBaseURL(baseURL))
			gt.Error(t, err).Is(gollem.ErrInvalidParameter)
		}
	}
	t.Run("base URL without scheme", runInvalidBaseURL("localhost:11434"))
	t.Run("base URL with unsupported scheme", runInvalidBaseURL("ftp://example.com"))
	t.Run("base URL without host", runInvalidBaseURL("http://"))
	t.Run("unparsable base URL", runInvalidBaseURL("http://[::1"))

	t.Run("nil HTTP client", func(t *testing.T) {
		_, err := ollama.New(ctx, "m", ollama.WithHTTPClient(nil))
		gt.Error(t, err).Is(gollem.ErrInvalidParameter)
	})

	t.Run("model name", func(t *testing.T) {
		client, err := ollama.New(ctx, "qwen3:8b")
		gt.NoError(t, err).Required()
		gt.Equal(t, client.Model(), "qwen3:8b")
	})

	t.Run("base URL with path prefix", func(t *testing.T) {
		fs := newFakeServer(t, replyText("ok", ""))
		client, err := ollama.New(ctx, "m", ollama.WithBaseURL(fs.URL+"/proxy/"))
		gt.NoError(t, err).Required()
		_, err = newTestSession(t, client).Generate(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		gt.Equal(t, fs.Last(t).Path, "/proxy/api/chat")
	})
}

func TestGenerate(t *testing.T) {
	ctx := context.Background()

	t.Run("text input and response", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeJSON(w, http.StatusOK, map[string]any{
				"model":                    "test-model",
				"message":                  map[string]any{"role": "assistant", "content": "hi", "thinking": "t"},
				"done":                     true,
				"prompt_eval_count":        10,
				"prompt_eval_cached_count": 4,
				"eval_count":               3,
			})
		})
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionSystemPrompt("be brief"))

		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("hello")})
		gt.NoError(t, err).Required()
		gt.Equal(t, resp.Texts, []string{"hi"})
		gt.Equal(t, resp.Thoughts, []string{"t"})
		gt.Equal(t, resp.InputToken, 10)
		gt.Equal(t, resp.OutputToken, 3)
		gt.Equal(t, resp.CacheReadInputToken, 4)

		req := fs.Last(t)
		gt.Equal(t, req.Path, "/api/chat")
		gt.Equal(t, req.Body["model"], any("test-model"))
		gt.Equal(t, req.Body["stream"], any(false))
		gt.Equal(t, req.Body["truncate"], any(false))
		gt.Equal(t, req.Body["shift"], any(false))
		msgs := messagesOf(t, req)
		gt.A(t, msgs).Length(2).Required()
		gt.Equal(t, msgs[0]["role"], any("system"))
		gt.Equal(t, msgs[0]["content"], any("be brief"))
		gt.Equal(t, msgs[1]["role"], any("user"))
		gt.Equal(t, msgs[1]["content"], any("hello"))
	})

	t.Run("done_reason becomes FinishReason", func(t *testing.T) {
		type testCase struct {
			content    string
			doneReason string
		}
		runTest := func(tc testCase) func(t *testing.T) {
			return func(t *testing.T) {
				fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
					writeJSON(w, http.StatusOK, map[string]any{
						"model":       "test-model",
						"message":     map[string]any{"role": "assistant", "content": tc.content},
						"done":        true,
						"done_reason": tc.doneReason,
					})
				})
				rec := trace.New()
				tctx := trace.WithHandler(rec.StartAgentExecute(ctx), rec)
				resp, err := newTestSession(t, newTestClient(t, fs)).Generate(tctx, []gollem.Input{gollem.Text("hello")})
				gt.NoError(t, err).Required()
				gt.Equal(t, resp.FinishReason, tc.doneReason)

				spans := llmCallSpans(rec.Trace().RootSpan)
				gt.A(t, spans).Length(1).Required()
				gt.Equal(t, spans[0].LLMCall.Response.FinishReason, resp.FinishReason)
			}
		}
		t.Run("with content", runTest(testCase{content: "cut", doneReason: "length"}))
		t.Run("without content", runTest(testCase{content: "", doneReason: "load"}))
	})

	t.Run("thinking is recorded in the trace", func(t *testing.T) {
		fs := newFakeServer(t, replyText("answer", "plan"))
		rec := trace.New()
		tctx := trace.WithHandler(rec.StartAgentExecute(ctx), rec)
		resp, err := newTestSession(t, newTestClient(t, fs)).Generate(tctx, []gollem.Input{gollem.Text("hello")})
		gt.NoError(t, err).Required()
		gt.Equal(t, resp.Thoughts, []string{"plan"})

		spans := llmCallSpans(rec.Trace().RootSpan)
		gt.A(t, spans).Length(1).Required()
		gt.Equal(t, spans[0].LLMCall.Response.Thoughts, resp.Thoughts)
		gt.Equal(t, spans[0].LLMCall.Response.Texts, []string{"answer"})
		gt.A(t, spans[0].LLMCall.Response.OtherBlockTypes).Length(0)
	})

	// The request that follows a response without text is what gollem.Query
	// sends when it asks the model to continue.
	t.Run("history after a response without text", func(t *testing.T) {
		type testCase struct {
			thinking string
			roles    []any
		}
		runTest := func(tc testCase) func(t *testing.T) {
			return func(t *testing.T) {
				fs := newFakeServer(t, replyText("", tc.thinking))
				session := newTestSession(t, newTestClient(t, fs))

				resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("question")})
				gt.NoError(t, err).Required()
				gt.A(t, resp.Texts).Length(0)
				_, err = session.Generate(ctx, []gollem.Input{gollem.Text("continue")})
				gt.NoError(t, err).Required()

				var roles []any
				for _, m := range messagesOf(t, fs.Last(t)) {
					roles = append(roles, m["role"])
				}
				gt.Equal(t, tc.roles, roles)
			}
		}

		t.Run("thinking only", runTest(testCase{
			thinking: "plan",
			roles:    []any{"user", "assistant", "user"},
		}))
		// An assistant message without content is not kept, so the next user
		// message follows the previous one.
		t.Run("no content", runTest(testCase{
			roles: []any{"user", "user"},
		}))
	})

	t.Run("cache count absent", func(t *testing.T) {
		fs := newFakeServer(t, replyText("hi", ""))
		resp, err := newTestSession(t, newTestClient(t, fs)).Generate(ctx, []gollem.Input{gollem.Text("hello")})
		gt.NoError(t, err).Required()
		gt.Equal(t, resp.CacheReadInputToken, 0)
	})

	t.Run("history across calls", func(t *testing.T) {
		fs := newFakeServer(t, replyText("answer", "reasoning"))
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionSystemPrompt("sys"))

		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("q1")})
		gt.NoError(t, err).Required()
		_, err = session.Generate(ctx, []gollem.Input{gollem.Text("q2")})
		gt.NoError(t, err).Required()

		msgs := messagesOf(t, fs.Last(t))
		gt.A(t, msgs).Length(4).Required()
		gt.Equal(t, msgs[0]["role"], any("system"))
		gt.Equal(t, msgs[1]["content"], any("q1"))
		gt.Equal(t, msgs[2]["role"], any("assistant"))
		gt.Equal(t, msgs[2]["content"], any("answer"))
		gt.Equal(t, msgs[2]["thinking"], any("reasoning"))
		gt.Equal(t, msgs[3]["content"], any("q2"))

		history, err := session.History()
		gt.NoError(t, err).Required()
		gt.Equal(t, history.LLType, gollem.LLMTypeOllama)
		gt.A(t, history.Messages).Length(4)
		for _, m := range history.Messages {
			gt.NotEqual(t, m.Role, gollem.RoleSystem)
		}
	})

	type systemPromptCase struct {
		client   string
		session  string
		expected string
	}
	runSystemPrompt := func(tc systemPromptCase) func(t *testing.T) {
		return func(t *testing.T) {
			fs := newFakeServer(t, replyText("ok", ""))
			var opts []ollama.Option
			if tc.client != "" {
				opts = append(opts, ollama.WithSystemPrompt(tc.client))
			}
			var sessionOpts []gollem.SessionOption
			if tc.session != "" {
				sessionOpts = append(sessionOpts, gollem.WithSessionSystemPrompt(tc.session))
			}
			session := newTestSession(t, newTestClient(t, fs, opts...), sessionOpts...)
			_, err := session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
			gt.NoError(t, err).Required()

			msgs := messagesOf(t, fs.Last(t))
			if tc.expected == "" {
				gt.A(t, msgs).Length(1)
				gt.Equal(t, msgs[0]["role"], any("user"))
				return
			}
			gt.Equal(t, msgs[0]["role"], any("system"))
			gt.Equal(t, msgs[0]["content"], any(tc.expected))
		}
	}
	t.Run("client system prompt", runSystemPrompt(systemPromptCase{client: "client", expected: "client"}))
	t.Run("session system prompt wins", runSystemPrompt(systemPromptCase{client: "client", session: "session", expected: "session"}))
	t.Run("no system prompt", runSystemPrompt(systemPromptCase{}))

	t.Run("generation options", func(t *testing.T) {
		fs := newFakeServer(t, replyText("ok", ""))
		client := newTestClient(t, fs,
			ollama.WithTemperature(0),
			ollama.WithTopP(0.9),
			ollama.WithTopK(40),
			ollama.WithMaxTokens(128),
			ollama.WithNumCtx(8192),
		)
		session := newTestSession(t, client)

		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		gt.Equal(t, optionsOf(fs.Last(t)), map[string]any{
			"temperature": float64(0),
			"top_p":       0.9,
			"top_k":       float64(40),
			"num_predict": float64(128),
			"num_ctx":     float64(8192),
		})

		_, err = session.Generate(ctx, []gollem.Input{gollem.Text("hi")},
			gollem.WithMaxTokens(64))
		gt.NoError(t, err).Required()
		options := optionsOf(fs.Last(t))
		gt.Equal(t, options["temperature"], any(float64(0)))
		gt.Equal(t, options["top_p"], any(0.9))
		gt.Equal(t, options["num_predict"], any(float64(64)))

		_, err = session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		gt.Equal(t, optionsOf(fs.Last(t))["num_predict"], any(float64(128)))
	})

	t.Run("no generation options", func(t *testing.T) {
		fs := newFakeServer(t, replyText("ok", ""))
		_, err := newTestSession(t, newTestClient(t, fs)).Generate(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		req := fs.Last(t)
		_, hasOptions := req.Body["options"]
		_, hasThink := req.Body["think"]
		_, hasKeepAlive := req.Body["keep_alive"]
		_, hasFormat := req.Body["format"]
		_, hasTools := req.Body["tools"]
		gt.False(t, hasOptions)
		gt.False(t, hasThink)
		gt.False(t, hasKeepAlive)
		gt.False(t, hasFormat)
		gt.False(t, hasTools)
	})

	type fieldCase struct {
		opts     []ollama.Option
		field    string
		expected any
	}
	runField := func(tc fieldCase) func(t *testing.T) {
		return func(t *testing.T) {
			fs := newFakeServer(t, replyText("ok", ""))
			_, err := newTestSession(t, newTestClient(t, fs, tc.opts...)).Generate(ctx, []gollem.Input{gollem.Text("hi")})
			gt.NoError(t, err).Required()
			gt.Equal(t, fs.Last(t).Body[tc.field], tc.expected)
		}
	}
	t.Run("think true", runField(fieldCase{opts: []ollama.Option{ollama.WithThink(true)}, field: "think", expected: true}))
	t.Run("think false", runField(fieldCase{opts: []ollama.Option{ollama.WithThink(false)}, field: "think", expected: false}))
	t.Run("think level", runField(fieldCase{opts: []ollama.Option{ollama.WithThinkLevel("high")}, field: "think", expected: "high"}))
	t.Run("later think option wins", runField(fieldCase{opts: []ollama.Option{ollama.WithThinkLevel("high"), ollama.WithThink(false)}, field: "think", expected: false}))
	t.Run("keep alive", runField(fieldCase{opts: []ollama.Option{ollama.WithKeepAlive(10 * time.Minute)}, field: "keep_alive", expected: "10m0s"}))
	t.Run("keep alive forever", runField(fieldCase{opts: []ollama.Option{ollama.WithKeepAlive(-1)}, field: "keep_alive", expected: float64(-1)}))

	t.Run("text and image input", func(t *testing.T) {
		fs := newFakeServer(t, replyText("ok", ""))
		img, err := gollem.NewImage(pngData)
		gt.NoError(t, err).Required()

		_, err = newTestSession(t, newTestClient(t, fs)).Generate(ctx, []gollem.Input{
			gollem.Text("first"), img, gollem.Text("second"),
		})
		gt.NoError(t, err).Required()

		msgs := messagesOf(t, fs.Last(t))
		gt.A(t, msgs).Length(1).Required()
		gt.Equal(t, msgs[0]["content"], any("first\nsecond"))
		gt.Equal(t, msgs[0]["images"], any([]any{img.Base64()}))
	})

	t.Run("PDF input is rejected", func(t *testing.T) {
		fs := newFakeServer(t, replyText("ok", ""))
		pdf, err := gollem.NewPDF([]byte("%PDF-1.4\n%%EOF"))
		gt.NoError(t, err).Required()

		_, err = newTestSession(t, newTestClient(t, fs)).Generate(ctx, []gollem.Input{pdf})
		gt.Error(t, err).Is(gollem.ErrInvalidParameter)
		gt.A(t, fs.Requests()).Length(0)
	})

	t.Run("tool definitions", func(t *testing.T) {
		fs := newFakeServer(t, replyText("ok", ""))
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionTools(weatherTool{}))
		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("weather?")})
		gt.NoError(t, err).Required()

		tools, ok := fs.Last(t).Body["tools"].([]any)
		gt.True(t, ok)
		gt.A(t, tools).Length(1).Required()
		tl := tools[0].(map[string]any)
		gt.Equal(t, tl["type"], any("function"))
		fn := tl["function"].(map[string]any)
		gt.Equal(t, fn["name"], any("get_weather"))
		gt.Equal(t, fn["description"], any("Get the weather of a city"))
		params := fn["parameters"].(map[string]any)
		gt.Equal(t, params["type"], any("object"))
		gt.Equal(t, params["required"], any([]any{"city"}))
	})

	t.Run("tool call with ID", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeJSON(w, http.StatusOK, map[string]any{
				"message": map[string]any{
					"role": "assistant",
					"tool_calls": []any{map[string]any{
						"id": "call_abc12345",
						"function": map[string]any{
							"index":     0,
							"name":      "get_weather",
							"arguments": json.RawMessage(`{"city":"Tokyo","days":3,"big":9007199254740993}`),
						},
					}},
				},
				"done": true,
			})
		})
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionTools(weatherTool{}))
		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("weather?")})
		gt.NoError(t, err).Required()

		gt.A(t, resp.FunctionCalls).Length(1).Required()
		call := resp.FunctionCalls[0]
		gt.Equal(t, call.ID, "call_abc12345")
		gt.Equal(t, call.Name, "get_weather")
		gt.Equal(t, call.Arguments["city"], any("Tokyo"))
		gt.Equal(t, call.Arguments["days"], any(float64(3)))
		gt.Equal(t, call.Arguments["big"], any(json.Number("9007199254740993")))
	})

	t.Run("tool calls without ID", func(t *testing.T) {
		calls := 0
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			calls++
			if calls > 1 {
				replyText("done", "")(w, req)
				return
			}
			writeJSON(w, http.StatusOK, map[string]any{
				"message": map[string]any{
					"role": "assistant",
					"tool_calls": []any{
						map[string]any{"function": map[string]any{"name": "get_weather", "arguments": map[string]any{"city": "Tokyo"}}},
						map[string]any{"function": map[string]any{"name": "get_weather", "arguments": map[string]any{"city": "Osaka"}}},
					},
				},
				"done": true,
			})
		})
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionTools(weatherTool{}))
		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("weather?")})
		gt.NoError(t, err).Required()
		gt.A(t, resp.FunctionCalls).Length(2).Required()
		id0, id1 := resp.FunctionCalls[0].ID, resp.FunctionCalls[1].ID
		gt.True(t, strings.HasPrefix(id0, "call_"))
		gt.True(t, strings.HasPrefix(id1, "call_"))
		gt.NotEqual(t, id0, id1)

		_, err = session.Generate(ctx, []gollem.Input{
			gollem.FunctionResponse{ID: id0, Name: "get_weather", Data: map[string]any{"weather": "sunny"}},
			gollem.FunctionResponse{ID: id1, Name: "get_weather", Error: errors.New("timeout")},
		})
		gt.NoError(t, err).Required()

		msgs := messagesOf(t, fs.Last(t))
		gt.A(t, msgs).Length(4).Required()
		assistantCalls := msgs[1]["tool_calls"].([]any)
		gt.Equal(t, assistantCalls[0].(map[string]any)["id"], any(id0))
		gt.Equal(t, assistantCalls[1].(map[string]any)["id"], any(id1))

		gt.Equal(t, msgs[2], map[string]any{
			"role":         "tool",
			"content":      `{"weather":"sunny"}`,
			"tool_name":    "get_weather",
			"tool_call_id": id0,
		})
		gt.Equal(t, msgs[3]["content"], any("Error message: timeout"))
		gt.Equal(t, msgs[3]["tool_call_id"], any(id1))
	})

	t.Run("tool call IDs differ across turns", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeJSON(w, http.StatusOK, map[string]any{
				"message": map[string]any{
					"role":       "assistant",
					"tool_calls": []any{map[string]any{"function": map[string]any{"name": "get_weather", "arguments": map[string]any{"city": "Tokyo"}}}},
				},
				"done": true,
			})
		})
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionTools(weatherTool{}))
		first, err := session.Generate(ctx, []gollem.Input{gollem.Text("weather?")})
		gt.NoError(t, err).Required()
		second, err := session.Generate(ctx, []gollem.Input{
			gollem.FunctionResponse{ID: first.FunctionCalls[0].ID, Name: "get_weather", Data: map[string]any{"weather": "sunny"}},
		})
		gt.NoError(t, err).Required()
		gt.NotEqual(t, first.FunctionCalls[0].ID, second.FunctionCalls[0].ID)
	})

	t.Run("tool call arguments that are not an object", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeJSON(w, http.StatusOK, map[string]any{
				"message": map[string]any{
					"role":       "assistant",
					"tool_calls": []any{map[string]any{"id": "c1", "function": map[string]any{"name": "get_weather", "arguments": "x"}}},
				},
				"done": true,
			})
		})
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionTools(weatherTool{}))
		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("weather?")})
		gt.Error(t, err)

		// The call failed, so the input is not added to the history.
		history, err := session.History()
		gt.NoError(t, err).Required()
		gt.A(t, history.Messages).Length(0)
	})

	t.Run("JSON content type without schema", func(t *testing.T) {
		fs := newFakeServer(t, replyText(`{"a":1}`, ""))
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionContentType(gollem.ContentTypeJSON))
		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("json")})
		gt.NoError(t, err).Required()
		gt.Equal(t, fs.Last(t).Body["format"], any("json"))
	})

	userSchema := &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"name": {Type: gollem.TypeString, Required: true},
		},
	}
	callSchema := &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"score": {Type: gollem.TypeNumber, Required: true},
		},
	}

	t.Run("session and per-call response schema", func(t *testing.T) {
		fs := newFakeServer(t, replyText(`{"name":"a"}`, ""))
		session := newTestSession(t, newTestClient(t, fs),
			gollem.WithSessionContentType(gollem.ContentTypeJSON),
			gollem.WithSessionResponseSchema(userSchema))

		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("json")})
		gt.NoError(t, err).Required()
		format := fs.Last(t).Body["format"].(map[string]any)
		gt.Equal(t, format["required"], any([]any{"name"}))

		_, err = session.Generate(ctx, []gollem.Input{gollem.Text("json")}, gollem.WithGenerateResponseSchema(callSchema))
		gt.NoError(t, err).Required()
		format = fs.Last(t).Body["format"].(map[string]any)
		gt.Equal(t, format["required"], any([]any{"score"}))
	})

	t.Run("invalid response schema", func(t *testing.T) {
		fs := newFakeServer(t, replyText("ok", ""))
		session := newTestSession(t, newTestClient(t, fs))
		invalid := &gollem.Parameter{Type: "unknown"}
		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("json")}, gollem.WithGenerateResponseSchema(invalid))
		gt.Error(t, err)
		gt.A(t, fs.Requests()).Length(0)
	})

	t.Run("tool calls disabled", func(t *testing.T) {
		fs := newFakeServer(t, replyText("ok", ""))
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionTools(weatherTool{}))

		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("hi")}, gollem.WithToolCallsDisabled())
		gt.NoError(t, err).Required()
		_, hasTools := fs.Last(t).Body["tools"]
		gt.False(t, hasTools)

		_, err = session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		_, hasTools = fs.Last(t).Body["tools"]
		gt.True(t, hasTools)
	})

	t.Run("tool calls disabled without tools", func(t *testing.T) {
		fs := newFakeServer(t, replyText("ok", ""))
		_, err := newTestSession(t, newTestClient(t, fs)).Generate(ctx, []gollem.Input{gollem.Text("hi")}, gollem.WithToolCallsDisabled())
		gt.NoError(t, err).Required()
		_, hasTools := fs.Last(t).Body["tools"]
		gt.False(t, hasTools)
	})

	t.Run("empty response", func(t *testing.T) {
		fs := newFakeServer(t, replyText("", ""))
		session := newTestSession(t, newTestClient(t, fs))
		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		gt.A(t, resp.Texts).Length(0)

		history, err := session.History()
		gt.NoError(t, err).Required()
		gt.A(t, history.Messages).Length(1)
	})

	t.Run("API key header", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeJSON(w, http.StatusUnauthorized, map[string]any{"error": "unauthorized"})
		})
		session := newTestSession(t, newTestClient(t, fs, ollama.WithAPIKey("secret-key")))
		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
		gt.Error(t, err)
		gt.Equal(t, fs.Last(t).Header.Get("Authorization"), "Bearer secret-key")

		gt.False(t, strings.Contains(err.Error(), "secret-key"))
		var goErr *goerr.Error
		gt.True(t, errors.As(err, &goErr))
		for _, v := range goErr.Values() {
			gt.False(t, strings.Contains(stringify(t, v), "secret-key"))
		}
	})

	t.Run("no API key header", func(t *testing.T) {
		fs := newFakeServer(t, replyText("ok", ""))
		_, err := newTestSession(t, newTestClient(t, fs)).Generate(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		gt.Equal(t, fs.Last(t).Header.Get("Authorization"), "")
	})

	t.Run("content block middleware", func(t *testing.T) {
		fs := newFakeServer(t, replyText("ok", ""))
		replaced, err := ollama.HistoryFromWire([]byte(`[{"role":"user","content":"replaced"},{"role":"assistant","content":"earlier"}]`))
		gt.NoError(t, err).Required()

		var seenSystemPrompt string
		mw := func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
			return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
				seenSystemPrompt = req.SystemPrompt
				req.History = replaced
				return next(ctx, req)
			}
		}
		session := newTestSession(t, newTestClient(t, fs),
			gollem.WithSessionSystemPrompt("sys"),
			gollem.WithSessionContentBlockMiddleware(mw))

		_, err = session.Generate(ctx, []gollem.Input{gollem.Text("now")})
		gt.NoError(t, err).Required()
		gt.Equal(t, seenSystemPrompt, "sys")

		msgs := messagesOf(t, fs.Last(t))
		gt.A(t, msgs).Length(4).Required()
		gt.Equal(t, msgs[1]["content"], any("replaced"))
		gt.Equal(t, msgs[2]["content"], any("earlier"))
		gt.Equal(t, msgs[3]["content"], any("now"))
	})

	t.Run("trace", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeJSON(w, http.StatusOK, map[string]any{
				"model": "test-model",
				"message": map[string]any{
					"role":       "assistant",
					"content":    "calling",
					"tool_calls": []any{map[string]any{"id": "c1", "function": map[string]any{"name": "get_weather", "arguments": map[string]any{"city": "Tokyo"}}}},
				},
				"done":              true,
				"prompt_eval_count": 12,
				"eval_count":        5,
			})
		})
		rec := trace.New()
		tctx := trace.WithHandler(rec.StartAgentExecute(ctx), rec)
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionSystemPrompt("sys"))
		_, err := session.Generate(tctx, []gollem.Input{gollem.Text("weather?")})
		gt.NoError(t, err).Required()
		rec.EndAgentExecute(tctx, nil)

		spans := llmCallSpans(rec.Trace().RootSpan)
		gt.A(t, spans).Length(1).Required()
		data := spans[0].LLMCall
		gt.NotNil(t, data)
		gt.Equal(t, data.Model, "test-model")
		gt.Equal(t, data.InputTokens, 12)
		gt.Equal(t, data.OutputTokens, 5)
		gt.Equal(t, data.Request.SystemPrompt, "sys")
		gt.A(t, data.Request.Messages).Length(1)
		gt.Equal(t, data.Response.Texts, []string{"calling"})
		gt.A(t, data.Response.FunctionCalls).Length(1)
		gt.Equal(t, data.Response.FunctionCalls[0].Name, "get_weather")
	})

	t.Run("trace records error", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeJSON(w, http.StatusNotFound, map[string]any{"error": "model 'test-model' not found"})
		})
		rec := trace.New()
		tctx := trace.WithHandler(rec.StartAgentExecute(ctx), rec)
		_, err := newTestSession(t, newTestClient(t, fs)).Generate(tctx, []gollem.Input{gollem.Text("hi")})
		gt.Error(t, err)
		rec.EndAgentExecute(tctx, nil)

		spans := llmCallSpans(rec.Trace().RootSpan)
		gt.A(t, spans).Length(1).Required()
		gt.True(t, spans[0].Error != "")
	})

	t.Run("failed call leaves the history unchanged and a retry sends the input once", func(t *testing.T) {
		var calls atomic.Int32
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			if calls.Add(1) == 2 {
				writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "server busy"})
				return
			}
			replyText("ok", "")(w, req)
		})
		session := newTestSession(t, newTestClient(t, fs))

		_, err := session.Generate(ctx, []gollem.Input{gollem.Text("first")})
		gt.NoError(t, err).Required()
		before, err := session.History()
		gt.NoError(t, err).Required()

		_, err = session.Generate(ctx, []gollem.Input{gollem.Text("second")})
		gt.Error(t, err)
		after, err := session.History()
		gt.NoError(t, err).Required()
		gt.Equal(t, after.Messages, before.Messages)

		_, err = session.Generate(ctx, []gollem.Input{gollem.Text("second")})
		gt.NoError(t, err).Required()
		msgs := messagesOf(t, fs.Last(t))
		gt.A(t, msgs).Length(3).Required()
		gt.Equal(t, msgs[0]["content"], any("first"))
		gt.Equal(t, msgs[1]["content"], any("ok"))
		gt.Equal(t, msgs[2]["content"], any("second"))

		history, err := session.History()
		gt.NoError(t, err).Required()
		gt.A(t, history.Messages).Length(4)
	})
}

func TestGenerateTokenLimit(t *testing.T) {
	ctx := context.Background()
	// The body the server returned for a prompt larger than num_ctx with
	// truncate=false (Ollama v0.35.1, qwen3:0.6b).
	const overflowError = `{"error":{"code":400,"message":"request (4021 tokens) exceeds the available context size (512 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":4021,"n_ctx":512}}`

	type testCase struct {
		status   int
		message  string
		expected bool
	}
	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
				writeJSON(w, tc.status, map[string]any{"error": tc.message})
			})
			session := newTestSession(t, newTestClient(t, fs))
			_, err := session.Generate(ctx, []gollem.Input{gollem.Text("long")})
			gt.Error(t, err)
			gt.Equal(t, goerr.HasTag(err, gollem.ErrTagTokenExceeded), tc.expected)
		}
	}

	t.Run("context size exceeded", runTest(testCase{status: http.StatusBadRequest, message: overflowError, expected: true}))
	t.Run("other runner error type", runTest(testCase{
		status:  http.StatusBadRequest,
		message: `{"error":{"code":400,"message":"bad","type":"invalid_request_error"}}`,
	}))
	t.Run("plain message", runTest(testCase{status: http.StatusBadRequest, message: "model is required"}))
	t.Run("same body with another status", runTest(testCase{status: http.StatusInternalServerError, message: overflowError}))
}

func TestStream(t *testing.T) {
	ctx := context.Background()

	writeLines := func(w http.ResponseWriter, lines ...string) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)
		for _, line := range lines {
			_, _ = io.WriteString(w, line+"\n")
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}
	collect := func(t *testing.T, ch <-chan *gollem.Response) []*gollem.Response {
		t.Helper()
		var result []*gollem.Response
		for resp := range ch {
			result = append(result, resp)
		}
		return result
	}

	t.Run("thinking, text, tool calls and token counts", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeLines(w,
				`{"message":{"role":"assistant","content":"","thinking":"th1"},"done":false}`,
				`{"message":{"role":"assistant","content":"","thinking":"th2"},"done":false}`,
				`{"message":{"role":"assistant","content":"Hel"},"done":false}`,
				`{"message":{"role":"assistant","content":"lo"},"done":false}`,
				`{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c1","function":{"index":0,"name":"get_weather","arguments":{"city":"Tokyo"}}}]},"done":false}`,
				`{"message":{"role":"assistant","content":"","tool_calls":[{"id":"c2","function":{"index":1,"name":"get_weather","arguments":{"city":"Osaka"}}}]},"done":false}`,
				`{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":20,"prompt_eval_cached_count":7,"eval_count":9}`,
			)
		})
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionTools(weatherTool{}))
		rec := trace.New()
		tctx := trace.WithHandler(rec.StartAgentExecute(ctx), rec)
		ch, err := session.Stream(tctx, []gollem.Input{gollem.Text("weather?")})
		gt.NoError(t, err).Required()
		responses := collect(t, ch)

		spans := llmCallSpans(rec.Trace().RootSpan)
		gt.A(t, spans).Length(1).Required()
		gt.Equal(t, spans[0].LLMCall.Response.FinishReason, "stop")
		gt.Equal(t, spans[0].LLMCall.Response.Thoughts, []string{"th1th2"})
		gt.Equal(t, spans[0].LLMCall.Response.Texts, []string{"Hello"})

		gt.Equal(t, fs.Last(t).Body["stream"], any(true))
		gt.Equal(t, fs.Last(t).Body["truncate"], any(false))
		gt.Equal(t, fs.Last(t).Body["shift"], any(false))
		gt.Equal(t, fs.Last(t).Header.Get("Accept"), "application/x-ndjson")

		gt.A(t, responses).Length(6).Required()
		gt.Equal(t, responses[0].Thoughts, []string{"th1"})
		gt.Equal(t, responses[1].Thoughts, []string{"th2"})
		gt.Equal(t, responses[2].Texts, []string{"Hel"})
		gt.Equal(t, responses[3].Texts, []string{"lo"})
		gt.A(t, responses[4].FunctionCalls).Length(2).Required()
		gt.Equal(t, responses[4].FunctionCalls[0].ID, "c1")
		gt.Equal(t, responses[4].FunctionCalls[1].Arguments["city"], any("Osaka"))
		gt.Equal(t, responses[5].InputToken, 20)
		gt.Equal(t, responses[5].OutputToken, 9)
		gt.Equal(t, responses[5].CacheReadInputToken, 7)
		gt.Equal(t, responses[5].FinishReason, "stop")
		for _, r := range responses[:5] {
			gt.Equal(t, r.FinishReason, "")
		}
		for _, r := range responses {
			gt.NoError(t, r.Error)
		}

		history, err := session.History()
		gt.NoError(t, err).Required()
		gt.A(t, history.Messages).Length(2).Required()
		last := history.Messages[1]
		gt.Equal(t, last.Role, gollem.RoleAssistant)
		gt.A(t, last.Contents).Length(4).Required()
		thinking, err := last.Contents[0].GetThinkingContent()
		gt.NoError(t, err).Required()
		gt.Equal(t, thinking.Text, "th1th2")
		text, err := last.Contents[1].GetTextContent()
		gt.NoError(t, err).Required()
		gt.Equal(t, text.Text, "Hello")
	})

	runStreamError := func(lines []string, expectedTag bool) func(t *testing.T) {
		return func(t *testing.T) {
			fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
				writeLines(w, lines...)
			})
			session := newTestSession(t, newTestClient(t, fs))
			ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("hi")})
			gt.NoError(t, err).Required()
			responses := collect(t, ch)
			gt.A(t, responses).Longer(0).Required()
			lastResp := responses[len(responses)-1]
			gt.Error(t, lastResp.Error)
			gt.Equal(t, goerr.HasTag(lastResp.Error, gollem.ErrTagTokenExceeded), expectedTag)

			// Neither the input nor the partial reply is kept.
			history, err := session.History()
			gt.NoError(t, err).Required()
			gt.A(t, history.Messages).Length(0)
		}
	}
	t.Run("error line", runStreamError([]string{
		`{"message":{"role":"assistant","content":"par"},"done":false}`,
		`{"error":"boom"}`,
	}, false))
	t.Run("body ends before done", runStreamError([]string{
		`{"message":{"role":"assistant","content":"par"},"done":false}`,
	}, false))
	t.Run("error line reporting context overflow", runStreamError([]string{
		`{"error":"{\"error\":{\"code\":400,\"message\":\"exceeds\",\"type\":\"exceed_context_size_error\"}}","status":400}`,
	}, true))

	t.Run("HTTP error is returned by Stream", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeJSON(w, http.StatusNotFound, map[string]any{"error": "model 'test-model' not found"})
		})
		ch, err := newTestSession(t, newTestClient(t, fs)).Stream(ctx, []gollem.Input{gollem.Text("hi")})
		gt.Error(t, err).Contains("model 'test-model' not found")
		gt.Value(t, ch).Nil()
	})

	t.Run("HTTP context overflow is tagged", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeJSON(w, http.StatusBadRequest, map[string]any{
				"error": `{"error":{"code":400,"message":"exceeds","type":"exceed_context_size_error"}}`,
			})
		})
		_, err := newTestSession(t, newTestClient(t, fs)).Stream(ctx, []gollem.Input{gollem.Text("hi")})
		gt.Error(t, err)
		gt.True(t, goerr.HasTag(err, gollem.ErrTagTokenExceeded))
	})

	t.Run("context cancelled while receiving", func(t *testing.T) {
		release := make(chan struct{})
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeLines(w, `{"message":{"role":"assistant","content":"first"},"done":false}`)
			<-release
		})
		defer close(release)

		cctx, cancel := context.WithCancel(ctx)
		ch, err := newTestSession(t, newTestClient(t, fs)).Stream(cctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()

		first := <-ch
		gt.Equal(t, first.Texts, []string{"first"})
		cancel()

		var last *gollem.Response
		for resp := range ch {
			last = resp
		}
		gt.NotNil(t, last)
		gt.Error(t, last.Error).Is(context.Canceled)
	})

	t.Run("stream middleware receives chunks", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeLines(w,
				`{"message":{"role":"assistant","content":"a"},"done":false}`,
				`{"message":{"role":"assistant","content":"b"},"done":true,"prompt_eval_count":1,"eval_count":2}`,
			)
		})
		var mu sync.Mutex
		var seen []string
		mw := func(next gollem.ContentStreamHandler) gollem.ContentStreamHandler {
			return func(ctx context.Context, req *gollem.ContentRequest) (<-chan *gollem.ContentResponse, error) {
				src, err := next(ctx, req)
				if err != nil {
					return nil, err
				}
				out := make(chan *gollem.ContentResponse)
				go func() {
					defer close(out)
					for r := range src {
						mu.Lock()
						seen = append(seen, r.Texts...)
						mu.Unlock()
						out <- r
					}
				}()
				return out, nil
			}
		}
		session := newTestSession(t, newTestClient(t, fs), gollem.WithSessionContentStreamMiddleware(mw))
		ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		collect(t, ch)
		mu.Lock()
		defer mu.Unlock()
		gt.Equal(t, seen, []string{"a", "b"})
	})

	t.Run("trace", func(t *testing.T) {
		fs := newFakeServer(t, func(w http.ResponseWriter, req recordedRequest) {
			writeLines(w,
				`{"message":{"role":"assistant","content":"a"},"done":false}`,
				`{"message":{"role":"assistant","content":"b"},"done":true,"prompt_eval_count":4,"eval_count":2}`,
			)
		})
		rec := trace.New()
		tctx := trace.WithHandler(rec.StartAgentExecute(ctx), rec)
		ch, err := newTestSession(t, newTestClient(t, fs)).Stream(tctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		collect(t, ch)
		rec.EndAgentExecute(tctx, nil)

		spans := llmCallSpans(rec.Trace().RootSpan)
		gt.A(t, spans).Length(1).Required()
		gt.Equal(t, spans[0].LLMCall.Response.Texts, []string{"ab"})
		gt.Equal(t, spans[0].LLMCall.InputTokens, 4)
		gt.Equal(t, spans[0].LLMCall.Model, "test-model")
	})
}

func TestCountToken(t *testing.T) {
	fs := newFakeServer(t, replyText("ok", ""))
	session := newTestSession(t, newTestClient(t, fs))
	n, err := session.CountToken(context.Background(), gollem.Text("hello"))
	gt.Error(t, err).Is(gollem.ErrUnsupportedOperation)
	gt.Equal(t, n, 0)
	gt.A(t, fs.Requests()).Length(0)
}

func TestAppendHistory(t *testing.T) {
	ctx := context.Background()
	fs := newFakeServer(t, replyText("ok", ""))
	session := newTestSession(t, newTestClient(t, fs))

	gt.NoError(t, session.AppendHistory(nil))

	appended, err := ollama.HistoryFromWire([]byte(`[{"role":"user","content":"before"},{"role":"assistant","content":"reply"}]`))
	gt.NoError(t, err).Required()
	gt.NoError(t, session.AppendHistory(appended)).Required()

	_, err = session.Generate(ctx, []gollem.Input{gollem.Text("after")})
	gt.NoError(t, err).Required()
	msgs := messagesOf(t, fs.Last(t))
	gt.A(t, msgs).Length(3).Required()
	gt.Equal(t, msgs[0]["content"], any("before"))
	gt.Equal(t, msgs[1]["content"], any("reply"))
	gt.Equal(t, msgs[2]["content"], any("after"))
}

func llmCallSpans(span *trace.Span) []*trace.Span {
	if span == nil {
		return nil
	}
	var result []*trace.Span
	if span.Kind == trace.SpanKindLLMCall {
		result = append(result, span)
	}
	for _, child := range span.Children {
		result = append(result, llmCallSpans(child)...)
	}
	return result
}

func stringify(t *testing.T, v any) string {
	t.Helper()
	data, err := json.Marshal(v)
	gt.NoError(t, err).Required()
	return string(data)
}

// pngData is a 1x1 PNG image.
var pngData = []byte{
	0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
	0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
	0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4, 0x89, 0x00, 0x00, 0x00,
	0x0d, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
	0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00, 0x00, 0x00, 0x00, 0x49,
	0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
}

// Integration tests against a real Ollama server. They run only when
// TEST_OLLAMA_MODEL is set.
func newIntegrationClient(t *testing.T, opts ...ollama.Option) *ollama.Client {
	t.Helper()
	model, ok := os.LookupEnv("TEST_OLLAMA_MODEL")
	if !ok {
		t.Skip("TEST_OLLAMA_MODEL is not set")
	}
	if baseURL, ok := os.LookupEnv("TEST_OLLAMA_BASE_URL"); ok {
		opts = append(opts, ollama.WithBaseURL(baseURL))
	}
	client, err := ollama.New(context.Background(), model, opts...)
	gt.NoError(t, err).Required()
	return client
}

func TestIntegration(t *testing.T) {
	client := newIntegrationClient(t, ollama.WithThink(false), ollama.WithTemperature(0))
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	t.Run("generate", func(t *testing.T) {
		session := newTestSession(t, client)
		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("Reply with the single word: hello")})
		gt.NoError(t, err).Required()
		gt.A(t, resp.Texts).Longer(0)
		gt.N(t, resp.InputToken).Greater(0)
		gt.N(t, resp.OutputToken).Greater(0)
	})

	t.Run("stream", func(t *testing.T) {
		session := newTestSession(t, client)
		ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("Count from 1 to 5.")})
		gt.NoError(t, err).Required()
		var text strings.Builder
		for resp := range ch {
			gt.NoError(t, resp.Error).Required()
			for _, s := range resp.Texts {
				text.WriteString(s)
			}
		}
		gt.True(t, text.Len() > 0)

		history, err := session.History()
		gt.NoError(t, err).Required()
		gt.A(t, history.Messages).Length(2)
	})

	t.Run("tool call round trip", func(t *testing.T) {
		session := newTestSession(t, client, gollem.WithSessionTools(weatherTool{}))
		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("What is the weather in Tokyo? Use the get_weather tool.")})
		gt.NoError(t, err).Required()
		gt.A(t, resp.FunctionCalls).Longer(0).Required()
		call := resp.FunctionCalls[0]
		gt.Equal(t, call.Name, "get_weather")
		gt.True(t, call.ID != "")

		resp, err = session.Generate(ctx, []gollem.Input{
			gollem.FunctionResponse{ID: call.ID, Name: call.Name, Data: map[string]any{"weather": "sunny"}},
		})
		gt.NoError(t, err).Required()
		gt.A(t, resp.Texts).Longer(0)
	})

	t.Run("response schema", func(t *testing.T) {
		schema := &gollem.Parameter{
			Type: gollem.TypeObject,
			Properties: map[string]*gollem.Parameter{
				"city": {Type: gollem.TypeString, Required: true},
			},
		}
		session := newTestSession(t, client,
			gollem.WithSessionContentType(gollem.ContentTypeJSON),
			gollem.WithSessionResponseSchema(schema))
		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("Name the capital of Japan as JSON.")})
		gt.NoError(t, err).Required()
		gt.A(t, resp.Texts).Longer(0).Required()
		var out map[string]any
		gt.NoError(t, json.Unmarshal([]byte(strings.Join(resp.Texts, "")), &out)).Required()
		_, ok := out["city"]
		gt.True(t, ok)
	})

	t.Run("response schema with a union", func(t *testing.T) {
		session := newTestSession(t, client,
			gollem.WithSessionContentType(gollem.ContentTypeJSON),
			gollem.WithSessionResponseSchema(newBlocksParameter()))
		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text(
			"Write a three-block note about tea: a paragraph, a warning callout, then a paragraph. Reply in JSON.")})
		gt.NoError(t, err).Required()
		gt.A(t, resp.Texts).Longer(0).Required()

		var reply struct {
			Blocks []map[string]any `json:"blocks"`
		}
		gt.NoError(t, json.Unmarshal([]byte(strings.Join(resp.Texts, "")), &reply)).Required()
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
	})

	t.Run("context overflow is tagged", func(t *testing.T) {
		small := newIntegrationClient(t, ollama.WithThink(false), ollama.WithNumCtx(512))
		session := newTestSession(t, small)
		long := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 400)
		_, err := session.Generate(ctx, []gollem.Input{gollem.Text(long)})
		gt.Error(t, err)
		gt.True(t, goerr.HasTag(err, gollem.ErrTagTokenExceeded))
	})
}
