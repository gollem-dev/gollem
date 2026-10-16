package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/gt"
	openaiapi "github.com/sashabaranov/go-openai"
)

// sentResponsesRequest is the part of a /v1/responses request body the tests inspect.
type sentResponsesRequest struct {
	Model           string           `json:"model"`
	Instructions    string           `json:"instructions"`
	Store           *bool            `json:"store"`
	Include         []string         `json:"include"`
	MaxOutputTokens *int             `json:"max_output_tokens"`
	Reasoning       *sentReasoning   `json:"reasoning"`
	ToolChoice      any              `json:"tool_choice"`
	Tools           []map[string]any `json:"tools"`
	Input           []sentInputItem  `json:"input"`
	Text            *sentTextConfig  `json:"text"`
	Stream          bool             `json:"stream"`
	Raw             map[string]any   `json:"-"`
}

type sentReasoning struct {
	Effort string `json:"effort"`
}

type sentTextConfig struct {
	Format *struct {
		Type   string          `json:"type"`
		Name   string          `json:"name"`
		Schema json.RawMessage `json:"schema"`
	} `json:"format"`
	Verbosity string `json:"verbosity"`
}

type sentInputItem struct {
	Type             string          `json:"type"`
	Role             string          `json:"role"`
	Content          json.RawMessage `json:"content"`
	Phase            string          `json:"phase"`
	ID               string          `json:"id"`
	CallID           string          `json:"call_id"`
	Name             string          `json:"name"`
	Arguments        string          `json:"arguments"`
	Output           string          `json:"output"`
	EncryptedContent string          `json:"encrypted_content"`
	Summary          []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"summary"`
}

// responsesReply writes the reply for one request. n is the 0-based index of the
// request among those the server received. It runs in the server's goroutine,
// so it reports failures with t.Errorf.
type responsesReply func(t *testing.T, w http.ResponseWriter, req sentResponsesRequest, n int)

// responsesServer is an httptest server for /v1/responses that records every
// request it receives.
type responsesServer struct {
	srv      *httptest.Server
	mu       sync.Mutex
	requests []sentResponsesRequest
}

func newResponsesServer(t *testing.T, reply responsesReply) *responsesServer {
	t.Helper()
	rs := &responsesServer{}
	rs.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Errorf("unexpected path %s", r.URL.Path)
			http.Error(w, "unexpected path "+r.URL.Path, http.StatusNotFound)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("failed to read request body: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		var req sentResponsesRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("failed to decode request body: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if err := json.Unmarshal(body, &req.Raw); err != nil {
			t.Errorf("failed to decode request body: %v", err)
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		rs.mu.Lock()
		n := len(rs.requests)
		rs.requests = append(rs.requests, req)
		rs.mu.Unlock()
		reply(t, w, req, n)
	}))
	t.Cleanup(rs.srv.Close)
	return rs
}

func (rs *responsesServer) client(t *testing.T, options ...openai.Option) *openai.Client {
	t.Helper()
	options = append([]openai.Option{
		openai.WithBaseURL(rs.srv.URL + "/v1"),
		openai.WithModel("gpt-test"),
		openai.WithResponsesAPI(),
	}, options...)
	client, err := openai.New(context.Background(), "test-key", options...)
	gt.NoError(t, err).Required()
	return client
}

func (rs *responsesServer) sent() []sentResponsesRequest {
	rs.mu.Lock()
	defer rs.mu.Unlock()
	return append([]sentResponsesRequest(nil), rs.requests...)
}

// replyJSON replies with the same response body to every request.
func replyJSON(body string) responsesReply {
	return func(t *testing.T, w http.ResponseWriter, _ sentResponsesRequest, _ int) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := io.WriteString(w, body); err != nil {
			t.Errorf("failed to write reply: %v", err)
		}
	}
}

// replySequence replies with bodies[n] to the n-th request.
func replySequence(bodies ...string) responsesReply {
	return func(t *testing.T, w http.ResponseWriter, req sentResponsesRequest, n int) {
		if n >= len(bodies) {
			t.Errorf("request %d has no reply", n)
			http.Error(w, "no more replies", http.StatusInternalServerError)
			return
		}
		replyJSON(bodies[n])(t, w, req, n)
	}
}

// writeEvent writes one server-sent event, naming it by the event's type field.
func writeEvent(t *testing.T, w http.ResponseWriter, event string) {
	var head struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal([]byte(event), &head); err != nil {
		t.Errorf("invalid test event %s: %v", event, err)
		return
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", head.Type, event); err != nil {
		t.Errorf("failed to write event: %v", err)
		return
	}
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

// replyEvents streams the events as server-sent events.
func replyEvents(events ...string) responsesReply {
	return func(t *testing.T, w http.ResponseWriter, _ sentResponsesRequest, _ int) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, event := range events {
			writeEvent(t, w, event)
		}
	}
}

const (
	textReplyBody = `{
		"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-test-2026",
		"output": [
			{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
			 "content": [{"type": "output_text", "text": "hello there", "annotations": []}]}
		],
		"usage": {
			"input_tokens": 1000,
			"input_tokens_details": {"cached_tokens": 300, "cache_write_tokens": 200},
			"output_tokens": 50,
			"output_tokens_details": {"reasoning_tokens": 20},
			"total_tokens": 1050
		}
	}`

	functionCallReplyBody = `{
		"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-test",
		"output": [
			{"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-1",
			 "summary": [{"type": "summary_text", "text": "need to look it up"}]},
			{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup",
			 "arguments": "{\"key\":\"alpha\"}", "status": "completed"}
		],
		"usage": {"input_tokens": 100, "output_tokens": 10}
	}`

	finalAnswerReplyBody = `{
		"id": "resp_2", "object": "response", "status": "completed", "model": "gpt-test",
		"output": [
			{"type": "reasoning", "id": "rs_2", "encrypted_content": "enc-2", "summary": []},
			{"type": "message", "id": "msg_2", "role": "assistant", "status": "completed",
			 "content": [{"type": "output_text", "text": "the value is ok", "annotations": []}]}
		],
		"usage": {"input_tokens": 200, "output_tokens": 20}
	}`
)

// inputText returns the text of an input message item.
func inputText(t *testing.T, item sentInputItem) string {
	t.Helper()
	var s string
	if err := json.Unmarshal(item.Content, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	gt.NoError(t, json.Unmarshal(item.Content, &parts)).Required()
	var texts []string
	for _, p := range parts {
		texts = append(texts, p.Text)
	}
	return strings.Join(texts, "")
}

func TestResponsesGenerateText(t *testing.T) {
	rs := newResponsesServer(t, replyJSON(textReplyBody))
	client := rs.client(t)

	session, err := client.NewSession(context.Background(), gollem.WithSessionSystemPrompt("be brief"))
	gt.NoError(t, err).Required()

	rec := trace.New()
	ctx := rec.StartAgentExecute(context.Background())
	ctx = trace.WithHandler(ctx, rec)

	resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("say hello")})
	gt.NoError(t, err).Required()
	rec.EndAgentExecute(ctx, nil)

	gt.A(t, resp.Texts).Length(1).Required()
	gt.Equal(t, "hello there", resp.Texts[0])
	gt.A(t, resp.FunctionCalls).Length(0)
	gt.Equal(t, 1000, resp.InputToken)
	gt.Equal(t, 50, resp.OutputToken)
	gt.Equal(t, 300, resp.CacheReadInputToken)
	gt.Equal(t, 200, resp.CacheCreationInputToken)

	sent := rs.sent()
	gt.A(t, sent).Length(1).Required()
	gt.Equal(t, "gpt-test", sent[0].Model)
	gt.Equal(t, "be brief", sent[0].Instructions)
	gt.A(t, sent[0].Input).Length(1).Required()
	gt.Equal(t, "message", sent[0].Input[0].Type)
	gt.Equal(t, "user", sent[0].Input[0].Role)
	gt.Equal(t, "say hello", inputText(t, sent[0].Input[0]))

	var llmSpan *trace.Span
	for _, child := range rec.Trace().RootSpan.Children {
		if child.Kind == trace.SpanKindLLMCall {
			llmSpan = child
		}
	}
	gt.V(t, llmSpan).NotNil().Required()
	gt.Equal(t, "gpt-test-2026", llmSpan.LLMCall.Model)
	gt.Equal(t, 1000, llmSpan.LLMCall.InputTokens)
	gt.Equal(t, 50, llmSpan.LLMCall.OutputTokens)
	gt.Equal(t, 300, llmSpan.LLMCall.CacheReadInputTokens)
	gt.Equal(t, 200, llmSpan.LLMCall.CacheCreationInputTokens)
	gt.Equal(t, "be brief", llmSpan.LLMCall.Request.SystemPrompt)
	gt.A(t, llmSpan.LLMCall.Request.Messages).Length(1).Required()
	gt.Equal(t, "say hello", llmSpan.LLMCall.Request.Messages[0].Contents[0].Text)
	gt.A(t, llmSpan.LLMCall.Response.Texts).Length(1).Required()
	gt.Equal(t, "hello there", llmSpan.LLMCall.Response.Texts[0])
}

func TestResponsesFinishReason(t *testing.T) {
	const incompleteReplyBody = `{
		"id": "resp_1", "object": "response", "status": "incomplete", "model": "gpt-test",
		"incomplete_details": {"reason": "max_output_tokens"},
		"output": [
			{"type": "message", "id": "msg_1", "role": "assistant", "status": "incomplete",
			 "content": [{"type": "output_text", "text": "partial", "annotations": []}]}
		],
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`

	type testCase struct {
		body     string
		expected string
	}
	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			rs := newResponsesServer(t, replyJSON(tc.body))
			session, err := rs.client(t).NewSession(context.Background())
			gt.NoError(t, err).Required()

			rec := trace.New()
			ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
			resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
			gt.NoError(t, err).Required()
			gt.Equal(t, tc.expected, resp.FinishReason)
			gt.Equal(t, resp.FinishReason, findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response.FinishReason)
		}
	}
	t.Run("completed reports the status", runTest(testCase{body: textReplyBody, expected: "completed"}))
	t.Run("incomplete reports the reason", runTest(testCase{body: incompleteReplyBody, expected: "max_output_tokens"}))

	t.Run("Stream sets the reason on the response for the final event", func(t *testing.T) {
		rs := newResponsesServer(t, replyEvents(
			`{"type":"response.output_text.delta","sequence_number":1,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"partial"}`,
			`{"type":"response.output_item.done","sequence_number":2,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"incomplete","content":[{"type":"output_text","text":"partial","annotations":[]}]}}`,
			`{"type":"response.incomplete","sequence_number":3,"response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"model":"gpt-test","output":[],"usage":{"input_tokens":10,"output_tokens":5}}}`,
		))
		session, err := rs.client(t).NewSession(context.Background())
		gt.NoError(t, err).Required()

		rec := trace.New()
		ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
		ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		var reasons []string
		for resp := range ch {
			gt.NoError(t, resp.Error).Required()
			reasons = append(reasons, resp.FinishReason)
		}
		gt.Equal(t, []string{"", "max_output_tokens"}, reasons)
		gt.Equal(t, "max_output_tokens", findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response.FinishReason)
	})
}

// TestResponsesHistoryAfterResponseWithoutText pins the request that follows a
// response without text, which is what gollem.Query sends when it asks the
// model to continue.
func TestResponsesHistoryAfterResponseWithoutText(t *testing.T) {
	type testCase struct {
		output string
		items  []string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			rs := newResponsesServer(t, replyJSON(`{
				"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-test",
				"output": [`+tc.output+`],
				"usage": {"input_tokens": 100, "output_tokens": 10}
			}`))
			session, err := rs.client(t).NewSession(context.Background())
			gt.NoError(t, err).Required()

			resp, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("question")})
			gt.NoError(t, err).Required()
			gt.A(t, resp.Texts).Length(0)
			_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("continue")})
			gt.NoError(t, err).Required()

			sent := rs.sent()
			gt.A(t, sent).Length(2).Required()
			var items []string
			for _, item := range sent[1].Input {
				items = append(items, item.Type+":"+item.Role)
			}
			gt.Equal(t, tc.items, items)
		}
	}

	t.Run("reasoning only", runTest(testCase{
		output: `{"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-1", "summary": []}`,
		items:  []string{"message:user", "reasoning:", "message:user"},
	}))

	// A turn without output items is not kept, so the next user message
	// follows the previous one.
	t.Run("no output item", runTest(testCase{
		items: []string{"message:user", "message:user"},
	}))
}

func TestResponsesTraceRecordsThoughts(t *testing.T) {
	t.Run("Generate", func(t *testing.T) {
		rs := newResponsesServer(t, replyJSON(functionCallReplyBody))
		session, err := rs.client(t).NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
		gt.NoError(t, err).Required()

		rec := trace.New()
		ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("what is alpha?")})
		gt.NoError(t, err).Required()
		gt.Equal(t, []string{"need to look it up"}, resp.Thoughts)

		traced := findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response
		gt.Equal(t, resp.Thoughts, traced.Thoughts)
		gt.A(t, traced.OtherBlockTypes).Length(0)
	})

	t.Run("Stream", func(t *testing.T) {
		rs := newResponsesServer(t, replyEvents(
			`{"type":"response.reasoning_summary_text.delta","sequence_number":1,"output_index":0,"item_id":"rs_1","summary_index":0,"delta":"need to "}`,
			`{"type":"response.reasoning_summary_text.delta","sequence_number":2,"output_index":0,"item_id":"rs_1","summary_index":0,"delta":"look it up"}`,
			`{"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"enc-1","summary":[{"type":"summary_text","text":"need to look it up"}]}}`,
			`{"type":"response.output_item.done","sequence_number":4,"output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"key\":\"alpha\"}","status":"completed"}}`,
			`{"type":"response.completed","sequence_number":5,"response":{"id":"resp_1","status":"completed","model":"gpt-test","output":[],"usage":{"input_tokens":100,"output_tokens":10}}}`,
		))
		session, err := rs.client(t).NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
		gt.NoError(t, err).Required()

		rec := trace.New()
		ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
		ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("what is alpha?")})
		gt.NoError(t, err).Required()
		for resp := range ch {
			gt.NoError(t, resp.Error).Required()
		}

		traced := findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response
		gt.Equal(t, []string{"need to look it up"}, traced.Thoughts)
	})
}

func TestResponsesFunctionCallRoundTrip(t *testing.T) {
	rs := newResponsesServer(t, replySequence(functionCallReplyBody, finalAnswerReplyBody))
	client := rs.client(t)

	session, err := client.NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
	gt.NoError(t, err).Required()

	first, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("what is alpha?")})
	gt.NoError(t, err).Required()
	gt.A(t, first.Thoughts).Length(1).Required()
	gt.Equal(t, "need to look it up", first.Thoughts[0])
	gt.A(t, first.FunctionCalls).Length(1).Required()
	gt.Equal(t, "call_1", first.FunctionCalls[0].ID)
	gt.Equal(t, "lookup", first.FunctionCalls[0].Name)
	gt.Equal(t, map[string]any{"key": "alpha"}, first.FunctionCalls[0].Arguments)

	second, err := session.Generate(context.Background(), []gollem.Input{
		gollem.FunctionResponse{ID: "call_1", Name: "lookup", Data: map[string]any{"value": "ok"}},
	})
	gt.NoError(t, err).Required()
	gt.A(t, second.Texts).Length(1).Required()
	gt.Equal(t, "the value is ok", second.Texts[0])

	sent := rs.sent()
	gt.A(t, sent).Length(2).Required()

	// The tool is sent as an inline function tool.
	gt.A(t, sent[0].Tools).Length(1).Required()
	gt.Equal(t, "function", sent[0].Tools[0]["type"])
	gt.Equal(t, "lookup", sent[0].Tools[0]["name"])
	// strict is sent as false so that optional parameters stay optional.
	gt.Equal(t, any(false), sent[0].Tools[0]["strict"])

	// The second request carries the first turn's reasoning item and the
	// function call, followed by its output under the same call_id.
	input := sent[1].Input
	gt.A(t, input).Length(4).Required()
	gt.Equal(t, "message", input[0].Type)
	gt.Equal(t, "what is alpha?", inputText(t, input[0]))

	gt.Equal(t, "reasoning", input[1].Type)
	gt.Equal(t, "rs_1", input[1].ID)
	gt.Equal(t, "enc-1", input[1].EncryptedContent)
	gt.A(t, input[1].Summary).Length(1).Required()
	gt.Equal(t, "summary_text", input[1].Summary[0].Type)
	gt.Equal(t, "need to look it up", input[1].Summary[0].Text)

	gt.Equal(t, "function_call", input[2].Type)
	gt.Equal(t, "call_1", input[2].CallID)
	gt.Equal(t, "lookup", input[2].Name)
	gt.Equal(t, `{"key":"alpha"}`, input[2].Arguments)

	gt.Equal(t, "function_call_output", input[3].Type)
	gt.Equal(t, "call_1", input[3].CallID)
	gt.Equal(t, `{"value":"ok"}`, input[3].Output)
}

func TestResponsesFunctionErrorOutput(t *testing.T) {
	rs := newResponsesServer(t, replySequence(functionCallReplyBody, finalAnswerReplyBody))
	client := rs.client(t)

	session, err := client.NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
	gt.NoError(t, err).Required()

	_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("what is alpha?")})
	gt.NoError(t, err).Required()
	_, err = session.Generate(context.Background(), []gollem.Input{
		gollem.FunctionResponse{ID: "call_1", Name: "lookup", Error: errors.New("not found")},
	})
	gt.NoError(t, err).Required()

	input := rs.sent()[1].Input
	gt.A(t, input).Length(4).Required()
	gt.Equal(t, "function_call_output", input[3].Type)
	gt.Equal(t, "call_1", input[3].CallID)
	gt.Equal(t, `{"error":"not found"}`, input[3].Output)
}

func TestResponsesRequestParameters(t *testing.T) {
	type testCase struct {
		options         []openai.Option
		generateOptions []gollem.GenerateOption
		expectEffort    string
		expectVerbosity string
		expectMaxTokens *int
	}

	intPtr := func(n int) *int { return &n }

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			rs := newResponsesServer(t, replyJSON(textReplyBody))
			client := rs.client(t, tc.options...)
			session, err := client.NewSession(context.Background())
			gt.NoError(t, err).Required()

			_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")}, tc.generateOptions...)
			gt.NoError(t, err).Required()

			sent := rs.sent()
			gt.A(t, sent).Length(1).Required()
			req := sent[0]

			gt.V(t, req.Store).NotNil().Required()
			gt.False(t, *req.Store)
			_, hasPrevious := req.Raw["previous_response_id"]
			gt.False(t, hasPrevious)
			gt.Equal(t, []string{"reasoning.encrypted_content"}, req.Include)

			if tc.expectEffort == "" {
				_, hasReasoning := req.Raw["reasoning"]
				gt.False(t, hasReasoning)
			} else {
				gt.V(t, req.Reasoning).NotNil().Required()
				gt.Equal(t, tc.expectEffort, req.Reasoning.Effort)
			}

			// The text session sends no response format, so text appears only
			// to carry the verbosity.
			if tc.expectVerbosity == "" {
				_, hasText := req.Raw["text"]
				gt.False(t, hasText)
			} else {
				gt.V(t, req.Text).NotNil().Required()
				gt.Equal(t, tc.expectVerbosity, req.Text.Verbosity)
			}

			if tc.expectMaxTokens == nil {
				_, hasMax := req.Raw["max_output_tokens"]
				gt.False(t, hasMax)
			} else {
				gt.V(t, req.MaxOutputTokens).NotNil().Required()
				gt.Equal(t, *tc.expectMaxTokens, *req.MaxOutputTokens)
			}
		}
	}

	t.Run("sends no reasoning, verbosity or max_output_tokens by default", runTest(testCase{}))
	t.Run("sends reasoning.effort when set", runTest(testCase{
		options:      []openai.Option{openai.WithReasoningEffort("low")},
		expectEffort: "low",
	}))
	t.Run("sends text.verbosity when set", runTest(testCase{
		options:         []openai.Option{openai.WithVerbosity("low")},
		expectVerbosity: "low",
	}))
	t.Run("sends max_output_tokens from WithMaxTokens", runTest(testCase{
		options:         []openai.Option{openai.WithMaxTokens(100)},
		expectMaxTokens: intPtr(100),
	}))
	t.Run("per-call max tokens overrides the client option", runTest(testCase{
		options:         []openai.Option{openai.WithMaxTokens(100)},
		generateOptions: []gollem.GenerateOption{gollem.WithMaxTokens(200)},
		expectMaxTokens: intPtr(200),
	}))
	t.Run("per-call max tokens without a client option", runTest(testCase{
		generateOptions: []gollem.GenerateOption{gollem.WithMaxTokens(300)},
		expectMaxTokens: intPtr(300),
	}))
}

func TestResponsesPerCallOptions(t *testing.T) {
	t.Run("response schema is sent as text.format", func(t *testing.T) {
		rs := newResponsesServer(t, replyJSON(textReplyBody))
		session, err := rs.client(t).NewSession(context.Background())
		gt.NoError(t, err).Required()

		schema := &gollem.Parameter{
			Title: "answer",
			Type:  gollem.TypeObject,
			Properties: map[string]*gollem.Parameter{
				"value": {Type: gollem.TypeString, Required: true},
			},
		}
		_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")},
			gollem.WithGenerateResponseSchema(schema))
		gt.NoError(t, err).Required()

		req := rs.sent()[0]
		gt.V(t, req.Text).NotNil().Required()
		gt.V(t, req.Text.Format).NotNil().Required()
		gt.Equal(t, "json_schema", req.Text.Format.Type)
		gt.Equal(t, "answer", req.Text.Format.Name)
	})

	t.Run("response schema with a union is sent as anyOf", func(t *testing.T) {
		rs := newResponsesServer(t, replyJSON(textReplyBody))
		session, err := rs.client(t).NewSession(context.Background())
		gt.NoError(t, err).Required()

		_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")},
			gollem.WithGenerateResponseSchema(newBlocksParameter()))
		gt.NoError(t, err).Required()

		req := rs.sent()[0]
		gt.V(t, req.Text).NotNil().Required()
		gt.V(t, req.Text.Format).NotNil().Required()
		schema := req.Text.Format.Schema
		gt.Equal(t, `{"additionalProperties":false,"properties":{"blocks":{"items":{"anyOf":[`+
			`{"additionalProperties":false,"properties":{"kind":{"enum":["paragraph"],"type":"string"},"text":{"type":"string"}},"required":["kind","text"],"type":"object"},`+
			`{"additionalProperties":false,"properties":{"kind":{"enum":["callout"],"type":"string"},"text":{"type":"string"},"tone":{"enum":["info","warning"],"type":"string"}},"required":["kind","text"],"type":"object"}`+
			`],"description":"a block"},"type":"array"}},"required":["blocks"],"type":"object"}`, string(schema))
	})

	t.Run("tool calls disabled sends tool_choice none", func(t *testing.T) {
		rs := newResponsesServer(t, replyJSON(textReplyBody))
		session, err := rs.client(t).NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
		gt.NoError(t, err).Required()

		_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")},
			gollem.WithToolCallsDisabled())
		gt.NoError(t, err).Required()

		req := rs.sent()[0]
		gt.Equal(t, any("none"), req.ToolChoice)
		gt.A(t, req.Tools).Length(1)
	})
}

func TestResponsesStream(t *testing.T) {
	t.Run("text deltas in order and final usage", func(t *testing.T) {
		rs := newResponsesServer(t, replyEvents(
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","status":"in_progress","output":[]}}`,
			`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}`,
			`{"type":"response.output_text.delta","sequence_number":2,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"Hel"}`,
			`{"type":"response.output_text.delta","sequence_number":3,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"lo"}`,
			`{"type":"response.output_text.delta","sequence_number":4,"output_index":0,"content_index":0,"item_id":"msg_1","delta":" world"}`,
			`{"type":"response.output_item.done","sequence_number":5,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Hello world","annotations":[]}]}}`,
			`{"type":"response.completed","sequence_number":6,"response":{"id":"resp_1","status":"completed","model":"gpt-test","output":[],"usage":{"input_tokens":1000,"input_tokens_details":{"cached_tokens":300,"cache_write_tokens":200},"output_tokens":50,"output_tokens_details":{"reasoning_tokens":20},"total_tokens":1050}}}`,
		))
		session, err := rs.client(t).NewSession(context.Background())
		gt.NoError(t, err).Required()

		rec := trace.New()
		ctx := rec.StartAgentExecute(context.Background())
		ctx = trace.WithHandler(ctx, rec)

		ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("say hello")})
		gt.NoError(t, err).Required()

		var texts []string
		var last *gollem.Response
		for resp := range ch {
			gt.NoError(t, resp.Error).Required()
			texts = append(texts, resp.Texts...)
			last = resp
		}
		rec.EndAgentExecute(ctx, nil)

		gt.Equal(t, []string{"Hel", "lo", " world"}, texts)
		gt.V(t, last).NotNil().Required()
		gt.Equal(t, 1000, last.InputToken)
		gt.Equal(t, 50, last.OutputToken)
		gt.Equal(t, 300, last.CacheReadInputToken)
		gt.Equal(t, 200, last.CacheCreationInputToken)

		var llmSpan *trace.Span
		for _, child := range rec.Trace().RootSpan.Children {
			if child.Kind == trace.SpanKindLLMCall {
				llmSpan = child
			}
		}
		gt.V(t, llmSpan).NotNil().Required()
		gt.Equal(t, 1000, llmSpan.LLMCall.InputTokens)
		gt.Equal(t, 50, llmSpan.LLMCall.OutputTokens)
		gt.Equal(t, 300, llmSpan.LLMCall.CacheReadInputTokens)
		gt.Equal(t, 200, llmSpan.LLMCall.CacheCreationInputTokens)
		gt.Equal(t, []string{"Hello world"}, llmSpan.LLMCall.Response.Texts)

		gt.True(t, rs.sent()[0].Stream)

		h, err := session.History()
		gt.NoError(t, err).Required()
		gt.A(t, h.Messages).Length(2).Required()
		gt.Equal(t, gollem.RoleAssistant, h.Messages[1].Role)
		text, err := h.Messages[1].Contents[0].GetTextContent()
		gt.NoError(t, err).Required()
		gt.Equal(t, "Hello world", text.Text)
	})

	t.Run("function call with reasoning", func(t *testing.T) {
		rs := newResponsesServer(t, replyEvents(
			`{"type":"response.reasoning_summary_text.delta","sequence_number":1,"output_index":0,"item_id":"rs_1","summary_index":0,"delta":"need to "}`,
			`{"type":"response.reasoning_summary_text.delta","sequence_number":2,"output_index":0,"item_id":"rs_1","summary_index":0,"delta":"look it up"}`,
			`{"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"enc-1","summary":[{"type":"summary_text","text":"need to look it up"}]}}`,
			`{"type":"response.function_call_arguments.delta","sequence_number":4,"output_index":1,"item_id":"fc_1","delta":"{\"key\":"}`,
			`{"type":"response.output_item.done","sequence_number":5,"output_index":1,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{\"key\":\"alpha\"}","status":"completed"}}`,
			`{"type":"response.completed","sequence_number":6,"response":{"id":"resp_1","status":"completed","model":"gpt-test","output":[],"usage":{"input_tokens":100,"output_tokens":10}}}`,
		))
		session, err := rs.client(t).NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
		gt.NoError(t, err).Required()

		ch, err := session.Stream(context.Background(), []gollem.Input{gollem.Text("what is alpha?")})
		gt.NoError(t, err).Required()

		var thoughts []string
		var calls []*gollem.FunctionCall
		for resp := range ch {
			gt.NoError(t, resp.Error).Required()
			thoughts = append(thoughts, resp.Thoughts...)
			calls = append(calls, resp.FunctionCalls...)
		}
		gt.Equal(t, []string{"need to ", "look it up"}, thoughts)
		gt.A(t, calls).Length(1).Required()
		gt.Equal(t, "call_1", calls[0].ID)
		gt.Equal(t, map[string]any{"key": "alpha"}, calls[0].Arguments)

		h, err := session.History()
		gt.NoError(t, err).Required()
		gt.A(t, h.Messages).Length(2).Required()
		gt.A(t, h.Messages[1].Contents).Length(2).Required()
		gt.Equal(t, gollem.MessageContentTypeThinking, h.Messages[1].Contents[0].Type)
		gt.Equal(t, gollem.MessageContentTypeToolCall, h.Messages[1].Contents[1].Type)
	})

	t.Run("failed response is delivered as an error", func(t *testing.T) {
		rs := newResponsesServer(t, replyEvents(
			`{"type":"response.failed","sequence_number":1,"response":{"id":"resp_1","status":"failed","output":[],"error":{"code":"server_error","message":"boom"}}}`,
		))
		session, err := rs.client(t).NewSession(context.Background())
		gt.NoError(t, err).Required()

		ch, err := session.Stream(context.Background(), []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()

		var errs []error
		for resp := range ch {
			if resp.Error != nil {
				errs = append(errs, resp.Error)
			}
		}
		gt.A(t, errs).Length(1)
	})

	t.Run("stream that ends before completion is an error", func(t *testing.T) {
		rs := newResponsesServer(t, replyEvents(
			`{"type":"response.output_text.delta","sequence_number":1,"output_index":0,"item_id":"msg_1","delta":"Hel"}`,
		))
		session, err := rs.client(t).NewSession(context.Background())
		gt.NoError(t, err).Required()

		ch, err := session.Stream(context.Background(), []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()

		var errs []error
		for resp := range ch {
			if resp.Error != nil {
				errs = append(errs, resp.Error)
			}
		}
		gt.A(t, errs).Length(1)
	})
}

func TestResponsesHistoryRestore(t *testing.T) {
	t.Run("history saved in Responses mode continues the conversation", func(t *testing.T) {
		rs := newResponsesServer(t, replySequence(functionCallReplyBody, finalAnswerReplyBody))
		client := rs.client(t)

		first, err := client.NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
		gt.NoError(t, err).Required()
		_, err = first.Generate(context.Background(), []gollem.Input{gollem.Text("what is alpha?")})
		gt.NoError(t, err).Required()

		saved, err := first.History()
		gt.NoError(t, err).Required()
		data, err := json.Marshal(saved)
		gt.NoError(t, err).Required()

		var restored gollem.History
		gt.NoError(t, json.Unmarshal(data, &restored)).Required()

		second, err := client.NewSession(context.Background(),
			gollem.WithSessionTools(&lookupTool{}),
			gollem.WithSessionHistory(&restored))
		gt.NoError(t, err).Required()

		resp, err := second.Generate(context.Background(), []gollem.Input{
			gollem.FunctionResponse{ID: "call_1", Name: "lookup", Data: map[string]any{"value": "ok"}},
		})
		gt.NoError(t, err).Required()
		gt.Equal(t, []string{"the value is ok"}, resp.Texts)

		input := rs.sent()[1].Input
		gt.A(t, input).Length(4).Required()
		gt.Equal(t, "reasoning", input[1].Type)
		gt.Equal(t, "rs_1", input[1].ID)
		gt.Equal(t, "enc-1", input[1].EncryptedContent)
		gt.Equal(t, "function_call", input[2].Type)
		gt.Equal(t, "call_1", input[2].CallID)
		gt.Equal(t, "function_call_output", input[3].Type)
		gt.Equal(t, "call_1", input[3].CallID)

		// The restored session keeps the new turn, so its History now holds
		// the second reasoning item too.
		h, err := second.History()
		gt.NoError(t, err).Required()
		gt.A(t, h.Messages).Length(4)
	})

	t.Run("reasoning items are sent only to a client of the same model", func(t *testing.T) {
		rs := newResponsesServer(t, replySequence(functionCallReplyBody, finalAnswerReplyBody))

		first, err := rs.client(t).NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
		gt.NoError(t, err).Required()
		_, err = first.Generate(context.Background(), []gollem.Input{gollem.Text("what is alpha?")})
		gt.NoError(t, err).Required()
		saved, err := first.History()
		gt.NoError(t, err).Required()

		// The reasoning item records the session's issuer.
		var reasoning *gollem.MessageContent
		for i := range saved.Messages {
			for j := range saved.Messages[i].Contents {
				if saved.Messages[i].Contents[j].Type == gollem.MessageContentTypeThinking {
					reasoning = &saved.Messages[i].Contents[j]
				}
			}
		}
		gt.V(t, reasoning).NotNil().Required()
		gt.V(t, reasoning.Provider).NotNil().Required()
		gt.Equal(t, testIssuer, reasoning.Provider.Issuer)

		second, err := rs.client(t, openai.WithModel("gpt-other")).NewSession(context.Background(),
			gollem.WithSessionTools(&lookupTool{}),
			gollem.WithSessionHistory(saved))
		gt.NoError(t, err).Required()
		_, err = second.Generate(context.Background(), []gollem.Input{
			gollem.FunctionResponse{ID: "call_1", Name: "lookup", Data: map[string]any{"value": "ok"}},
		})
		gt.NoError(t, err).Required()

		for _, item := range rs.sent()[1].Input {
			gt.NotEqual(t, "reasoning", item.Type)
		}
	})

	t.Run("history saved by the chat path loads", func(t *testing.T) {
		chatMessages := []openaiapi.ChatCompletionMessage{
			{Role: openaiapi.ChatMessageRoleUser, Content: "what is alpha?"},
			{
				Role:             openaiapi.ChatMessageRoleAssistant,
				ReasoningContent: "chat reasoning text",
				ToolCalls: []openaiapi.ToolCall{{
					ID:       "call_9",
					Type:     openaiapi.ToolTypeFunction,
					Function: openaiapi.FunctionCall{Name: "lookup", Arguments: `{"key":"alpha"}`},
				}},
			},
			{Role: openaiapi.ChatMessageRoleTool, ToolCallID: "call_9", Content: `{"value":"ok"}`},
			{Role: openaiapi.ChatMessageRoleAssistant, Content: "the value is ok"},
		}
		chatHistory, err := openai.NewHistory(chatMessages, testIssuer)
		gt.NoError(t, err).Required()
		data, err := json.Marshal(chatHistory)
		gt.NoError(t, err).Required()
		var restored gollem.History
		gt.NoError(t, json.Unmarshal(data, &restored)).Required()

		rs := newResponsesServer(t, replyJSON(textReplyBody))
		session, err := rs.client(t).NewSession(context.Background(),
			gollem.WithSessionTools(&lookupTool{}),
			gollem.WithSessionHistory(&restored))
		gt.NoError(t, err).Required()

		_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("thanks")})
		gt.NoError(t, err).Required()

		// Chat reasoning text has no Responses reasoning ID, so it is not sent.
		input := rs.sent()[0].Input
		gt.A(t, input).Length(5).Required()
		gt.Equal(t, "message", input[0].Type)
		gt.Equal(t, "what is alpha?", inputText(t, input[0]))
		gt.Equal(t, "function_call", input[1].Type)
		gt.Equal(t, "call_9", input[1].CallID)
		gt.Equal(t, `{"key":"alpha"}`, input[1].Arguments)
		gt.Equal(t, "function_call_output", input[2].Type)
		gt.Equal(t, "call_9", input[2].CallID)
		gt.Equal(t, `{"value":"ok"}`, input[2].Output)
		gt.Equal(t, "message", input[3].Type)
		gt.Equal(t, "assistant", input[3].Role)
		gt.Equal(t, "the value is ok", inputText(t, input[3]))
		gt.Equal(t, "thanks", inputText(t, input[4]))
	})

	t.Run("history saved in Responses mode loads in the chat path", func(t *testing.T) {
		rs := newResponsesServer(t, replyJSON(functionCallReplyBody))
		session, err := rs.client(t).NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
		gt.NoError(t, err).Required()
		_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("what is alpha?")})
		gt.NoError(t, err).Required()
		h, err := session.History()
		gt.NoError(t, err).Required()

		messages, err := openai.ToMessages(h, testIssuer)
		gt.NoError(t, err).Required()
		gt.A(t, messages).Length(2).Required()
		gt.A(t, messages[1].ToolCalls).Length(1).Required()
		gt.Equal(t, "call_1", messages[1].ToolCalls[0].ID)
		// The Responses API reasoning item is not Chat Completions reasoning.
		gt.Equal(t, "", messages[1].ReasoningContent)
	})
}

func TestResponsesMessagePhase(t *testing.T) {
	commentaryReplyBody := `{
		"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-test",
		"output": [
			{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "phase": "commentary",
			 "content": [{"type": "output_text", "text": "let me look it up", "annotations": []}]},
			{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup",
			 "arguments": "{\"key\":\"alpha\"}", "status": "completed"}
		],
		"usage": {"input_tokens": 100, "output_tokens": 10}
	}`
	rs := newResponsesServer(t, replySequence(commentaryReplyBody, finalAnswerReplyBody))
	session, err := rs.client(t).NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
	gt.NoError(t, err).Required()

	first, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("what is alpha?")})
	gt.NoError(t, err).Required()
	gt.Equal(t, []string{"let me look it up"}, first.Texts)

	_, err = session.Generate(context.Background(), []gollem.Input{
		gollem.FunctionResponse{ID: "call_1", Name: "lookup", Data: map[string]any{"value": "ok"}},
	})
	gt.NoError(t, err).Required()

	input := rs.sent()[1].Input
	gt.A(t, input).Length(4).Required()
	gt.Equal(t, "message", input[1].Type)
	gt.Equal(t, "assistant", input[1].Role)
	gt.Equal(t, "commentary", input[1].Phase)
	gt.Equal(t, "let me look it up", inputText(t, input[1]))
	// A user message carries no phase.
	gt.Equal(t, "", input[0].Phase)
}

func TestResponsesRefusal(t *testing.T) {
	t.Run("Generate", func(t *testing.T) {
		rs := newResponsesServer(t, replyJSON(`{
			"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-test",
			"output": [{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
				"content": [{"type": "refusal", "refusal": "I can't help with that."}]}],
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`))
		session, err := rs.client(t).NewSession(context.Background())
		gt.NoError(t, err).Required()

		rec := trace.New()
		ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
		resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		gt.Equal(t, []string{"I can't help with that."}, resp.Texts)
		gt.Equal(t, &gollem.Refusal{Reason: "refusal", Explanation: "I can't help with that."}, resp.Refusal)
		gt.Equal(t, &trace.Refusal{Reason: "refusal", Explanation: "I can't help with that."},
			findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response.Refusal)
	})

	t.Run("Generate reports content_filter", func(t *testing.T) {
		rs := newResponsesServer(t, replyJSON(`{
			"id": "resp_1", "object": "response", "status": "incomplete", "model": "gpt-test",
			"incomplete_details": {"reason": "content_filter"},
			"output": [],
			"usage": {"input_tokens": 10, "output_tokens": 0}
		}`))
		session, err := rs.client(t).NewSession(context.Background())
		gt.NoError(t, err).Required()

		resp, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		gt.Equal(t, "content_filter", resp.FinishReason)
		gt.Equal(t, &gollem.Refusal{Reason: "content_filter"}, resp.Refusal)
	})

	t.Run("Generate reports no refusal for text", func(t *testing.T) {
		rs := newResponsesServer(t, replyJSON(textReplyBody))
		session, err := rs.client(t).NewSession(context.Background())
		gt.NoError(t, err).Required()

		resp, err := session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		gt.Nil(t, resp.Refusal)
	})

	t.Run("Stream", func(t *testing.T) {
		rs := newResponsesServer(t, replyEvents(
			`{"type":"response.refusal.delta","sequence_number":1,"output_index":0,"item_id":"msg_1","delta":"I can't "}`,
			`{"type":"response.refusal.delta","sequence_number":2,"output_index":0,"item_id":"msg_1","delta":"help with that."}`,
			`{"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"refusal","refusal":"I can't help with that."}]}}`,
			`{"type":"response.completed","sequence_number":4,"response":{"id":"resp_1","status":"completed","model":"gpt-test","output":[],"usage":{"input_tokens":10,"output_tokens":5}}}`,
		))
		session, err := rs.client(t).NewSession(context.Background())
		gt.NoError(t, err).Required()

		rec := trace.New()
		ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), rec)
		ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("hi")})
		gt.NoError(t, err).Required()
		var texts []string
		var refusals []*gollem.Refusal
		for resp := range ch {
			gt.NoError(t, resp.Error).Required()
			texts = append(texts, resp.Texts...)
			if resp.Refusal != nil {
				refusals = append(refusals, resp.Refusal)
			}
		}
		gt.Equal(t, []string{"I can't ", "help with that."}, texts)
		gt.Equal(t, []*gollem.Refusal{{Reason: "refusal", Explanation: "I can't help with that."}}, refusals)
		gt.Equal(t, &trace.Refusal{Reason: "refusal", Explanation: "I can't help with that."},
			findLLMCallSpan(t, rec.Trace().RootSpan).LLMCall.Response.Refusal)

		h, err := session.History()
		gt.NoError(t, err).Required()
		gt.A(t, h.Messages).Length(2).Required()
		text, err := h.Messages[1].Contents[0].GetTextContent()
		gt.NoError(t, err).Required()
		gt.Equal(t, "I can't help with that.", text.Text)
	})
}

// llmCallEndSignal reports when an LLM call span ends.
type llmCallEndSignal struct {
	trace.Handler
	ended chan error
}

func (h *llmCallEndSignal) EndLLMCall(ctx context.Context, data *trace.LLMCallData, err error) {
	h.Handler.EndLLMCall(ctx, data, err)
	h.ended <- err
}

// TestResponsesStreamCancel verifies that a caller that cancels the context and
// stops reading the channel does not leave the stream goroutine blocked: the
// LLM call span still ends.
func TestResponsesStreamCancel(t *testing.T) {
	release := make(chan struct{})
	rs := newResponsesServer(t, func(t *testing.T, w http.ResponseWriter, _ sentResponsesRequest, _ int) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeEvent(t, w, `{"type":"response.output_text.delta","sequence_number":1,"output_index":0,"item_id":"msg_1","delta":"Hel"}`)
		writeEvent(t, w, `{"type":"response.output_text.delta","sequence_number":2,"output_index":0,"item_id":"msg_1","delta":"lo"}`)
		<-release
	})
	// Registered after the server, so it runs before the server is closed and
	// lets the blocked handler return.
	t.Cleanup(func() { close(release) })

	session, err := rs.client(t).NewSession(context.Background())
	gt.NoError(t, err).Required()

	rec := trace.New()
	signal := &llmCallEndSignal{Handler: rec, ended: make(chan error, 1)}
	ctx := trace.WithHandler(rec.StartAgentExecute(context.Background()), signal)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("hi")})
	gt.NoError(t, err).Required()

	first := <-ch
	gt.NoError(t, first.Error).Required()
	gt.Equal(t, []string{"Hel"}, first.Texts)

	// Stop reading and cancel.
	cancel()

	select {
	case err := <-signal.ended:
		gt.True(t, errors.Is(err, context.Canceled))
	case <-time.After(5 * time.Second):
		t.Fatal("the LLM call span did not end after the context was cancelled")
	}
}

func TestResponsesInvalidFunctionArguments(t *testing.T) {
	rs := newResponsesServer(t, replyJSON(`{
		"id": "resp_1", "object": "response", "status": "completed", "model": "gpt-test",
		"output": [{"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup",
			"arguments": "{\"key\":", "status": "completed"}],
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`))
	session, err := rs.client(t).NewSession(context.Background(), gollem.WithSessionTools(&lookupTool{}))
	gt.NoError(t, err).Required()

	_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("hi")})
	gt.Error(t, err)
}

func TestResponsesAppendHistory(t *testing.T) {
	question, err := gollem.NewTextContent("earlier question")
	gt.NoError(t, err).Required()
	answer, err := gollem.NewTextContent("earlier answer")
	gt.NoError(t, err).Required()
	earlier := &gollem.History{
		LLType:  gollem.LLMTypeOpenAI,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{question}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{answer}},
		},
	}

	rs := newResponsesServer(t, replyJSON(textReplyBody))
	session, err := rs.client(t).NewSession(context.Background())
	gt.NoError(t, err).Required()
	gt.NoError(t, session.AppendHistory(earlier)).Required()

	_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("next question")})
	gt.NoError(t, err).Required()

	input := rs.sent()[0].Input
	gt.A(t, input).Length(3).Required()
	gt.Equal(t, "earlier question", inputText(t, input[0]))
	gt.Equal(t, "assistant", input[1].Role)
	gt.Equal(t, "earlier answer", inputText(t, input[1]))
	gt.Equal(t, "next question", inputText(t, input[2]))
}

func TestResponsesMiddlewareRewritesHistory(t *testing.T) {
	rs := newResponsesServer(t, replySequence(textReplyBody, textReplyBody))
	client := rs.client(t)

	replaced, err := gollem.NewTextContent("replaced history")
	gt.NoError(t, err).Required()

	var seenHistory int
	middleware := func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
		return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
			seenHistory = req.History.ToCount()
			if req.History != nil {
				req.History.Messages = []gollem.Message{
					{Role: gollem.RoleUser, Contents: []gollem.MessageContent{replaced}},
				}
			}
			req.SystemPrompt = "rewritten prompt"
			return next(ctx, req)
		}
	}

	session, err := client.NewSession(context.Background(), gollem.WithSessionContentBlockMiddleware(middleware))
	gt.NoError(t, err).Required()

	_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("first")})
	gt.NoError(t, err).Required()
	gt.Equal(t, 0, seenHistory)

	_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("second")})
	gt.NoError(t, err).Required()
	gt.Equal(t, 2, seenHistory)

	req := rs.sent()[1]
	gt.Equal(t, "rewritten prompt", req.Instructions)
	gt.A(t, req.Input).Length(2).Required()
	gt.Equal(t, "replaced history", inputText(t, req.Input[0]))
	gt.Equal(t, "second", inputText(t, req.Input[1]))
}

func TestResponsesParallelSessions(t *testing.T) {
	// Each reply echoes the last user text so that a request mixed up between
	// sessions shows in the answer.
	rs := newResponsesServer(t, func(t *testing.T, w http.ResponseWriter, req sentResponsesRequest, _ int) {
		var parts []struct {
			Text string `json:"text"`
		}
		if err := json.Unmarshal(req.Input[len(req.Input)-1].Content, &parts); err != nil || len(parts) != 1 {
			t.Errorf("unexpected last input item: %s", req.Input[len(req.Input)-1].Content)
			http.Error(w, "unexpected input", http.StatusBadRequest)
			return
		}
		body := map[string]any{
			"id": "resp", "object": "response", "status": "completed", "model": "gpt-test",
			"output": []any{map[string]any{
				"type": "message", "id": "msg", "role": "assistant", "status": "completed",
				"content": []any{map[string]any{"type": "output_text", "text": "echo " + parts[0].Text, "annotations": []any{}}},
			}},
			"usage": map[string]any{"input_tokens": 10, "output_tokens": 2},
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(body); err != nil {
			t.Errorf("failed to write reply: %v", err)
		}
	})
	client := rs.client(t)

	const turns = 5
	var wg sync.WaitGroup
	for _, name := range []string{"a", "b"} {
		wg.Add(1)
		go func(name string) {
			defer wg.Done()
			session, err := client.NewSession(context.Background())
			if err != nil {
				t.Error(err)
				return
			}
			for i := 0; i < turns; i++ {
				text := fmt.Sprintf("%s-%d", name, i)
				resp, err := session.Generate(context.Background(), []gollem.Input{gollem.Text(text)})
				if err != nil {
					t.Error(err)
					return
				}
				if len(resp.Texts) != 1 || resp.Texts[0] != "echo "+text {
					t.Errorf("session %s turn %d: unexpected texts %v", name, i, resp.Texts)
				}
			}
			h, err := session.History()
			if err != nil {
				t.Error(err)
				return
			}
			if len(h.Messages) != turns*2 {
				t.Errorf("session %s: history has %d messages, want %d", name, len(h.Messages), turns*2)
			}
		}(name)
	}
	wg.Wait()
	gt.A(t, rs.sent()).Length(turns * 2)
}

// TestResponsesCountToken is gated like the live tests because tiktoken downloads
// its encoding from an OpenAI host on first use.
func TestResponsesCountToken(t *testing.T) {
	if _, ok := os.LookupEnv("TEST_OPENAI_API_KEY"); !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}
	rs := newResponsesServer(t, replyJSON(textReplyBody))
	session, err := rs.client(t).NewSession(context.Background(), gollem.WithSessionSystemPrompt("be brief"))
	gt.NoError(t, err).Required()

	count, err := session.CountToken(context.Background(), gollem.Text("hello world"))
	gt.NoError(t, err).Required()
	gt.True(t, count > 0)

	// Counting does not add the input to the history.
	h, err := session.History()
	gt.NoError(t, err).Required()
	gt.A(t, h.Messages).Length(0)
	gt.A(t, rs.sent()).Length(0)
}

// TestResponsesToolCallLive runs a two-turn tool call against the real API:
// the model calls a tool, reads its result and answers.
func TestResponsesToolCallLive(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}

	runTest := func(model string) func(t *testing.T) {
		return func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()

			client, err := openai.New(ctx, apiKey,
				openai.WithModel(model),
				openai.WithResponsesAPI(),
				openai.WithReasoningEffort("low"),
			)
			gt.NoError(t, err).Required()

			session, err := client.NewSession(ctx,
				gollem.WithSessionTools(&lookupTool{}),
				gollem.WithSessionSystemPrompt("Use the lookup tool to answer. Reply with the value only."),
			)
			gt.NoError(t, err).Required()

			first, err := session.Generate(ctx, []gollem.Input{gollem.Text("What is the value for key 'alpha'?")})
			gt.NoError(t, err).Required()
			gt.A(t, first.FunctionCalls).Longer(0).Required()
			t.Logf("turn 1: input=%d cache_read=%d cache_write=%d output=%d",
				first.InputToken, first.CacheReadInputToken, first.CacheCreationInputToken, first.OutputToken)
			gt.True(t, first.InputToken >= first.CacheReadInputToken+first.CacheCreationInputToken)

			var results []gollem.Input
			for _, fc := range first.FunctionCalls {
				gt.Equal(t, "lookup", fc.Name)
				results = append(results, gollem.FunctionResponse{
					ID: fc.ID, Name: fc.Name, Data: map[string]any{"value": "ok"},
				})
			}

			second, err := session.Generate(ctx, results)
			gt.NoError(t, err).Required()
			gt.A(t, second.Texts).Longer(0).Required()
			t.Logf("turn 2: %q input=%d cache_read=%d cache_write=%d output=%d",
				strings.Join(second.Texts, ""), second.InputToken,
				second.CacheReadInputToken, second.CacheCreationInputToken, second.OutputToken)
			gt.True(t, second.InputToken > 0)
			gt.True(t, second.InputToken >= second.CacheReadInputToken+second.CacheCreationInputToken)
			gt.S(t, strings.ToLower(strings.Join(second.Texts, ""))).Contains("ok")
		}
	}

	t.Run("gpt-6.1-sol", runTest("gpt-6.1-sol"))
	t.Run("gpt-6-luna", runTest("gpt-6-luna"))
}
