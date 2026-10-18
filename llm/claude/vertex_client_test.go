package claude_test

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/gt"
)

func TestNewWithVertex(t *testing.T) {
	ctx := context.Background()

	t.Run("missing projectID", func(t *testing.T) {
		client, err := claude.NewWithVertex(ctx, "us-central1", "")
		gt.Error(t, err)
		gt.Nil(t, client)
		gt.True(t, strings.Contains(err.Error(), "projectID is required"))
	})

	t.Run("missing region", func(t *testing.T) {
		client, err := claude.NewWithVertex(ctx, "", "test-project")
		gt.Error(t, err)
		gt.Nil(t, client)
		gt.True(t, strings.Contains(err.Error(), "region is required"))
	})

	t.Run("valid parameters with options", func(t *testing.T) {
		prj, ok := os.LookupEnv("TEST_CLAUDE_VERTEX_AI_PROJECT_ID")
		if !ok {
			t.Skip("TEST_CLAUDE_VERTEX_AI_PROJECT_ID is not set")
		}
		client, err := claude.NewWithVertex(ctx, "us-central1", prj,
			claude.WithVertexModel("claude-sonnet-4@20250514"),
			claude.WithVertexTemperature(0.5),
			claude.WithVertexTopP(0.8),
			claude.WithVertexMaxTokens(2048),
			claude.WithVertexSystemPrompt("You are a helpful assistant"),
		)

		// Note: This will likely fail in CI/testing without proper GCP credentials
		// but we're mainly testing the parameter validation and setup
		if err != nil {
			// Expected in test environment without GCP credentials
			gt.True(t, strings.Contains(err.Error(), "failed to") || strings.Contains(err.Error(), "auth"))
			return
		}

		// If it succeeds, validate the configuration
		gt.NotNil(t, client)
		gt.Equal(t, int64(2048), claude.VertexMaxTokensOf(client))
	})

	t.Run("resolves max tokens from the model when not set", func(t *testing.T) {
		prj, ok := os.LookupEnv("TEST_CLAUDE_VERTEX_AI_PROJECT_ID")
		if !ok {
			t.Skip("TEST_CLAUDE_VERTEX_AI_PROJECT_ID is not set")
		}
		client, err := claude.NewWithVertex(ctx, "us-central1", prj,
			claude.WithVertexModel("claude-opus-4-5@20251101"),
		)
		if err != nil {
			// Expected in test environment without GCP credentials
			gt.True(t, strings.Contains(err.Error(), "failed to") || strings.Contains(err.Error(), "auth"))
			return
		}

		gt.Equal(t, int64(64000), claude.VertexMaxTokensOf(client))
	})
}

func TestVertexClient(t *testing.T) {
	projectID := os.Getenv("TEST_CLAUDE_VERTEX_AI_PROJECT_ID")
	if projectID == "" {
		t.Skip("TEST_CLAUDE_VERTEX_AI_PROJECT_ID not set, skipping test")
	}

	location := os.Getenv("TEST_CLAUDE_VERTEX_AI_LOCATION")
	if location == "" {
		location = "us-east5" // Default to us-east5 where Claude Sonnet 4 is working
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	// Create Vertex AI client using Anthropic's official SDK
	client, err := claude.NewWithVertex(ctx, location, projectID,
		claude.WithVertexModel("claude-sonnet-4@20250514"),
		claude.WithVertexMaxTokens(512),
		claude.WithVertexTemperature(0.5),
	)
	gt.NoError(t, err)

	// Create session
	session, err := client.NewSession(ctx)
	gt.NoError(t, err)

	// Test basic text generation
	response, err := session.Generate(ctx, []gollem.Input{gollem.Text("Hello! Please respond with 'Vertex AI working!' to confirm this integration works.")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err)
	gt.NotNil(t, response)
	gt.True(t, len(response.Texts) > 0)

	gt.True(t, strings.Contains(response.Texts[0], "Vertex AI working!"))
}

func TestVertexClientWithTools(t *testing.T) {
	projectID := os.Getenv("TEST_CLAUDE_VERTEX_AI_PROJECT_ID")
	if projectID == "" {
		t.Skip("TEST_CLAUDE_VERTEX_AI_PROJECT_ID not set, skipping test")
	}

	location := os.Getenv("TEST_CLAUDE_VERTEX_AI_LOCATION")
	if location == "" {
		location = "us-east5" // Default to us-east5 where Claude Sonnet 4 is working
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	// Create Vertex AI client using Anthropic's official SDK
	client, err := claude.NewWithVertex(ctx, location, projectID,
		claude.WithVertexModel("claude-sonnet-4@20250514"),
		claude.WithVertexMaxTokens(512),
		claude.WithVertexTemperature(0.5),
	)
	gt.NoError(t, err)

	// Create simple test tool
	testTool := &calculatorTool{}

	// Create session with tool
	session, err := client.NewSession(ctx, gollem.WithSessionTools(testTool))
	gt.NoError(t, err)

	// Test tool calling
	response, err := session.Generate(ctx, []gollem.Input{gollem.Text("Please calculate 15 + 27 using the calculator tool.")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err)
	gt.NotNil(t, response)

	// Should either return text or function calls
	gt.True(t, len(response.Texts) > 0 || len(response.FunctionCalls) > 0)

	if len(response.FunctionCalls) > 0 {

		// Execute the function call
		result, err := testTool.Run(ctx, response.FunctionCalls[0].Arguments)
		gt.NoError(t, err)

		// Send result back
		funcResp := gollem.FunctionResponse{
			ID:   response.FunctionCalls[0].ID,
			Name: response.FunctionCalls[0].Name,
			Data: result,
		}

		finalResponse, err := session.Generate(ctx, []gollem.Input{funcResp}, gollem.WithMaxTokens(maxTestTokens))
		gt.NoError(t, err)
		gt.NotNil(t, finalResponse)
		gt.True(t, len(finalResponse.Texts) > 0)

		gt.True(t, strings.Contains(finalResponse.Texts[0], "42"))
	}
}

// calculatorTool is a simple tool for testing
type calculatorTool struct{}

func (c *calculatorTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "calculator",
		Description: "Perform basic arithmetic operations",
		Parameters: map[string]*gollem.Parameter{
			"operation": {
				Type:        gollem.TypeString,
				Description: "The operation to perform (add, subtract, multiply, divide)",
				Required:    true,
			},
			"a": {
				Type:        gollem.TypeNumber,
				Description: "First number",
				Required:    true,
			},
			"b": {
				Type:        gollem.TypeNumber,
				Description: "Second number",
				Required:    true,
			},
		},
	}
}

func (c *calculatorTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	operation, ok := args["operation"].(string)
	if !ok {
		return nil, gollem.ErrInvalidParameter
	}

	a, ok := args["a"].(float64)
	if !ok {
		return nil, gollem.ErrInvalidParameter
	}

	b, ok := args["b"].(float64)
	if !ok {
		return nil, gollem.ErrInvalidParameter
	}

	var result float64
	switch operation {
	case "add":
		result = a + b
	case "subtract":
		result = a - b
	case "multiply":
		result = a * b
	case "divide":
		if b == 0 {
			return map[string]any{"error": "division by zero"}, nil
		}
		result = a / b
	default:
		return nil, gollem.ErrInvalidParameter
	}

	return map[string]any{"result": result}, nil
}

var _ gollem.ModelNamer = (*claude.VertexClient)(nil)

// TestVertexClientModel verifies that the client reports the model name it was
// configured with, without consulting the API. The client is built through
// NewVertexClientWithOptions, which runs the same defaults and option handling
// as NewWithVertex but stops before the GCP credential lookup that
// NewWithVertex performs.
func TestVertexClientModel(t *testing.T) {
	type testCase struct {
		options  []claude.VertexOption
		expected string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			client := claude.NewVertexClientWithOptions(tc.options...)
			gt.Equal(t, tc.expected, client.Model())
		}
	}

	t.Run("configured model", runTest(testCase{
		options:  []claude.VertexOption{claude.WithVertexModel("claude-opus-4-5@20251101")},
		expected: "claude-opus-4-5@20251101",
	}))

	t.Run("default model when no option is given", runTest(testCase{
		expected: claude.DefaultVertexClaudeModel,
	}))

	t.Run("last option wins", runTest(testCase{
		options: []claude.VertexOption{
			claude.WithVertexModel("claude-sonnet-4@20250514"),
			claude.WithVertexModel("claude-opus-4-5@20251101"),
		},
		expected: "claude-opus-4-5@20251101",
	}))

	t.Run("empty model is reported as configured", runTest(testCase{
		options:  []claude.VertexOption{claude.WithVertexModel("")},
		expected: "",
	}))
}

func newScopedVertexSession(t *testing.T, baseURL, model, scope string, opts ...gollem.SessionOption) gollem.Session {
	t.Helper()
	anthropicClient := anthropic.NewClient(
		option.WithAPIKey("test-key"),
		option.WithBaseURL(baseURL),
		option.WithMaxRetries(0),
	)
	client := claude.NewVertexClientWithAnthropicClient(&anthropicClient,
		claude.WithVertexModel(model), claude.WithVertexIssuerScope(scope))
	session, err := client.NewSession(context.Background(), opts...)
	gt.NoError(t, err)
	return session
}

const sseMessageStart = `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"usage":{"input_tokens":1,"output_tokens":0}}}`

var sseMessageEnd = [][2]string{
	{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`},
	{"message_stop", `{"type":"message_stop"}`},
}

// collectStream drains a stream and returns the thoughts, texts and the first error.
func collectStream(t *testing.T, ch <-chan *gollem.Response) (thoughts, texts []string, calls []*gollem.FunctionCall, err error) {
	t.Helper()
	for resp := range ch {
		if resp.Error != nil && err == nil {
			err = resp.Error
		}
		thoughts = append(thoughts, resp.Thoughts...)
		texts = append(texts, resp.Texts...)
		calls = append(calls, resp.FunctionCalls...)
	}
	return thoughts, texts, calls, err
}

func TestVertexStreamRecordsThinking(t *testing.T) {
	const model = "claude-test"
	issuer := gollem.Issuer{Provider: gollem.LLMTypeClaude, Model: model, Scope: "v"}
	question := []gollem.Input{gollem.Text("question")}

	t.Run("thinking, text and tool use are kept in order", func(t *testing.T) {
		events := append([][2]string{
			{"message_start", sseMessageStart},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"pl"}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"an"}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":1}`},
			{"content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"call_1","name":"search","input":{}}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"q\":\"x\"}"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":2}`},
		}, sseMessageEnd...)
		ss := newScriptedServer(t, "", events)
		session := newScopedVertexSession(t, ss.srv.URL, model, "v")

		ch, err := session.Stream(context.Background(), question)
		gt.NoError(t, err)
		thoughts, texts, calls, streamErr := collectStream(t, ch)
		gt.NoError(t, streamErr)
		gt.Equal(t, []string{"pl", "an"}, thoughts)
		gt.Equal(t, []string{"answer"}, texts)
		gt.A(t, calls).Length(1)

		h, err := session.History()
		gt.NoError(t, err)
		contents := assistantContents(t, h)
		gt.A(t, contents).Length(3).Required()
		gt.Equal(t, gollem.MessageContentTypeThinking, contents[0].Type)
		gt.Equal(t, gollem.MessageContentTypeText, contents[1].Type)
		gt.Equal(t, gollem.MessageContentTypeToolCall, contents[2].Type)

		thinking, err := contents[0].GetThinkingContent()
		gt.NoError(t, err)
		gt.Equal(t, "plan", thinking.Text)
		gt.NotNil(t, contents[0].Provider)
		gt.Equal(t, issuer, contents[0].Provider.Issuer)
		gt.Equal(t, `{"signature":"sig"}`, string(contents[0].Provider.Data))
	})

	t.Run("redacted thinking is kept and adds no Thoughts", func(t *testing.T) {
		events := append([][2]string{
			{"message_start", sseMessageStart},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		}, sseMessageEnd...)
		ss := newScriptedServer(t, "", events)
		session := newScopedVertexSession(t, ss.srv.URL, model, "v")

		ch, err := session.Stream(context.Background(), question)
		gt.NoError(t, err)
		thoughts, _, _, streamErr := collectStream(t, ch)
		gt.NoError(t, streamErr)
		gt.A(t, thoughts).Length(0)

		h, err := session.History()
		gt.NoError(t, err)
		contents := assistantContents(t, h)
		gt.A(t, contents).Length(2).Required()
		gt.Equal(t, gollem.MessageContentTypeThinking, contents[0].Type)
		gt.NotNil(t, contents[0].Provider)
		gt.Equal(t, `{"signature":"opaque","redacted":true}`, string(contents[0].Provider.Data))
	})

	t.Run("blocks are sent back in the order and with the text received", func(t *testing.T) {
		events := append([][2]string{
			{"message_start", sseMessageStart},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Here is {\"a\":1}"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":0}`},
			{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":"","signature":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"sig"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":1}`},
			{"content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"B"}}`},
			{"content_block_stop", `{"type":"content_block_stop","index":2}`},
		}, sseMessageEnd...)
		ss := newScriptedServer(t, messageJSON(`{"type":"text","text":"ok"}`), events)
		// A JSON content type must not rewrite the recorded text either.
		session := newScopedVertexSession(t, ss.srv.URL, model, "v", gollem.WithSessionContentType(gollem.ContentTypeJSON))

		ch, err := session.Stream(context.Background(), question)
		gt.NoError(t, err)
		_, _, _, streamErr := collectStream(t, ch)
		gt.NoError(t, streamErr)

		_, err = session.Generate(context.Background(), []gollem.Input{gollem.Text("follow-up")})
		gt.NoError(t, err)

		messages := ss.lastMessages(t)
		gt.Equal(t, [][]string{{"text"}, {"text", "thinking", "text"}, {"text"}}, blockTypes(messages))
		blocks := messages[1]["content"].([]any)
		gt.Equal(t, `Here is {"a":1}`, blocks[0].(map[string]any)["text"])
		gt.Equal(t, "sig", blocks[1].(map[string]any)["signature"])
		gt.Equal(t, "B", blocks[2].(map[string]any)["text"])
	})

	t.Run("a stream ending in an error leaves the history unchanged", func(t *testing.T) {
		events := [][2]string{
			{"message_start", sseMessageStart},
			{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`},
			{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}`},
			{"error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`},
		}
		ss := newScriptedServer(t, "", events)
		session := newScopedVertexSession(t, ss.srv.URL, model, "v")

		ch, err := session.Stream(context.Background(), question)
		gt.NoError(t, err)
		_, _, _, streamErr := collectStream(t, ch)
		gt.Error(t, streamErr)

		h, err := session.History()
		gt.NoError(t, err)
		gt.A(t, h.Messages).Length(0)
	})
}

func TestVertexIssuerScopeSeparatesThinking(t *testing.T) {
	const model = "claude-test"
	response := messageJSON(`{"type":"thinking","thinking":"plan","signature":"sig"},{"type":"text","text":"answer"}`)

	type testCase struct {
		nextScope string
		expected  [][]string
	}
	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			ss := newScriptedServer(t, response, nil)
			first := newScopedVertexSession(t, ss.srv.URL, model, "a")
			_, err := first.Generate(context.Background(), []gollem.Input{gollem.Text("question")})
			gt.NoError(t, err)
			h, err := first.History()
			gt.NoError(t, err)

			next := newScopedVertexSession(t, ss.srv.URL, model, tc.nextScope, gollem.WithSessionHistory(h))
			_, err = next.Generate(context.Background(), []gollem.Input{gollem.Text("follow-up")})
			gt.NoError(t, err)
			gt.Equal(t, tc.expected, blockTypes(ss.lastMessages(t)))
		}
	}

	t.Run("same scope sends the thinking block", runTest(testCase{
		nextScope: "a",
		expected:  [][]string{{"text"}, {"thinking", "text"}, {"text"}},
	}))
	t.Run("different scope drops the thinking block", runTest(testCase{
		nextScope: "b",
		expected:  [][]string{{"text"}, {"text"}, {"text"}},
	}))
}

func TestVertexGenerateRefusalWithoutContent(t *testing.T) {
	ss := newScriptedServer(t, `{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],`+
		`"stop_reason":"refusal","stop_sequence":null,`+
		`"stop_details":{"type":"refusal","category":"frontier_llm","explanation":"explained"},`+
		`"usage":{"input_tokens":12,"output_tokens":0,"cache_creation_input_tokens":3,"cache_read_input_tokens":5}}`, nil)
	session := newScopedVertexSession(t, ss.srv.URL, "claude-test", "")

	ctx, rec := newTraceContext()
	resp, err := session.Generate(ctx, []gollem.Input{gollem.Text("question")},
		gollem.WithGenerateResponseSchema(structuredOutputTestSchema()))
	gt.NoError(t, err).Required()
	gt.Equal(t, "refusal", resp.FinishReason)
	gt.A(t, resp.Texts).Length(0)
	gt.Equal(t, 20, resp.InputToken)
	gt.Equal(t, 0, resp.OutputToken)
	gt.Equal(t, 3, resp.CacheCreationInputToken)
	gt.Equal(t, 5, resp.CacheReadInputToken)
	gt.Equal(t, &gollem.Refusal{
		Reason:      "refusal",
		Categories:  []string{"frontier_llm"},
		Explanation: "explained",
	}, resp.Refusal)

	traced := llmCallResponse(t, rec)
	gt.Equal(t, "refusal", traced.FinishReason)
	gt.A(t, traced.Texts).Length(0)
	gt.Equal(t, &trace.Refusal{
		Reason:      "refusal",
		Categories:  []string{"frontier_llm"},
		Explanation: "explained",
	}, traced.Refusal)
}

func TestVertexStreamRefusalWithoutContent(t *testing.T) {
	ss := newScriptedServer(t, "", [][2]string{
		{"message_start", sseMessageStart},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"refusal",` +
			`"stop_details":{"type":"refusal","category":"reasoning_extraction","explanation":null}},` +
			`"usage":{"output_tokens":0}}`},
		{"message_stop", `{"type":"message_stop"}`},
	})
	session := newScopedVertexSession(t, ss.srv.URL, "claude-test", "")

	ctx, rec := newTraceContext()
	ch, err := session.Stream(ctx, []gollem.Input{gollem.Text("question")})
	gt.NoError(t, err).Required()
	var reasons []string
	var refusals []*gollem.Refusal
	for resp := range ch {
		gt.NoError(t, resp.Error)
		if resp.FinishReason != "" {
			reasons = append(reasons, resp.FinishReason)
		}
		if resp.Refusal != nil {
			refusals = append(refusals, resp.Refusal)
		}
	}
	gt.Equal(t, []string{"refusal"}, reasons)
	expected := &gollem.Refusal{Reason: "refusal", Categories: []string{"reasoning_extraction"}}
	gt.Equal(t, []*gollem.Refusal{expected}, refusals)

	traced := llmCallResponse(t, rec)
	gt.Equal(t, "refusal", traced.FinishReason)
	gt.A(t, traced.Texts).Length(0)
	gt.Equal(t, &trace.Refusal{Reason: "refusal", Categories: []string{"reasoning_extraction"}}, traced.Refusal)
}

func TestWithVertexEffort(t *testing.T) {
	type testCase struct {
		effort   claude.Effort
		stream   bool
		expected string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			rs := newRecordingServer(t, "ok")
			anthropicClient := anthropic.NewClient(
				option.WithAPIKey("test-key"),
				option.WithBaseURL(rs.srv.URL),
				option.WithMaxRetries(0),
			)
			options := []claude.VertexOption{claude.WithVertexModel("claude-opus-5-5")}
			if tc.effort != "" {
				options = append(options, claude.WithVertexEffort(tc.effort))
			}
			client := claude.NewVertexClientWithAnthropicClient(&anthropicClient, options...)
			session, err := client.NewSession(context.Background())
			gt.NoError(t, err).Required()

			requestPath{stream: tc.stream}.send(t, session)

			outputConfig, ok := rs.lastBody(t)["output_config"]
			if tc.expected == "" {
				gt.False(t, ok)
				return
			}
			gt.True(t, ok).Required()
			assertJSONEqual(t, tc.expected, outputConfig)
		}
	}

	t.Run("low", runTest(testCase{effort: claude.EffortLow, expected: `{"effort":"low"}`}))
	t.Run("medium", runTest(testCase{effort: claude.EffortMedium, expected: `{"effort":"medium"}`}))
	t.Run("high", runTest(testCase{effort: claude.EffortHigh, expected: `{"effort":"high"}`}))
	t.Run("xhigh", runTest(testCase{effort: claude.EffortXHigh, expected: `{"effort":"xhigh"}`}))
	t.Run("max", runTest(testCase{effort: claude.EffortMax, expected: `{"effort":"max"}`}))
	t.Run("stream", runTest(testCase{effort: claude.EffortLow, stream: true, expected: `{"effort":"low"}`}))
	t.Run("not set", runTest(testCase{}))
}

func TestWithVertexAdaptiveThinking(t *testing.T) {
	type testCase struct {
		options  []claude.VertexOption
		expected string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			for _, stream := range []bool{false, true} {
				rs := newRecordingServer(t, "ok")
				anthropicClient := anthropic.NewClient(
					option.WithAPIKey("test-key"),
					option.WithBaseURL(rs.srv.URL),
					option.WithMaxRetries(0),
				)
				options := append([]claude.VertexOption{claude.WithVertexModel("claude-sonnet-5-5")}, tc.options...)
				client := claude.NewVertexClientWithAnthropicClient(&anthropicClient, options...)
				session, err := client.NewSession(context.Background())
				gt.NoError(t, err).Required()

				requestPath{stream: stream}.send(t, session)

				thinking, ok := rs.lastBody(t)["thinking"]
				if tc.expected == "" {
					gt.False(t, ok)
					continue
				}
				gt.True(t, ok).Required()
				assertJSONEqual(t, tc.expected, thinking)
			}
		}
	}

	t.Run("not set", runTest(testCase{}))
	t.Run("summarized", runTest(testCase{
		options:  []claude.VertexOption{claude.WithVertexAdaptiveThinking(claude.ThinkingDisplaySummarized)},
		expected: `{"type":"adaptive","display":"summarized"}`,
	}))
	t.Run("omitted", runTest(testCase{
		options:  []claude.VertexOption{claude.WithVertexAdaptiveThinking(claude.ThinkingDisplayOmitted)},
		expected: `{"type":"adaptive","display":"omitted"}`,
	}))
}

// instructionLiveModels are the models the instruction message is checked
// against. Both run the prefix check on thinking blocks, so a request that
// edited an earlier turn would be rejected with a 400.
var instructionLiveModels = []string{"claude-sonnet-5-5", "claude-haiku-5-5"}

// instructionLiveTimeout covers several calls with adaptive thinking.
const instructionLiveTimeout = 5 * time.Minute

// newInstructionLiveSession returns a Vertex AI session that sends response
// schemas as instruction messages, with adaptive thinking on so that thinking
// blocks are carried through the history.
func newInstructionLiveSession(ctx context.Context, t *testing.T, model string, opts ...gollem.SessionOption) gollem.Session {
	t.Helper()
	projectID, ok := os.LookupEnv("TEST_CLAUDE_VERTEX_AI_PROJECT_ID")
	if !ok {
		t.Skip("TEST_CLAUDE_VERTEX_AI_PROJECT_ID is not set")
	}
	client, err := claude.NewWithVertex(ctx, "global", projectID,
		claude.WithVertexModel(model),
		claude.WithVertexMaxTokens(8192),
		claude.WithVertexStructuredOutputsDisabled(),
		claude.WithVertexEffort(claude.EffortHigh),
		claude.WithVertexAdaptiveThinking(claude.ThinkingDisplaySummarized))
	gt.NoError(t, err).Required()
	session, err := client.NewSession(ctx, opts...)
	gt.NoError(t, err).Required()
	return session
}

func weatherSchema() *gollem.Parameter {
	return &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"city":          {Type: gollem.TypeString, Required: true},
			"temperature_c": {Type: gollem.TypeInteger, Required: true},
		},
	}
}

func summarySchema() *gollem.Parameter {
	return &gollem.Parameter{
		Type: gollem.TypeObject,
		Properties: map[string]*gollem.Parameter{
			"summary":    {Type: gollem.TypeString, Required: true},
			"word_count": {Type: gollem.TypeInteger, Required: true},
		},
	}
}

// generateText calls Generate and returns the joined response text.
func generateText(ctx context.Context, t *testing.T, session gollem.Session, input []gollem.Input, opts ...gollem.GenerateOption) (*gollem.Response, string) {
	t.Helper()
	resp, err := session.Generate(ctx, input, opts...)
	gt.NoError(t, err).Required()
	text := strings.Join(resp.Texts, "")
	t.Logf("reply: %s", text)
	return resp, text
}

// assertJSONReply checks that text is a JSON object that conforms to schema
// and returns the object.
func assertJSONReply(t *testing.T, text string, schema *gollem.Parameter) map[string]any {
	t.Helper()
	var reply map[string]any
	gt.NoError(t, json.Unmarshal([]byte(text), &reply)).Required()
	gt.NoError(t, schema.ValidateValue("reply", reply))
	return reply
}

// assertTextReply checks that text is a non-empty reply that is not valid
// JSON, which is what a call without a schema asks for.
func assertTextReply(t *testing.T, text string) {
	t.Helper()
	gt.True(t, strings.TrimSpace(text) != "")
	gt.False(t, json.Valid([]byte(text)))
}

// requireSignedThinking fails the test unless the history holds a thinking
// block with a signature. Adaptive thinking may skip thinking, and a run
// without one would not show that later calls keep thinking blocks valid.
func requireSignedThinking(t *testing.T, session gollem.Session) {
	t.Helper()
	history, err := session.History()
	gt.NoError(t, err).Required()
	for _, msg := range history.Messages {
		for _, content := range msg.Contents {
			if content.Type == gollem.MessageContentTypeThinking && content.Provider != nil && len(content.Provider.Data) > 0 {
				return
			}
		}
	}
	t.Fatal("history holds no signed thinking block")
}

// countInstructionMessages counts the user messages in the history that hold
// only an instruction for a JSON response.
func countInstructionMessages(t *testing.T, session gollem.Session) int {
	t.Helper()
	history, err := session.History()
	gt.NoError(t, err).Required()
	count := 0
	for _, msg := range history.Messages {
		if msg.Role != gollem.RoleUser || len(msg.Contents) != 1 || msg.Contents[0].Type != gollem.MessageContentTypeText {
			continue
		}
		text, err := msg.Contents[0].GetTextContent()
		gt.NoError(t, err).Required()
		if strings.HasPrefix(text.Text, "Format only your next response as valid JSON.") {
			count++
		}
	}
	return count
}

func TestAlternatingSchemaAndTextLive(t *testing.T) {
	for _, model := range instructionLiveModels {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), instructionLiveTimeout)
			defer cancel()
			session := newInstructionLiveSession(ctx, t, model)

			_, text := generateText(ctx, t, session,
				[]gollem.Input{gollem.Text("Tokyo's average August high is about 31 °C and the low about 24 °C. " +
					"Reason step by step about a realistic temperature at 3 pm on a typical August day, then give it as an example.")},
				gollem.WithGenerateResponseSchema(weatherSchema()))
			assertJSONReply(t, text, weatherSchema())
			requireSignedThinking(t, session)

			_, text = generateText(ctx, t, session,
				[]gollem.Input{gollem.Text("Describe that weather in one plain English sentence. Do not use JSON.")})
			assertTextReply(t, text)

			_, text = generateText(ctx, t, session,
				[]gollem.Input{gollem.Text("Summarize your previous sentence.")},
				gollem.WithGenerateResponseSchema(summarySchema()))
			reply := assertJSONReply(t, text, summarySchema())
			_, hasCity := reply["city"]
			gt.False(t, hasCity)

			_, text = generateText(ctx, t, session,
				[]gollem.Input{gollem.Text("Say thank you in plain text.")})
			assertTextReply(t, text)

			gt.Equal(t, 2, countInstructionMessages(t, session))
		})
	}
}

// weatherTool is a get_weather tool for the live tool loop. Its result is
// returned by the test itself, so Run is never called.
type weatherTool struct{}

func (weatherTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "get_weather",
		Description: "Returns the current weather of a city.",
		Parameters: map[string]*gollem.Parameter{
			"city": {Type: gollem.TypeString, Description: "City name", Required: true},
		},
	}
}

func (weatherTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	return map[string]any{"condition": "sunny", "temperature_c": 31}, nil
}

func TestToolLoopThenSchemaLive(t *testing.T) {
	for _, model := range instructionLiveModels {
		t.Run(model, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), instructionLiveTimeout)
			defer cancel()
			session := newInstructionLiveSession(ctx, t, model, gollem.WithSessionTools(weatherTool{}))

			resp, _ := generateText(ctx, t, session,
				[]gollem.Input{gollem.Text("Use get_weather to check the weather in Tokyo.")})
			gt.A(t, resp.FunctionCalls).Longer(0).Required()
			requireSignedThinking(t, session)

			var results []gollem.Input
			for _, call := range resp.FunctionCalls {
				gt.Equal(t, "get_weather", call.Name)
				results = append(results, gollem.FunctionResponse{
					ID:   call.ID,
					Name: call.Name,
					Data: map[string]any{"city": "Tokyo", "condition": "sunny", "temperature_c": 31},
				})
			}
			_, text := generateText(ctx, t, session, results,
				gollem.WithGenerateResponseSchema(weatherSchema()), gollem.WithToolCallsDisabled())
			assertJSONReply(t, text, weatherSchema())

			_, text = generateText(ctx, t, session,
				[]gollem.Input{gollem.Text("Explain that result in one plain English sentence. Do not use JSON.")},
				gollem.WithToolCallsDisabled())
			assertTextReply(t, text)
		})
	}
}
