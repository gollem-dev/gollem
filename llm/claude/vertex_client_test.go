package claude_test

import (
	"context"
	"os"
	"strings"
	"testing"

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
