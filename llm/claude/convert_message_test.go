package claude_test

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/internal/historytest"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/m-mizutani/gt"
)

// marshalMessages renders messages the way the Anthropic SDK sends them, so tests compare
// what the API receives instead of the Go representation that produced it.
func marshalMessages(t *testing.T, messages []anthropic.MessageParam) []string {
	t.Helper()
	out := make([]string, 0, len(messages))
	for _, msg := range messages {
		data, err := msg.MarshalJSON()
		gt.NoError(t, err)
		out = append(out, string(data))
	}
	return out
}

func TestClaudeMessageRoundTrip(t *testing.T) {
	type testCase struct {
		name     string
		messages []anthropic.MessageParam
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			// Convert Claude messages to gollem.History
			history, err := claude.NewHistory(tc.messages, testIssuer)
			gt.NoError(t, err)

			// Convert back to Claude messages
			restored, err := claude.ToMessages(history, testIssuer)
			gt.NoError(t, err)

			// Compare the wire form rather than the Go values. A tool_use input is carried
			// as json.RawMessage so that the SDK encoder emits the arguments verbatim, so
			// two message lists that produce the same request can differ as Go structs.
			gt.Equal(t, marshalMessages(t, tc.messages), marshalMessages(t, restored))
		}
	}

	t.Run("text messages", runTest(testCase{
		name: "text messages",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Hello")),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("Hi, how can I help you?")),
		},
	}))

	t.Run("tool use and results", runTest(testCase{
		name: "tool use and results",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("What's the weather?")),
			anthropic.NewAssistantMessage(
				anthropic.NewToolUseBlock(
					"toolu_abc123",
					map[string]interface{}{"location": "Tokyo"},
					"get_weather",
				),
			),
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(
					"toolu_abc123",
					// JSON keys are alphabetically sorted after round-trip
					`{"condition":"sunny","temperature":25}`,
					false,
				),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("The weather in Tokyo is sunny with a temperature of 25°C.")),
		},
	}))

	t.Run("multiple content blocks", runTest(testCase{
		name: "multiple content blocks",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Tell me a joke and the weather")),
			anthropic.NewAssistantMessage(
				anthropic.NewTextBlock("Here's a joke: Why did the chicken cross the road? Let me check the weather..."),
				anthropic.NewToolUseBlock(
					"toolu_xyz789",
					map[string]interface{}{"location": "London"},
					"get_weather",
				),
			),
			anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(
					"toolu_xyz789",
					// JSON keys are alphabetically sorted after round-trip
					`{"condition":"rainy","temperature":15}`,
					false,
				),
			),
		},
	}))

	t.Run("PDF document block", runTest(testCase{
		name: "PDF document block",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(
				anthropic.NewTextBlock("Analyze this PDF"),
				anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
					Data: base64.StdEncoding.EncodeToString([]byte("%PDF-1.4 test")),
				}),
			),
			anthropic.NewAssistantMessage(anthropic.NewTextBlock("This PDF contains test data.")),
		},
	}))

	t.Run("thinking and redacted thinking blocks", runTest(testCase{
		name: "thinking and redacted thinking blocks",
		messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Help me")),
			anthropic.NewAssistantMessage(
				anthropic.NewThinkingBlock("sig-123", "Let me think..."),
				anthropic.NewRedactedThinkingBlock("Redacted"),
				anthropic.NewTextBlock("Done."),
			),
		},
	}))
}

var testIssuer = gollem.Issuer{Provider: gollem.LLMTypeClaude, Model: "claude-test"}

func TestNewHistoryRecordsIssuerOnThinking(t *testing.T) {
	issuer := gollem.Issuer{Provider: gollem.LLMTypeClaude, Model: "claude-test", Scope: "tenant-a"}
	history, err := claude.NewHistory([]anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock("Help me")),
		anthropic.NewAssistantMessage(
			anthropic.NewThinkingBlock("sig-123", "Let me think..."),
			anthropic.NewRedactedThinkingBlock("Redacted"),
			anthropic.NewTextBlock("Done."),
		),
	}, issuer)
	gt.NoError(t, err)

	contents := history.Messages[1].Contents
	gt.A(t, contents).Length(3).Required()

	thinking, err := contents[0].GetThinkingContent()
	gt.NoError(t, err)
	gt.Equal(t, "Let me think...", thinking.Text)
	gt.NotNil(t, contents[0].Provider)
	gt.Equal(t, issuer, contents[0].Provider.Issuer)
	gt.Equal(t, `{"signature":"sig-123"}`, string(contents[0].Provider.Data))

	redacted, err := contents[1].GetThinkingContent()
	gt.NoError(t, err)
	gt.Equal(t, "", redacted.Text)
	gt.NotNil(t, contents[1].Provider)
	gt.Equal(t, issuer, contents[1].Provider.Issuer)
	gt.Equal(t, `{"signature":"Redacted","redacted":true}`, string(contents[1].Provider.Data))

	// Text carries no provider-bound data.
	gt.Nil(t, contents[2].Provider)
}

func TestToMessagesFiltersThinkingByIssuer(t *testing.T) {
	original := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock("Help me")),
		anthropic.NewAssistantMessage(
			anthropic.NewThinkingBlock("sig-123", "Let me think..."),
			anthropic.NewTextBlock("Calling the tool."),
			anthropic.NewToolUseBlock("call_1", map[string]any{"q": "x"}, "search"),
		),
	}
	history, err := claude.NewHistory(original, testIssuer)
	gt.NoError(t, err)

	t.Run("same issuer sends the signed thinking block", func(t *testing.T) {
		restored, err := claude.ToMessages(history, testIssuer)
		gt.NoError(t, err)
		gt.Equal(t, marshalMessages(t, original), marshalMessages(t, restored))
	})

	t.Run("another model drops the thinking block only", func(t *testing.T) {
		restored, err := claude.ToMessages(history, gollem.Issuer{Provider: gollem.LLMTypeClaude, Model: "claude-other"})
		gt.NoError(t, err)
		gt.A(t, restored).Length(2).Required()
		gt.A(t, restored[1].Content).Length(2).Required()
		gt.NotNil(t, restored[1].Content[0].OfText)
		gt.NotNil(t, restored[1].Content[1].OfToolUse)
	})
}

func TestToMessagesDropsGeminiThinking(t *testing.T) {
	gemini := gollem.Issuer{Provider: gollem.LLMTypeGemini, Model: "gemini-test"}
	thinking, err := gollem.NewThinkingContent("")
	gt.NoError(t, err)
	thinking.Provider = &gollem.ProviderData{Issuer: gemini, Data: json.RawMessage(`{"thought_signature":"c2ln"}`)}
	call, err := gollem.NewToolCallContent("call_1", "search", map[string]any{"q": "x"})
	gt.NoError(t, err)
	call.Provider = &gollem.ProviderData{Issuer: gemini, Data: json.RawMessage(`{"thought_signature":"c2ln"}`)}
	user, err := gollem.NewTextContent("go")
	gt.NoError(t, err)

	history := &gollem.History{
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{user}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{thinking, call}},
		},
	}

	messages, err := claude.ToMessages(history, testIssuer)
	gt.NoError(t, err)
	gt.A(t, messages).Length(2).Required()
	gt.A(t, messages[1].Content).Length(1).Required()
	gt.NotNil(t, messages[1].Content[0].OfToolUse)
}

func TestToMessagesThinkingProviderDataEdgeCases(t *testing.T) {
	newHistory := func(t *testing.T, data string) *gollem.History {
		thinking, err := gollem.NewThinkingContent("plan")
		gt.NoError(t, err)
		thinking.Provider = &gollem.ProviderData{Issuer: testIssuer}
		if data != "" {
			thinking.Provider.Data = json.RawMessage(data)
		}
		text, err := gollem.NewTextContent("answer")
		gt.NoError(t, err)
		return &gollem.History{
			Version: gollem.HistoryVersion,
			Messages: []gollem.Message{
				{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{thinking, text}},
			},
		}
	}

	t.Run("broken provider data returns an error", func(t *testing.T) {
		_, err := claude.ToMessages(newHistory(t, `{`), testIssuer)
		gt.Error(t, err)
	})

	t.Run("unsigned thinking from the same issuer is not sent", func(t *testing.T) {
		messages, err := claude.ToMessages(newHistory(t, ""), testIssuer)
		gt.NoError(t, err)
		gt.A(t, messages).Length(1).Required()
		gt.A(t, messages[0].Content).Length(1).Required()
		gt.NotNil(t, messages[0].Content[0].OfText)
	})
}

func TestToMessagesLeavesHistoryUnchanged(t *testing.T) {
	sys, err := gollem.NewTextContent("be brief")
	gt.NoError(t, err)
	user, err := gollem.NewTextContent("hello")
	gt.NoError(t, err)
	assistant, err := gollem.NewTextContent("hi")
	gt.NoError(t, err)
	resp1, err := gollem.NewToolResponseContent("c1", "alpha", map[string]any{"ok": true}, false)
	gt.NoError(t, err)
	resp2, err := gollem.NewToolResponseContent("c2", "beta", map[string]any{"ok": true}, false)
	gt.NoError(t, err)

	history := &gollem.History{
		LLType:  gollem.LLMTypeClaude,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			{Role: gollem.RoleSystem, Contents: []gollem.MessageContent{sys}},
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{user}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{assistant}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp1}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp2}},
		},
	}
	before := append([]gollem.Message(nil), history.Messages...)

	first, err := claude.ToMessages(history, testIssuer)
	gt.NoError(t, err)

	// The same History is converted again on every later request, so a conversion must not
	// consume the system message or duplicate the trailing message.
	gt.Equal(t, before, history.Messages)

	second, err := claude.ToMessages(history, testIssuer)
	gt.NoError(t, err)
	gt.Equal(t, first, second)
}

func TestToolResponsesInSeparateMessagesBecomeOneMessage(t *testing.T) {
	text, err := gollem.NewTextContent("go")
	gt.NoError(t, err)
	call1, err := gollem.NewToolCallContent("c1", "alpha", map[string]any{"x": 1})
	gt.NoError(t, err)
	call2, err := gollem.NewToolCallContent("c2", "beta", map[string]any{"y": 2})
	gt.NoError(t, err)
	resp1, err := gollem.NewToolResponseContent("c1", "alpha", map[string]any{"ok": true}, false)
	gt.NoError(t, err)
	resp2, err := gollem.NewToolResponseContent("c2", "beta", map[string]any{"ok": true}, false)
	gt.NoError(t, err)

	history := &gollem.History{
		LLType:  gollem.LLMTypeClaude,
		Version: gollem.HistoryVersion,
		Messages: []gollem.Message{
			{Role: gollem.RoleUser, Contents: []gollem.MessageContent{text}},
			{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{call1, call2}},
			// A runtime that runs the calls one at a time appends one message per result.
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp1}},
			{Role: gollem.RoleTool, Contents: []gollem.MessageContent{resp2}},
		},
	}

	messages, err := claude.ToMessages(history, testIssuer)
	gt.NoError(t, err)

	// Claude requires one tool_result for each tool_use block, all in the next user message.
	gt.Equal(t, 3, len(messages))
	gt.Equal(t, anthropic.MessageParamRoleUser, messages[2].Role)
	gt.Equal(t, 2, len(messages[2].Content))
	gt.NotNil(t, messages[2].Content[0].OfToolResult)
	gt.Value(t, messages[2].Content[0].OfToolResult.ToolUseID).Equal("c1")
	gt.NotNil(t, messages[2].Content[1].OfToolResult)
	gt.Value(t, messages[2].Content[1].OfToolResult.ToolUseID).Equal("c2")
}

// A Claude tool_result block carries no tool name. Gemini requires one on every
// functionResponse part, so the name is recovered from the tool_use block with the same ID.
func TestNewHistoryRecoversToolNameForToolResults(t *testing.T) {
	messages := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock("what is the weather")),
		anthropic.NewAssistantMessage(
			anthropic.NewToolUseBlock("call_1", map[string]any{"city": "Tokyo"}, "get_weather"),
		),
		anthropic.NewUserMessage(
			anthropic.NewToolResultBlock("call_1", `{"temperature":25}`, false),
		),
	}

	history, err := claude.NewHistory(messages, testIssuer)
	gt.NoError(t, err)

	resp, err := history.Messages[2].Contents[0].GetToolResponseContent()
	gt.NoError(t, err)
	gt.Equal(t, "call_1", resp.ToolCallID)
	gt.Equal(t, "get_weather", resp.Name)
}

// A tool_result for a tool_use that is not in the message list has no name to recover.
// The response must still convert rather than fail.
func TestNewHistoryToolResultWithoutMatchingToolUse(t *testing.T) {
	messages := []anthropic.MessageParam{
		anthropic.NewUserMessage(
			anthropic.NewToolResultBlock("orphan", `{"ok":true}`, false),
		),
	}

	history, err := claude.NewHistory(messages, testIssuer)
	gt.NoError(t, err)

	resp, err := history.Messages[0].Contents[0].GetToolResponseContent()
	gt.NoError(t, err)
	gt.Equal(t, "orphan", resp.ToolCallID)
	gt.Equal(t, "", resp.Name)
}

// Tool arguments and results wider than float64 must reach the Anthropic request with the
// value the model sent. The Anthropic SDK marshals tool_use input through its own encoder,
// so this exercises the whole path rather than encoding/json alone.
func TestClaudeHistoryPreservesWideIntegers(t *testing.T) {
	const wide = "9007199254740993"

	messages := []anthropic.MessageParam{
		anthropic.NewAssistantMessage(
			anthropic.NewToolUseBlock("call_1", map[string]any{"id": json.Number(wide)}, "lookup"),
		),
		anthropic.NewUserMessage(
			anthropic.NewToolResultBlock("call_1", `{"account":9007199254740993}`, false),
		),
	}

	history, err := claude.NewHistory(messages, testIssuer)
	gt.NoError(t, err)

	call, err := history.Messages[0].Contents[0].GetToolCallContent()
	gt.NoError(t, err)
	encodedArgs, err := json.Marshal(call.Arguments)
	gt.NoError(t, err)
	gt.Equal(t, `{"id":`+wide+`}`, string(encodedArgs))

	resp, err := history.Messages[1].Contents[0].GetToolResponseContent()
	gt.NoError(t, err)
	encodedResp, err := json.Marshal(resp.Response)
	gt.NoError(t, err)
	gt.Equal(t, `{"account":`+wide+`}`, string(encodedResp))

	// Back into Claude's own request types, then out through the SDK's encoder.
	// anthropic.MessageParam has its own MarshalJSON, so this is the wire form the
	// Anthropic API actually receives, not encoding/json's view of it.
	restored, err := claude.ToMessages(history, testIssuer)
	gt.NoError(t, err)
	wire, err := restored[0].MarshalJSON()
	gt.NoError(t, err)
	gt.S(t, string(wire)).Contains(`"id":` + wide)
}

// Every site that builds a tool_use block must encode the arguments the same way. Handing
// the SDK a map instead makes its encoder write a json.Number as a quoted string, so a
// wide integer argument would leave the streaming path as a string while the history path
// sent a number.
func TestToolUseInputSendsNumbersAsNumbers(t *testing.T) {
	const wide = "9007199254740993"

	input, err := claude.ToolUseInput(map[string]any{"id": json.Number(wide)})
	gt.NoError(t, err)

	msg := anthropic.NewAssistantMessage(anthropic.NewToolUseBlock("call_1", input, "lookup"))
	wire, err := msg.MarshalJSON()
	gt.NoError(t, err)
	gt.S(t, string(wire)).Contains(`"id":` + wide)
	gt.S(t, string(wire)).NotContains(`"id":"` + wide)
}

// toolUseInputOf returns the input of the tool_use block at messages[0].Content[0]
// as the bytes that will be sent.
func toolUseInputOf(t *testing.T, messages []anthropic.MessageParam) string {
	t.Helper()
	gt.A(t, messages).Longer(0).Required()
	gt.A(t, messages[0].Content).Longer(0).Required()
	toolUse := messages[0].Content[0].OfToolUse
	if toolUse == nil {
		t.Fatal("first block is not tool_use")
	}
	input, ok := toolUse.Input.(json.RawMessage)
	gt.True(t, ok).Required()
	return string(input)
}

func TestNewHistoryRecordsToolUseInput(t *testing.T) {
	issuer := gollem.Issuer{Provider: gollem.LLMTypeClaude, Model: "claude-test", Scope: "tenant-a"}

	type testCase struct {
		input    any
		recorded bool
		expected string
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			history, err := claude.NewHistory([]anthropic.MessageParam{
				anthropic.NewAssistantMessage(anthropic.NewToolUseBlock("call_1", tc.input, "search")),
			}, issuer)
			gt.NoError(t, err).Required()

			content := history.Messages[0].Contents[0]
			if !tc.recorded {
				gt.Nil(t, content.Provider)
			} else {
				if content.Provider == nil {
					t.Fatal("tool call has no provider data")
				}
				gt.Equal(t, issuer, content.Provider.Issuer)
			}

			restored, err := claude.ToMessages(history, issuer)
			gt.NoError(t, err).Required()
			gt.Equal(t, tc.expected, toolUseInputOf(t, restored))
		}
	}

	t.Run("raw JSON input is recorded and sent back", runTest(testCase{
		input:    json.RawMessage(`{"zeta":1,"alpha":"x"}`),
		recorded: true,
		expected: `{"zeta":1,"alpha":"x"}`,
	}))
	t.Run("string JSON input is recorded and sent back", runTest(testCase{
		input:    `{"zeta":1,"alpha":"x"}`,
		recorded: true,
		expected: `{"zeta":1,"alpha":"x"}`,
	}))
	t.Run("integer wider than float64 is recorded and sent back", runTest(testCase{
		input:    json.RawMessage(`{"zeta":9007199254740993,"alpha":1}`),
		recorded: true,
		expected: `{"zeta":9007199254740993,"alpha":1}`,
	}))
	t.Run("number literals that re-encoding rewrites are recorded and sent back", runTest(testCase{
		input:    json.RawMessage(`{"zeta":1e20,"beta":2.50,"alpha":1}`),
		recorded: true,
		expected: `{"zeta":1e20,"beta":2.50,"alpha":1}`,
	}))
	t.Run("map input built by the caller is not recorded", runTest(testCase{
		input:    map[string]any{"zeta": 1, "alpha": "x"},
		recorded: false,
		expected: `{"alpha":"x","zeta":1}`,
	}))
	t.Run("null input is not recorded", runTest(testCase{
		input:    json.RawMessage(`null`),
		recorded: false,
		expected: `null`,
	}))
}

func TestToolCallInputWithoutProviderData(t *testing.T) {
	// A tool call saved before the arguments JSON was recorded has no provider data.
	call, err := gollem.NewToolCallContent("call_1", "search", map[string]any{"zeta": 1, "alpha": "x"})
	gt.NoError(t, err).Required()
	history := &gollem.History{
		Version:  gollem.HistoryVersion,
		Messages: []gollem.Message{{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{call}}},
	}

	messages, err := claude.ToMessages(history, testIssuer)
	gt.NoError(t, err).Required()
	gt.Equal(t, `{"alpha":"x","zeta":1}`, toolUseInputOf(t, messages))
}

func TestToMessagesRejectsBrokenToolUseInput(t *testing.T) {
	runTest := func(data string) func(t *testing.T) {
		return func(t *testing.T) {
			call, err := gollem.NewToolCallContent("call_1", "search", map[string]any{"q": "x"})
			gt.NoError(t, err).Required()
			call.Provider = &gollem.ProviderData{Issuer: testIssuer, Data: json.RawMessage(data)}
			history := &gollem.History{
				Version:  gollem.HistoryVersion,
				Messages: []gollem.Message{{Role: gollem.RoleAssistant, Contents: []gollem.MessageContent{call}}},
			}

			_, err = claude.ToMessages(history, testIssuer)
			gt.Error(t, err)
		}
	}

	t.Run("recorded input is not JSON", runTest(`{"input":"not json"}`))
	t.Run("recorded input is not an object", runTest(`{"input":"[1]"}`))
	t.Run("recorded input is null", runTest(`{"input":"null"}`))
	t.Run("provider data is not JSON", runTest(`{`))
}

// The trace handler reads tool_use arguments back out of the request blocks. Those blocks
// now carry json.RawMessage, so a type assertion for a map alone would trace every
// history-replayed tool call with no arguments at all.
func TestTraceMessagesReadRawToolUseInput(t *testing.T) {
	input, err := claude.ToolUseInput(map[string]any{"city": "Tokyo"})
	gt.NoError(t, err)

	messages := []anthropic.MessageParam{
		anthropic.NewAssistantMessage(anthropic.NewToolUseBlock("call_1", input, "get_weather")),
	}

	traced := claude.ClaudeMessagesToTraceMessages(messages)
	gt.A(t, traced).Length(1)
	gt.A(t, traced[0].Contents).Length(1)
	gt.Equal(t, "get_weather", traced[0].Contents[0].Name)
	gt.Equal(t, "Tokyo", gt.Cast[string](t, traced[0].Contents[0].Arguments["city"]))
}

// Cross-provider conversion tests. Each one is split at the gollem.History shared
// through internal/historytest; the test with the same name in the other provider
// package converts that History. See the historytest package documentation.

// History → Claude. llm/openai converts OpenAI messages to the History.
func TestOpenAIToClaudeConversion(t *testing.T) {
	runTest := func(expected []anthropic.MessageParam) func(t *testing.T) {
		return func(t *testing.T) {
			messages, err := claude.ToMessages(historytest.Load(t, "openai_to_claude", "openai"), testIssuer)
			gt.NoError(t, err)

			// Compare the wire form rather than the Go values. A tool_use input is carried
			// as json.RawMessage so that the SDK encoder emits the arguments verbatim, so
			// two message lists that produce the same request can differ as Go structs.
			gt.Equal(t, marshalMessages(t, expected), marshalMessages(t, messages))
		}
	}

	t.Run("text messages with all fields", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock("Hello")),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("Hi there!")),
	}))

	t.Run("tool calls with all fields", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock("What's the weather in Tokyo and London?")),
		anthropic.NewAssistantMessage(
			anthropic.NewToolUseBlock("call_123", map[string]interface{}{
				"location": "Tokyo",
				"unit":     "celsius",
			}, "get_weather"),
			anthropic.NewToolUseBlock("call_456", map[string]interface{}{
				"location": "London",
				"unit":     "celsius",
			}, "get_weather"),
		),
		// Claude requires every tool_result answering one assistant turn to be in a
		// single user message, so the two OpenAI tool messages become one.
		anthropic.NewUserMessage(
			anthropic.NewToolResultBlock("call_123", `{"condition":"sunny","humidity":60,"temperature":25}`, false),
			anthropic.NewToolResultBlock("call_456", `{"condition":"rainy","humidity":80,"temperature":15}`, false),
		),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("Tokyo is sunny at 25°C. London is rainy at 15°C.")),
	}))

	t.Run("multi-content with images", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(
			anthropic.NewTextBlock("What's in this image?"),
			anthropic.NewImageBlockBase64("image/png", "iVBORw0KGgo="),
		),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("I see a cat in the image.")),
	}))

	t.Run("system message", runTest([]anthropic.MessageParam{
		// Claude merges system message into first user message with "\n\n" separator
		anthropic.NewUserMessage(
			anthropic.NewTextBlock("You are a helpful assistant.\n\n"),
			anthropic.NewTextBlock("Hello"),
		),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("Hi! How can I help you?")),
	}))

	t.Run("PDF content", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(
			anthropic.NewTextBlock("Analyze this PDF"),
			anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
				Data: "JVBERi0xLjQgdGVzdA==",
			}),
		),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("This PDF contains test data.")),
	}))
}

// Claude → History. llm/gemini converts the History to Gemini contents.
func TestClaudeToGeminiConversion(t *testing.T) {
	runTest := func(messages []anthropic.MessageParam) func(t *testing.T) {
		return func(t *testing.T) {
			history, err := claude.NewHistory(messages, testIssuer)
			gt.NoError(t, err).Required()
			historytest.Equal(t, "claude_to_gemini", "claude", history)
		}
	}

	t.Run("text messages", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock("Hello, how are you?")),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("I'm doing well, thank you!")),
	}))

	t.Run("tool use with multiple calls", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock("Calculate 5+3 and 10*2")),
		anthropic.NewAssistantMessage(
			anthropic.NewToolUseBlock("toolu_123", map[string]interface{}{"expression": "5+3"}, "calculate"),
			anthropic.NewToolUseBlock("toolu_456", map[string]interface{}{"expression": "10*2"}, "calculate"),
		),
		anthropic.NewUserMessage(
			anthropic.NewToolResultBlock("toolu_123", `{"result":8}`, false),
			anthropic.NewToolResultBlock("toolu_456", `{"result":20}`, false),
		),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("5+3 equals 8, and 10*2 equals 20.")),
	}))

	t.Run("mixed content blocks", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock("Tell me a joke and check the time")),
		anthropic.NewAssistantMessage(
			anthropic.NewTextBlock("Here's a joke: Why did the chicken cross the road?"),
			anthropic.NewToolUseBlock("toolu_789", map[string]interface{}{}, "get_current_time"),
		),
		anthropic.NewUserMessage(
			anthropic.NewToolResultBlock("toolu_789", `{"time":"14:30:00","timezone":"UTC"}`, false),
		),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("It's currently 14:30 UTC.")),
	}))

	t.Run("image content", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(
			anthropic.NewTextBlock("Analyze this image"),
			anthropic.NewImageBlockBase64("image/jpeg", "/9j/4AAQSkZJRg=="),
		),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("This appears to be a landscape photo.")),
	}))

	t.Run("error tool result", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock("Get the weather")),
		anthropic.NewAssistantMessage(
			anthropic.NewToolUseBlock("toolu_error", map[string]interface{}{"location": "InvalidCity"}, "get_weather"),
		),
		anthropic.NewUserMessage(
			anthropic.NewToolResultBlock("toolu_error", `{"error":"City not found"}`, true),
		),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("I couldn't find that city.")),
	}))

	t.Run("PDF content", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(
			anthropic.NewTextBlock("Analyze this PDF"),
			anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
				Data: base64.StdEncoding.EncodeToString([]byte("%PDF-1.4 test")),
			}),
		),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("This PDF contains test data.")),
	}))
}

// Claude → History → Gemini → History → Claude must restore the original messages.
// This test covers both ends; llm/gemini converts the first History through Gemini
// contents into the second.
func TestClaudeRoundTrip(t *testing.T) {
	runTest := func(messages []anthropic.MessageParam) func(t *testing.T) {
		return func(t *testing.T) {
			history, err := claude.NewHistory(messages, testIssuer)
			gt.NoError(t, err).Required()
			historytest.Equal(t, "claude_round_trip", "claude", history)

			restored, err := claude.ToMessages(historytest.Load(t, "claude_round_trip", "gemini"), testIssuer)
			gt.NoError(t, err)
			gt.Equal(t, messages, restored)
		}
	}

	t.Run("text messages", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock("Hello")),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("Hi!")),
	}))

	t.Run("PDF content", runTest([]anthropic.MessageParam{
		anthropic.NewUserMessage(
			anthropic.NewTextBlock("Analyze this PDF"),
			anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
				Data: base64.StdEncoding.EncodeToString([]byte("%PDF-1.4 test")),
			}),
		),
		anthropic.NewAssistantMessage(anthropic.NewTextBlock("This PDF contains test data.")),
	}))

	// Note: Tool IDs cannot be perfectly round-tripped because Gemini
	// regenerates tool call IDs. Text-only messages can round-trip perfectly.
}

// The Claude leg of OpenAI → History → Claude → History → OpenAI. llm/openai covers
// both ends and compares the restored messages with the original.
func TestOpenAIRoundTrip(t *testing.T) {
	run := func(t *testing.T) {
		messages, err := claude.ToMessages(historytest.Load(t, "openai_round_trip", "openai"), testIssuer)
		gt.NoError(t, err).Required()

		history, err := claude.NewHistory(messages, testIssuer)
		gt.NoError(t, err).Required()
		historytest.Equal(t, "openai_round_trip", "claude", history)
	}

	t.Run("text messages", run)
	t.Run("tool calls", run)
	t.Run("PDF content", run)
}
