package claude_test

import (
	"encoding/base64"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/m-mizutani/gt"
)

func TestClaudeMessageRoundTrip(t *testing.T) {
	type testCase struct {
		name     string
		messages []anthropic.MessageParam
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			// Convert Claude messages to gollem.History
			history, err := claude.NewHistory(tc.messages)
			gt.NoError(t, err)

			// Convert back to Claude messages
			restored, err := claude.ToMessages(history)
			gt.NoError(t, err)

			// Compare messages
			gt.Equal(t, tc.messages, restored)
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

	t.Run("thinking block", func(t *testing.T) {
		// Test thinking content conversion (Claude → gollem)
		block := anthropic.NewThinkingBlock("sig-123", "Let me think...")

		history, err := claude.NewHistory([]anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Help me")),
			anthropic.NewAssistantMessage(block),
		})
		gt.NoError(t, err)

		// Find assistant message with thinking content
		var assistantMsg *gollem.Message
		for i := range history.Messages {
			if history.Messages[i].Role == gollem.RoleAssistant {
				assistantMsg = &history.Messages[i]
				break
			}
		}

		gt.NotNil(t, assistantMsg)
		gt.Equal(t, 1, len(assistantMsg.Contents))

		content := assistantMsg.Contents[0]
		gt.Equal(t, gollem.MessageContentTypeThinking, content.Type)

		reasoning, err := content.GetThinkingContent()
		gt.NoError(t, err)
		gt.Equal(t, "Let me think...", reasoning.Text)
	})

	t.Run("redacted thinking block", func(t *testing.T) {
		// Test redacted thinking content conversion
		block := anthropic.NewRedactedThinkingBlock("Redacted")

		history, err := claude.NewHistory([]anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock("Help me")),
			anthropic.NewAssistantMessage(block),
		})
		gt.NoError(t, err)

		// Find assistant message with redacted thinking content
		var assistantMsg *gollem.Message
		for i := range history.Messages {
			if history.Messages[i].Role == gollem.RoleAssistant {
				assistantMsg = &history.Messages[i]
				break
			}
		}

		gt.NotNil(t, assistantMsg)
		gt.Equal(t, 1, len(assistantMsg.Contents))

		content := assistantMsg.Contents[0]
		gt.Equal(t, gollem.MessageContentTypeThinking, content.Type)

		reasoning, err := content.GetThinkingContent()
		gt.NoError(t, err)
		gt.Equal(t, "", reasoning.Text)
		// Signature is stored in meta
		gt.NotNil(t, content.Meta)
	})
}
