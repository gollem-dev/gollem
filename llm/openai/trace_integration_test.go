package openai_test

import (
	"context"
	"os"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/gt"
)

func TestOpenAITraceIntegration(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	client, err := openai.New(ctx, apiKey)
	gt.NoError(t, err)

	// Create recorder and inject into context
	rec := trace.New()
	rootCtx := rec.StartAgentExecute(ctx)
	rootCtx = trace.WithHandler(rootCtx, rec)

	session, err := client.NewSession(rootCtx)
	gt.NoError(t, err)

	result, err := session.Generate(rootCtx, []gollem.Input{gollem.Text("Say hello in one word")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err)
	gt.A(t, result.Texts).Longer(0)

	rec.EndAgentExecute(rootCtx, nil)

	// Verify trace data
	tr := rec.Trace()
	gt.Value(t, tr).NotNil()

	rootSpan := tr.RootSpan
	gt.Value(t, rootSpan).NotNil()
	gt.Equal(t, rootSpan.Kind, trace.SpanKindAgentExecute)

	// Should have at least one LLM call child
	gt.A(t, rootSpan.Children).Longer(0)

	llmSpan := rootSpan.Children[0]
	gt.Equal(t, llmSpan.Kind, trace.SpanKindLLMCall)
	gt.Value(t, llmSpan.LLMCall).NotNil()

	// Verify request data
	gt.Value(t, llmSpan.LLMCall.Request).NotNil()
	gt.A(t, llmSpan.LLMCall.Request.Messages).Longer(0)

	// Find the user message with our input
	var foundUserMsg bool
	for _, msg := range llmSpan.LLMCall.Request.Messages {
		if msg.Role == "user" {
			for _, c := range msg.Contents {
				if c.Type == "text" && c.Text != "" {
					foundUserMsg = true
				}
			}
		}
	}
	gt.B(t, foundUserMsg).True()

	// Verify response data
	gt.Value(t, llmSpan.LLMCall.Response).NotNil()
	gt.A(t, llmSpan.LLMCall.Response.Texts).Longer(0)

	// Verify token counts
	gt.N(t, llmSpan.LLMCall.InputTokens).Greater(0)
	gt.N(t, llmSpan.LLMCall.OutputTokens).Greater(0)

	// Verify model is set
	gt.S(t, llmSpan.LLMCall.Model).NotEqual("")
}
