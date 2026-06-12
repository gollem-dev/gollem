package gemini_test

import (
	"context"
	"os"
	"testing"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/gt"
)

func TestGeminiTraceIntegration(t *testing.T) {
	testProjectID, ok := os.LookupEnv("TEST_GCP_PROJECT_ID")
	if !ok {
		t.Skip("TEST_GCP_PROJECT_ID is not set")
	}

	testLocation, ok := os.LookupEnv("TEST_GCP_LOCATION")
	if !ok {
		t.Skip("TEST_GCP_LOCATION is not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), testTimeout)
	defer cancel()

	var opts []gemini.Option
	if model := os.Getenv("TEST_GCP_MODEL"); model != "" {
		opts = append(opts, gemini.WithModel(model))
	}

	client, err := gemini.New(ctx, testProjectID, testLocation, opts...)
	gt.NoError(t, err)

	// Create recorder and inject into context
	rec := trace.New()
	rootCtx := rec.StartAgentExecute(ctx)
	rootCtx = trace.WithHandler(rootCtx, rec)

	session, err := client.NewSession(rootCtx)
	gt.NoError(t, err)

	result, err := session.Generate(rootCtx, []gollem.Input{gollem.Text("Say hello in one word")}, gollem.WithMaxTokens(maxTestTokens))
	gt.NoError(t, err).Required()
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

	// The last message should be user with our input text
	lastMsg := llmSpan.LLMCall.Request.Messages[len(llmSpan.LLMCall.Request.Messages)-1]
	gt.Equal(t, lastMsg.Role, "user")
	gt.A(t, lastMsg.Contents).Longer(0)
	gt.Equal(t, lastMsg.Contents[0].Type, "text")
	gt.S(t, lastMsg.Contents[0].Text).Contains("hello")

	// Verify response data
	gt.Value(t, llmSpan.LLMCall.Response).NotNil()
	gt.A(t, llmSpan.LLMCall.Response.Texts).Longer(0)

	// Verify token counts
	gt.N(t, llmSpan.LLMCall.InputTokens).Greater(0)
	gt.N(t, llmSpan.LLMCall.OutputTokens).Greater(0)

	// Verify model is set
	gt.S(t, llmSpan.LLMCall.Model).NotEqual("")
}
