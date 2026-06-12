package trace_test

import (
	"context"
	"testing"

	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/gt"
)

// TestMultiAgentExecuteSharedRecorder simulates the pattern where a user creates
// a Recorder, starts a root trace, then passes the same Recorder to multiple
// Agent.Execute calls (each of which calls StartAgentExecute internally).
func TestMultiAgentExecuteSharedRecorder(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	// User creates root trace
	rootCtx := rec.StartAgentExecute(ctx)
	rootCtx = trace.WithHandler(rootCtx, rec)
	rootSpan := trace.CurrentSpanFrom(rootCtx)

	// Simulate first Agent.Execute: calls StartAgentExecute on the same recorder
	agent1Ctx := rec.StartAgentExecute(rootCtx)
	agent1Span := trace.CurrentSpanFrom(agent1Ctx)

	// Agent 1 makes LLM calls
	llm1Ctx := rec.StartLLMCall(agent1Ctx)
	rec.EndLLMCall(llm1Ctx, &trace.LLMCallData{
		InputTokens:  100,
		OutputTokens: 50,
		Model:        "claude-3-sonnet",
		Request: &trace.LLMRequest{
			SystemPrompt: "You are helpful",
			Messages: []trace.Message{
				{Role: "user", Contents: []trace.MessageContent{{Type: "text", Text: "Hello"}}},
			},
		},
		Response: &trace.LLMResponse{
			Texts: []string{"Hi there!"},
		},
	}, nil)

	// Agent 1 uses a tool
	toolCtx := rec.StartToolExec(agent1Ctx, "search", map[string]any{"q": "test"})
	rec.EndToolExec(toolCtx, map[string]any{"result": "found"}, nil)

	// Agent 1 makes another LLM call after tool
	llm1bCtx := rec.StartLLMCall(agent1Ctx)
	rec.EndLLMCall(llm1bCtx, &trace.LLMCallData{
		InputTokens:  200,
		OutputTokens: 100,
		Model:        "claude-3-sonnet",
	}, nil)

	rec.EndAgentExecute(agent1Ctx, nil)

	// Simulate second Agent.Execute: calls StartAgentExecute again
	agent2Ctx := rec.StartAgentExecute(rootCtx)
	agent2Span := trace.CurrentSpanFrom(agent2Ctx)

	// Agent 2 makes LLM calls
	llm2Ctx := rec.StartLLMCall(agent2Ctx)
	rec.EndLLMCall(llm2Ctx, &trace.LLMCallData{
		InputTokens:  150,
		OutputTokens: 75,
		Model:        "claude-3-sonnet",
		Request: &trace.LLMRequest{
			Messages: []trace.Message{
				{Role: "user", Contents: []trace.MessageContent{{Type: "text", Text: "Summarize"}}},
			},
		},
		Response: &trace.LLMResponse{
			Texts: []string{"Here is a summary..."},
		},
	}, nil)

	rec.EndAgentExecute(agent2Ctx, nil)
	rec.EndAgentExecute(rootCtx, nil)

	// === Verification ===

	// Root trace should exist and be intact
	tr := rec.Trace()
	gt.Value(t, tr).NotNil()
	gt.Equal(t, tr.RootSpan, rootSpan)

	// Root span should have 2 children (both agent_execute)
	gt.A(t, rootSpan.Children).Length(2)

	// First child: agent1
	gt.Equal(t, rootSpan.Children[0], agent1Span)
	gt.Equal(t, agent1Span.Kind, trace.SpanKindAgentExecute)
	gt.Equal(t, agent1Span.ParentID, rootSpan.SpanID)

	// Agent 1 should have 3 children: LLM, Tool, LLM
	gt.A(t, agent1Span.Children).Length(3)
	gt.Equal(t, agent1Span.Children[0].Kind, trace.SpanKindLLMCall)
	gt.Equal(t, agent1Span.Children[0].LLMCall.InputTokens, 100)
	gt.A(t, agent1Span.Children[0].LLMCall.Request.Messages).Length(1)
	gt.Equal(t, agent1Span.Children[0].LLMCall.Request.Messages[0].Contents[0].Text, "Hello")
	gt.Equal(t, agent1Span.Children[1].Kind, trace.SpanKindToolExec)
	gt.Equal(t, agent1Span.Children[1].ToolExec.ToolName, "search")
	gt.Equal(t, agent1Span.Children[2].Kind, trace.SpanKindLLMCall)
	gt.Equal(t, agent1Span.Children[2].LLMCall.InputTokens, 200)

	// Second child: agent2
	gt.Equal(t, rootSpan.Children[1], agent2Span)
	gt.Equal(t, agent2Span.Kind, trace.SpanKindAgentExecute)
	gt.Equal(t, agent2Span.ParentID, rootSpan.SpanID)

	// Agent 2 should have 1 child: LLM
	gt.A(t, agent2Span.Children).Length(1)
	gt.Equal(t, agent2Span.Children[0].Kind, trace.SpanKindLLMCall)
	gt.Equal(t, agent2Span.Children[0].LLMCall.InputTokens, 150)
	gt.A(t, agent2Span.Children[0].LLMCall.Request.Messages).Length(1)
	gt.Equal(t, agent2Span.Children[0].LLMCall.Request.Messages[0].Contents[0].Text, "Summarize")

	// All spans should be terminated
	gt.B(t, rootSpan.Duration > 0).True()
	gt.B(t, agent1Span.Duration > 0).True()
	gt.B(t, agent2Span.Duration > 0).True()
}

// TestAsChildAgentVsDirectRecorder verifies that using AsChildAgent wrapper
// produces the same result as the Recorder's built-in child fallback.
func TestAsChildAgentVsDirectRecorder(t *testing.T) {
	t.Run("direct recorder fallback", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()

		rootCtx := rec.StartAgentExecute(ctx)
		rootSpan := trace.CurrentSpanFrom(rootCtx)

		// Pass recorder directly (triggers fallback)
		childCtx := rec.StartAgentExecute(rootCtx)
		childSpan := trace.CurrentSpanFrom(childCtx)

		llmCtx := rec.StartLLMCall(childCtx)
		rec.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 42}, nil)

		rec.EndAgentExecute(childCtx, nil)
		rec.EndAgentExecute(rootCtx, nil)

		// Verify structure
		gt.A(t, rootSpan.Children).Length(1)
		gt.Equal(t, rootSpan.Children[0], childSpan)
		gt.Equal(t, childSpan.Kind, trace.SpanKindAgentExecute)
		gt.A(t, childSpan.Children).Length(1)
		gt.Equal(t, childSpan.Children[0].LLMCall.InputTokens, 42)
	})

	t.Run("AsChildAgent wrapper", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()

		rootCtx := rec.StartAgentExecute(ctx)
		rootSpan := trace.CurrentSpanFrom(rootCtx)

		// Use AsChildAgent wrapper
		childHandler := trace.AsChildAgent(rec, "task-1")
		childCtx := childHandler.StartAgentExecute(rootCtx)
		childSpan := trace.CurrentSpanFrom(childCtx)

		llmCtx := childHandler.StartLLMCall(childCtx)
		childHandler.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 42}, nil)

		childHandler.EndAgentExecute(childCtx, nil)
		rec.EndAgentExecute(rootCtx, nil)

		// Verify structure - should be same as direct
		gt.A(t, rootSpan.Children).Length(1)
		gt.Equal(t, rootSpan.Children[0], childSpan)
		gt.Equal(t, childSpan.Kind, trace.SpanKindAgentExecute)
		gt.A(t, childSpan.Children).Length(1)
		gt.Equal(t, childSpan.Children[0].LLMCall.InputTokens, 42)
	})
}

// TestRecorderFinishPreservesTrace verifies that calling Finish does not
// clear the trace, allowing further StartAgentExecute calls to still
// fall back to child spans.
func TestRecorderFinishPreservesTrace(t *testing.T) {
	dir := t.TempDir()
	repo := trace.NewFileRepository(dir)
	rec := trace.New(trace.WithRepository(repo))
	ctx := context.Background()

	rootCtx := rec.StartAgentExecute(ctx)

	// Simulate Agent.Execute finishing (calls Finish)
	err := rec.Finish(ctx)
	gt.NoError(t, err)

	// Trace should still be accessible
	gt.Value(t, rec.Trace()).NotNil()

	// Second StartAgentExecute should still create child
	childCtx := rec.StartAgentExecute(rootCtx)
	childSpan := trace.CurrentSpanFrom(childCtx)
	gt.Value(t, childSpan).NotNil()
	gt.Equal(t, childSpan.Kind, trace.SpanKindAgentExecute)

	rootSpan := rec.Trace().RootSpan
	gt.A(t, rootSpan.Children).Length(1)
	gt.Equal(t, rootSpan.Children[0], childSpan)

	rec.EndAgentExecute(childCtx, nil)
	rec.EndAgentExecute(rootCtx, nil)
}
