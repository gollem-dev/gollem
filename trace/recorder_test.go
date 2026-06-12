package trace_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/gt"
)

func TestRecorderContextPropagation(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	// Before setting, RecorderFrom returns nil
	gt.Value(t, trace.HandlerFrom(ctx)).Nil()

	// After setting, RecorderFrom returns the recorder
	ctx = trace.WithHandler(ctx, rec)
	gt.Value(t, trace.HandlerFrom(ctx)).NotNil()
	gt.Equal[trace.Handler](t, trace.HandlerFrom(ctx), rec)
}

func TestRecorderAgentExecuteSpan(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()
	ctx = trace.WithHandler(ctx, rec)

	// Start agent execute
	agentCtx := rec.StartAgentExecute(ctx)

	// Current span should exist
	span := trace.CurrentSpanFrom(agentCtx)
	gt.Value(t, span).NotNil()
	gt.Equal(t, span.Kind, trace.SpanKindAgentExecute)
	gt.Equal(t, span.Name, "agent_execute")

	// Trace should be initialized
	tr := rec.Trace()
	gt.Value(t, tr).NotNil()
	gt.Equal(t, tr.RootSpan, span)

	// End agent execute
	rec.EndAgentExecute(agentCtx, nil)
	gt.Equal(t, span.Status, trace.SpanStatusOK)
	gt.B(t, span.Duration > 0).True()
}

func TestRecorderAgentExecuteWithError(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)
	rec.EndAgentExecute(agentCtx, errors.New("execution failed"))

	span := trace.CurrentSpanFrom(agentCtx)
	gt.Equal(t, span.Status, trace.SpanStatusError)
	gt.Equal(t, span.Error, "execution failed")
}

func TestRecorderLLMCallSpan(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)
	llmCtx := rec.StartLLMCall(agentCtx)

	llmSpan := trace.CurrentSpanFrom(llmCtx)
	gt.Value(t, llmSpan).NotNil()
	gt.Equal(t, llmSpan.Kind, trace.SpanKindLLMCall)

	rec.EndLLMCall(llmCtx, &trace.LLMCallData{
		InputTokens:  100,
		OutputTokens: 50,
		Model:        "test-model",
		Request: &trace.LLMRequest{
			SystemPrompt: "You are helpful",
			Messages:     []trace.Message{{Role: "user", Contents: []trace.MessageContent{trace.NewTextContent("hello")}}},
		},
		Response: &trace.LLMResponse{
			Texts: []string{"Hi!"},
		},
	}, nil)

	gt.Equal(t, llmSpan.LLMCall.InputTokens, 100)
	gt.Equal(t, llmSpan.LLMCall.OutputTokens, 50)
	gt.Equal(t, llmSpan.LLMCall.Model, "test-model")
	gt.Equal(t, llmSpan.Status, trace.SpanStatusOK)

	// LLM span should be a child of the agent execute span
	rootSpan := rec.Trace().RootSpan
	gt.Equal(t, len(rootSpan.Children), 1)
	gt.Equal(t, rootSpan.Children[0], llmSpan)
}

func TestRecorderToolExecSpan(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)
	toolCtx := rec.StartToolExec(agentCtx, "search", map[string]any{"query": "test"})

	toolSpan := trace.CurrentSpanFrom(toolCtx)
	gt.Value(t, toolSpan).NotNil()
	gt.Equal(t, toolSpan.Kind, trace.SpanKindToolExec)
	gt.Equal(t, toolSpan.ToolExec.ToolName, "search")
	gt.Equal(t, toolSpan.ToolExec.Args["query"], "test")

	result := map[string]any{"found": true}
	rec.EndToolExec(toolCtx, result, nil)

	gt.Equal(t, toolSpan.ToolExec.Result["found"], true)
	gt.Equal(t, toolSpan.Status, trace.SpanStatusOK)
}

func TestRecorderToolExecWithError(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)
	toolCtx := rec.StartToolExec(agentCtx, "search", map[string]any{})

	rec.EndToolExec(toolCtx, nil, errors.New("tool failed"))

	toolSpan := trace.CurrentSpanFrom(toolCtx)
	gt.Equal(t, toolSpan.Status, trace.SpanStatusError)
	gt.Equal(t, toolSpan.ToolExec.Error, "tool failed")
}

func TestRecorderSubAgentSpan(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)
	subCtx := rec.StartSubAgent(agentCtx, "child_agent")

	subSpan := trace.CurrentSpanFrom(subCtx)
	gt.Value(t, subSpan).NotNil()
	gt.Equal(t, subSpan.Kind, trace.SpanKindSubAgent)
	gt.Equal(t, subSpan.Name, "child_agent")

	// LLM call inside sub agent
	llmCtx := rec.StartLLMCall(subCtx)
	rec.EndLLMCall(llmCtx, &trace.LLMCallData{
		InputTokens: 10,
	}, nil)

	// LLM span should be a child of the sub agent span
	gt.Equal(t, len(subSpan.Children), 1)
	gt.Equal(t, subSpan.Children[0].Kind, trace.SpanKindLLMCall)

	rec.EndSubAgent(subCtx, nil)
	gt.Equal(t, subSpan.Status, trace.SpanStatusOK)
}

func TestRecorderAddEvent(t *testing.T) {
	type testEvent struct {
		Goal string `json:"goal"`
	}

	rec := trace.New()
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)
	rec.AddEvent(agentCtx, "plan_created", &testEvent{Goal: "implement feature"})

	rootSpan := rec.Trace().RootSpan
	gt.Equal(t, len(rootSpan.Children), 1)
	gt.Equal(t, rootSpan.Children[0].Kind, trace.SpanKindEvent)
	gt.Equal(t, rootSpan.Children[0].Event.Kind, "plan_created")

	data, ok := rootSpan.Children[0].Event.Data.(*testEvent)
	gt.B(t, ok).True()
	gt.Equal(t, data.Goal, "implement feature")

	// Event span should have zero duration
	gt.Equal(t, rootSpan.Children[0].StartedAt, rootSpan.Children[0].EndedAt)
}

func TestRecorderChildOrdering(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)

	// Add children in order: LLM, Event, Tool, Event, LLM
	llm1 := rec.StartLLMCall(agentCtx)
	rec.EndLLMCall(llm1, &trace.LLMCallData{InputTokens: 10}, nil)

	rec.AddEvent(agentCtx, "plan_created", nil)

	tool1 := rec.StartToolExec(agentCtx, "search", nil)
	rec.EndToolExec(tool1, nil, nil)

	rec.AddEvent(agentCtx, "task_completed", nil)

	llm2 := rec.StartLLMCall(agentCtx)
	rec.EndLLMCall(llm2, &trace.LLMCallData{InputTokens: 20}, nil)

	rootSpan := rec.Trace().RootSpan
	gt.Equal(t, len(rootSpan.Children), 5)
	gt.Equal(t, rootSpan.Children[0].Kind, trace.SpanKindLLMCall)
	gt.Equal(t, rootSpan.Children[1].Kind, trace.SpanKindEvent)
	gt.Equal(t, rootSpan.Children[1].Event.Kind, "plan_created")
	gt.Equal(t, rootSpan.Children[2].Kind, trace.SpanKindToolExec)
	gt.Equal(t, rootSpan.Children[3].Kind, trace.SpanKindEvent)
	gt.Equal(t, rootSpan.Children[3].Event.Kind, "task_completed")
	gt.Equal(t, rootSpan.Children[4].Kind, trace.SpanKindLLMCall)
}

func TestRecorderFinish(t *testing.T) {
	dir := t.TempDir()
	repo := trace.NewFileRepository(dir)
	rec := trace.New(
		trace.WithRepository(repo),
		trace.WithMetadata(trace.TraceMetadata{
			Model:    "test-model",
			Strategy: "simple",
			Labels:   map[string]string{"env": "test"},
		}),
	)

	ctx := context.Background()
	agentCtx := rec.StartAgentExecute(ctx)
	rec.EndAgentExecute(agentCtx, nil)

	err := rec.Finish(ctx)
	gt.NoError(t, err)

	// Verify trace was saved
	tr := rec.Trace()
	gt.Value(t, tr).NotNil()
	gt.Equal(t, tr.Metadata.Model, "test-model")
	gt.Equal(t, tr.Metadata.Strategy, "simple")
}

func TestRecorderFinishWithoutRepo(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)
	rec.EndAgentExecute(agentCtx, nil)

	// Finish without repo should not error
	err := rec.Finish(ctx)
	gt.NoError(t, err)
}

func TestRecorderNoOpWhenNilContext(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	// These should all be no-ops since there's no current span
	llmCtx := rec.StartLLMCall(ctx)
	gt.Equal(t, llmCtx, ctx)

	toolCtx := rec.StartToolExec(ctx, "search", nil)
	gt.Equal(t, toolCtx, ctx)

	subCtx := rec.StartSubAgent(ctx, "child")
	gt.Equal(t, subCtx, ctx)

	// End methods should not panic with nil spans
	rec.EndLLMCall(ctx, nil, nil)
	rec.EndToolExec(ctx, nil, nil)
	rec.EndSubAgent(ctx, nil)
	rec.AddEvent(ctx, "test", nil)
}

func TestRecorderConcurrentAccess(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()
	agentCtx := rec.StartAgentExecute(ctx)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			llmCtx := rec.StartLLMCall(agentCtx)
			rec.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 1}, nil)
		}()
	}
	wg.Wait()

	rootSpan := rec.Trace().RootSpan
	gt.Equal(t, len(rootSpan.Children), 10)
}

func TestRecorderWithCustomTraceID(t *testing.T) {
	customID := "my-request-12345"
	rec := trace.New(trace.WithTraceID(customID))
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)
	rec.EndAgentExecute(agentCtx, nil)

	tr := rec.Trace()
	gt.Value(t, tr).NotNil()
	gt.Equal(t, tr.TraceID, customID)
}

func TestRecorderWithEmptyTraceID(t *testing.T) {
	rec := trace.New(trace.WithTraceID(""))
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)
	rec.EndAgentExecute(agentCtx, nil)

	tr := rec.Trace()
	gt.Value(t, tr).NotNil()
	// Empty string should fall back to auto-generated UUID
	gt.Value(t, tr.TraceID).NotEqual("")
}

func TestCaptureStackTrace(t *testing.T) {
	frames := trace.CaptureStackTrace(0)
	gt.A(t, frames).Longer(0)

	// The first frame should be this test function in this file
	gt.S(t, frames[0].Function).Contains("TestCaptureStackTrace")
	gt.S(t, frames[0].File).Contains("trace/recorder_test.go")
	gt.N(t, frames[0].Line).Greater(0)
}

func TestRecorderStackTraceDisabledByDefault(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)
	llmCtx := rec.StartLLMCall(agentCtx)
	rec.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 10}, nil)

	toolCtx := rec.StartToolExec(agentCtx, "search", nil)
	rec.EndToolExec(toolCtx, nil, nil)

	rec.AddEvent(agentCtx, "test_event", nil)

	subCtx := rec.StartSubAgent(agentCtx, "child")
	rec.EndSubAgent(subCtx, nil)

	rec.EndAgentExecute(agentCtx, nil)

	rootSpan := rec.Trace().RootSpan
	gt.Value(t, rootSpan.StackTrace).Nil()
	for _, child := range rootSpan.Children {
		gt.Value(t, child.StackTrace).Nil()
	}
}

func TestRecorderStackTraceEnabled(t *testing.T) {
	rec := trace.New(trace.WithStackTrace())
	ctx := context.Background()

	agentCtx := rec.StartAgentExecute(ctx)
	llmCtx := rec.StartLLMCall(agentCtx)
	rec.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 10}, nil)
	toolCtx := rec.StartToolExec(agentCtx, "search", map[string]any{"q": "test"})
	rec.EndToolExec(toolCtx, nil, nil)
	rec.AddEvent(agentCtx, "test_event", nil)
	subCtx := rec.StartSubAgent(agentCtx, "child")
	rec.EndSubAgent(subCtx, nil)
	rec.EndAgentExecute(agentCtx, nil)

	rootSpan := rec.Trace().RootSpan

	// Children: llm_call, tool_exec, event, sub_agent (in order)
	gt.A(t, rootSpan.Children).Length(4)

	// The top frame of each span's stack trace should be the caller of the
	// Recorder method, not the Recorder method itself. In production this
	// means gollem.go, llm/*/client.go, subagent.go, or strategy code.
	// In tests, the caller is this test function in recorder_test.go.

	t.Run("agent_execute span top frame is the caller of StartAgentExecute", func(t *testing.T) {
		gt.A(t, rootSpan.StackTrace).Longer(0)
		topFrame := rootSpan.StackTrace[0]
		gt.S(t, topFrame.Function).Contains("TestRecorderStackTraceEnabled")
		gt.S(t, topFrame.File).Contains("trace/recorder_test.go")
		gt.N(t, topFrame.Line).Greater(0)
	})

	t.Run("llm_call span top frame is the caller of StartLLMCall", func(t *testing.T) {
		span := rootSpan.Children[0]
		gt.V(t, span.Kind).Equal(trace.SpanKindLLMCall)
		gt.A(t, span.StackTrace).Longer(0)
		topFrame := span.StackTrace[0]
		gt.S(t, topFrame.Function).Contains("TestRecorderStackTraceEnabled")
		gt.S(t, topFrame.File).Contains("trace/recorder_test.go")
		gt.N(t, topFrame.Line).Greater(0)
	})

	t.Run("tool_exec span top frame is the caller of StartToolExec", func(t *testing.T) {
		span := rootSpan.Children[1]
		gt.V(t, span.Kind).Equal(trace.SpanKindToolExec)
		gt.A(t, span.StackTrace).Longer(0)
		topFrame := span.StackTrace[0]
		gt.S(t, topFrame.Function).Contains("TestRecorderStackTraceEnabled")
		gt.S(t, topFrame.File).Contains("trace/recorder_test.go")
		gt.N(t, topFrame.Line).Greater(0)
	})

	t.Run("event span top frame is the caller of AddEvent", func(t *testing.T) {
		span := rootSpan.Children[2]
		gt.V(t, span.Kind).Equal(trace.SpanKindEvent)
		gt.A(t, span.StackTrace).Longer(0)
		topFrame := span.StackTrace[0]
		gt.S(t, topFrame.Function).Contains("TestRecorderStackTraceEnabled")
		gt.S(t, topFrame.File).Contains("trace/recorder_test.go")
		gt.N(t, topFrame.Line).Greater(0)
	})

	t.Run("sub_agent span top frame is the caller of StartSubAgent", func(t *testing.T) {
		span := rootSpan.Children[3]
		gt.V(t, span.Kind).Equal(trace.SpanKindSubAgent)
		gt.A(t, span.StackTrace).Longer(0)
		topFrame := span.StackTrace[0]
		gt.S(t, topFrame.Function).Contains("TestRecorderStackTraceEnabled")
		gt.S(t, topFrame.File).Contains("trace/recorder_test.go")
		gt.N(t, topFrame.Line).Greater(0)
	})

	t.Run("each span has a different top frame line number", func(t *testing.T) {
		lines := map[int]trace.SpanKind{}
		allSpans := append([]*trace.Span{rootSpan}, rootSpan.Children...)
		for _, span := range allSpans {
			topLine := span.StackTrace[0].Line
			_, duplicate := lines[topLine]
			gt.B(t, duplicate).False()
			lines[topLine] = span.Kind
		}
	})
}

func TestRecorderStartAgentExecuteChildFallback(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	// First call: creates root trace
	rootCtx := rec.StartAgentExecute(ctx)
	rootSpan := trace.CurrentSpanFrom(rootCtx)
	gt.Value(t, rootSpan).NotNil()
	gt.Equal(t, rootSpan.Kind, trace.SpanKindAgentExecute)

	tr := rec.Trace()
	gt.Value(t, tr).NotNil()
	originalTraceID := tr.TraceID

	// Second call: should fall back to child span, NOT overwrite trace
	childCtx := rec.StartAgentExecute(rootCtx)
	childSpan := trace.CurrentSpanFrom(childCtx)
	gt.Value(t, childSpan).NotNil()
	gt.Equal(t, childSpan.Kind, trace.SpanKindAgentExecute)

	// Root trace should still exist with same ID
	gt.Equal(t, rec.Trace().TraceID, originalTraceID)
	gt.Equal(t, rec.Trace().RootSpan, rootSpan)

	// Child span should be a child of root span
	gt.Equal(t, len(rootSpan.Children), 1)
	gt.Equal(t, rootSpan.Children[0], childSpan)
	gt.Equal(t, childSpan.ParentID, rootSpan.SpanID)

	// End child
	rec.EndAgentExecute(childCtx, nil)
	gt.Equal(t, childSpan.Status, trace.SpanStatusOK)

	// End root
	rec.EndAgentExecute(rootCtx, nil)
	gt.Equal(t, rootSpan.Status, trace.SpanStatusOK)
}

func TestRecorderStartAgentExecuteMultipleChildren(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	// Create root trace
	rootCtx := rec.StartAgentExecute(ctx)
	rootSpan := trace.CurrentSpanFrom(rootCtx)

	// First child agent
	child1Ctx := rec.StartAgentExecute(rootCtx)
	child1Span := trace.CurrentSpanFrom(child1Ctx)

	// Add LLM call inside child1
	llmCtx := rec.StartLLMCall(child1Ctx)
	rec.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 10}, nil)

	rec.EndAgentExecute(child1Ctx, nil)

	// Second child agent
	child2Ctx := rec.StartAgentExecute(rootCtx)
	child2Span := trace.CurrentSpanFrom(child2Ctx)

	// Add LLM call inside child2
	llm2Ctx := rec.StartLLMCall(child2Ctx)
	rec.EndLLMCall(llm2Ctx, &trace.LLMCallData{InputTokens: 20}, nil)

	rec.EndAgentExecute(child2Ctx, nil)

	rec.EndAgentExecute(rootCtx, nil)

	// Verify tree structure
	gt.Equal(t, len(rootSpan.Children), 2)
	gt.Equal(t, rootSpan.Children[0], child1Span)
	gt.Equal(t, rootSpan.Children[1], child2Span)

	// Each child should have one LLM call child
	gt.Equal(t, len(child1Span.Children), 1)
	gt.Equal(t, child1Span.Children[0].Kind, trace.SpanKindLLMCall)
	gt.Equal(t, child1Span.Children[0].LLMCall.InputTokens, 10)

	gt.Equal(t, len(child2Span.Children), 1)
	gt.Equal(t, child2Span.Children[0].Kind, trace.SpanKindLLMCall)
	gt.Equal(t, child2Span.Children[0].LLMCall.InputTokens, 20)

	// Original trace should be preserved
	gt.Value(t, rec.Trace()).NotNil()
	gt.Equal(t, rec.Trace().RootSpan, rootSpan)
}

func TestRecorderStartAgentExecuteTripleNesting(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	// Root
	rootCtx := rec.StartAgentExecute(ctx)
	rootSpan := trace.CurrentSpanFrom(rootCtx)

	// Child 1 (inside root)
	child1Ctx := rec.StartAgentExecute(rootCtx)
	child1Span := trace.CurrentSpanFrom(child1Ctx)

	// Grandchild (inside child 1)
	grandchildCtx := rec.StartAgentExecute(child1Ctx)
	grandchildSpan := trace.CurrentSpanFrom(grandchildCtx)

	rec.EndAgentExecute(grandchildCtx, nil)
	rec.EndAgentExecute(child1Ctx, nil)
	rec.EndAgentExecute(rootCtx, nil)

	// Verify 3-level nesting
	gt.Equal(t, len(rootSpan.Children), 1)
	gt.Equal(t, rootSpan.Children[0], child1Span)
	gt.Equal(t, len(child1Span.Children), 1)
	gt.Equal(t, child1Span.Children[0], grandchildSpan)
	gt.Equal(t, grandchildSpan.ParentID, child1Span.SpanID)
}

func TestRecorderStartAgentExecuteFallbackToRootSpan(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()

	// Create root trace
	rootCtx := rec.StartAgentExecute(ctx)
	rootSpan := trace.CurrentSpanFrom(rootCtx)
	gt.Value(t, rootSpan).NotNil()

	// Call StartAgentExecute with a context that has NO current span
	// (e.g., a fresh context without span info). It should fall back
	// to using r.trace.RootSpan as the parent.
	freshCtx := context.Background()
	childCtx := rec.StartAgentExecute(freshCtx)
	childSpan := trace.CurrentSpanFrom(childCtx)
	gt.Value(t, childSpan).NotNil()
	gt.Equal(t, childSpan.Kind, trace.SpanKindAgentExecute)
	gt.Equal(t, childSpan.ParentID, rootSpan.SpanID)

	// Child should be attached to root span
	gt.Equal(t, len(rootSpan.Children), 1)
	gt.Equal(t, rootSpan.Children[0], childSpan)

	rec.EndAgentExecute(childCtx, nil)
	rec.EndAgentExecute(rootCtx, nil)
}
