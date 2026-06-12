package trace_test

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/gt"
)

func TestAsChildAgentDelegation(t *testing.T) {
	t.Run("StartAgentExecute maps to parent StartChildAgent", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()

		// Start root agent
		agentCtx := rec.StartAgentExecute(ctx)

		// Create AsChildAgent handler and start it (simulating child Agent.Execute)
		childHandler := trace.AsChildAgent(rec, "task-1")
		childCtx := childHandler.StartAgentExecute(agentCtx)

		// The child span should be an agent_execute span under the root
		childSpan := trace.CurrentSpanFrom(childCtx)
		gt.Value(t, childSpan).NotNil()
		gt.Equal(t, childSpan.Kind, trace.SpanKindAgentExecute)
		gt.Equal(t, childSpan.Name, "task-1")

		rootSpan := rec.Trace().RootSpan
		gt.Equal(t, len(rootSpan.Children), 1)
		gt.Equal(t, rootSpan.Children[0], childSpan)

		childHandler.EndAgentExecute(childCtx, nil)
		gt.Equal(t, childSpan.Status, trace.SpanStatusOK)
	})

	t.Run("EndAgentExecute maps to parent EndChildAgent with error", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()
		agentCtx := rec.StartAgentExecute(ctx)

		childHandler := trace.AsChildAgent(rec, "task-err")
		childCtx := childHandler.StartAgentExecute(agentCtx)

		childHandler.EndAgentExecute(childCtx, errors.New("child failed"))

		childSpan := trace.CurrentSpanFrom(childCtx)
		gt.Equal(t, childSpan.Status, trace.SpanStatusError)
		gt.Equal(t, childSpan.Error, "child failed")
	})

	t.Run("LLMCall delegates to parent", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()
		agentCtx := rec.StartAgentExecute(ctx)

		childHandler := trace.AsChildAgent(rec, "task-1")
		childCtx := childHandler.StartAgentExecute(agentCtx)

		llmCtx := childHandler.StartLLMCall(childCtx)
		childHandler.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 42}, nil)

		childSpan := trace.CurrentSpanFrom(childCtx)
		gt.Equal(t, len(childSpan.Children), 1)
		gt.Equal(t, childSpan.Children[0].Kind, trace.SpanKindLLMCall)
	})

	t.Run("ToolExec delegates to parent", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()
		agentCtx := rec.StartAgentExecute(ctx)

		childHandler := trace.AsChildAgent(rec, "task-1")
		childCtx := childHandler.StartAgentExecute(agentCtx)

		toolCtx := childHandler.StartToolExec(childCtx, "search", map[string]any{"q": "test"})
		childHandler.EndToolExec(toolCtx, map[string]any{"found": true}, nil)

		childSpan := trace.CurrentSpanFrom(childCtx)
		gt.Equal(t, len(childSpan.Children), 1)
		gt.Equal(t, childSpan.Children[0].Kind, trace.SpanKindToolExec)
	})

	t.Run("SubAgent delegates to parent", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()
		agentCtx := rec.StartAgentExecute(ctx)

		childHandler := trace.AsChildAgent(rec, "task-1")
		childCtx := childHandler.StartAgentExecute(agentCtx)

		subCtx := childHandler.StartSubAgent(childCtx, "searcher")
		childHandler.EndSubAgent(subCtx, nil)

		childSpan := trace.CurrentSpanFrom(childCtx)
		gt.Equal(t, len(childSpan.Children), 1)
		gt.Equal(t, childSpan.Children[0].Kind, trace.SpanKindSubAgent)
		gt.Equal(t, childSpan.Children[0].Name, "searcher")
	})

	t.Run("ChildAgent delegates to parent", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()
		agentCtx := rec.StartAgentExecute(ctx)

		childHandler := trace.AsChildAgent(rec, "task-1")
		childCtx := childHandler.StartAgentExecute(agentCtx)

		grandchildCtx := childHandler.StartChildAgent(childCtx, "sub-task")
		childHandler.EndChildAgent(grandchildCtx, nil)

		childSpan := trace.CurrentSpanFrom(childCtx)
		gt.Equal(t, len(childSpan.Children), 1)
		gt.Equal(t, childSpan.Children[0].Kind, trace.SpanKindAgentExecute)
		gt.Equal(t, childSpan.Children[0].Name, "sub-task")
	})

	t.Run("AddEvent delegates to parent", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()
		agentCtx := rec.StartAgentExecute(ctx)

		childHandler := trace.AsChildAgent(rec, "task-1")
		childCtx := childHandler.StartAgentExecute(agentCtx)

		childHandler.AddEvent(childCtx, "plan_created", map[string]any{"goal": "test"})

		childSpan := trace.CurrentSpanFrom(childCtx)
		gt.Equal(t, len(childSpan.Children), 1)
		gt.Equal(t, childSpan.Children[0].Kind, trace.SpanKindEvent)
	})

	t.Run("Finish is no-op", func(t *testing.T) {
		dir := t.TempDir()
		repo := trace.NewFileRepository(dir)
		rec := trace.New(trace.WithRepository(repo))
		ctx := context.Background()

		agentCtx := rec.StartAgentExecute(ctx)
		childHandler := trace.AsChildAgent(rec, "task-1")

		// Finish on child handler should not persist trace
		err := childHandler.Finish(ctx)
		gt.NoError(t, err)

		// Root handler Finish should still work
		rec.EndAgentExecute(agentCtx, nil)
		err = rec.Finish(ctx)
		gt.NoError(t, err)
	})
}

func TestAsChildAgentRecorderIntegration(t *testing.T) {
	t.Run("trace tree structure with AsChildAgent", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()

		// Root agent starts
		rootCtx := rec.StartAgentExecute(ctx)

		// Child agent via AsChildAgent
		childHandler := trace.AsChildAgent(rec, "analyzer")
		childCtx := childHandler.StartAgentExecute(rootCtx)

		// LLM call inside child
		llmCtx := childHandler.StartLLMCall(childCtx)
		childHandler.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 100}, nil)

		// Tool exec inside child
		toolCtx := childHandler.StartToolExec(childCtx, "search", map[string]any{"q": "test"})
		childHandler.EndToolExec(toolCtx, map[string]any{"found": true}, nil)

		childHandler.EndAgentExecute(childCtx, nil)

		// Direct LLM call on root
		rootLLMCtx := rec.StartLLMCall(rootCtx)
		rec.EndLLMCall(rootLLMCtx, &trace.LLMCallData{InputTokens: 50}, nil)

		rec.EndAgentExecute(rootCtx, nil)

		// Verify trace tree
		rootSpan := rec.Trace().RootSpan
		gt.Equal(t, rootSpan.Kind, trace.SpanKindAgentExecute)
		gt.Equal(t, len(rootSpan.Children), 2) // child agent + LLM call

		// First child: agent_execute (from AsChildAgent)
		childSpan := rootSpan.Children[0]
		gt.Equal(t, childSpan.Kind, trace.SpanKindAgentExecute)
		gt.Equal(t, childSpan.Name, "analyzer")
		gt.Equal(t, len(childSpan.Children), 2) // LLM + tool

		gt.Equal(t, childSpan.Children[0].Kind, trace.SpanKindLLMCall)
		gt.Equal(t, childSpan.Children[1].Kind, trace.SpanKindToolExec)

		// Second child: direct LLM call on root
		gt.Equal(t, rootSpan.Children[1].Kind, trace.SpanKindLLMCall)
	})

	t.Run("SpanKind distinction between AsChildAgent and native SubAgent", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()

		rootCtx := rec.StartAgentExecute(ctx)

		// AsChildAgent child -> SpanKindAgentExecute
		childHandler := trace.AsChildAgent(rec, "task-1")
		childCtx := childHandler.StartAgentExecute(rootCtx)
		childHandler.EndAgentExecute(childCtx, nil)

		// Native SubAgent -> SpanKindSubAgent
		subCtx := rec.StartSubAgent(rootCtx, "searcher")
		rec.EndSubAgent(subCtx, nil)

		rec.EndAgentExecute(rootCtx, nil)

		rootSpan := rec.Trace().RootSpan
		gt.Equal(t, len(rootSpan.Children), 2)
		gt.Equal(t, rootSpan.Children[0].Kind, trace.SpanKindAgentExecute)
		gt.Equal(t, rootSpan.Children[0].Name, "task-1")
		gt.Equal(t, rootSpan.Children[1].Kind, trace.SpanKindSubAgent)
		gt.Equal(t, rootSpan.Children[1].Name, "searcher")
	})
}

func TestAsChildAgentNestedScenarios(t *testing.T) {
	t.Run("AsChildAgent with nested native SubAgent", func(t *testing.T) {
		// Root(agent_execute) -> ChildAgent("task-1", agent_execute) -> SubAgent("searcher", sub_agent) -> LLM Call
		rec := trace.New()
		ctx := context.Background()

		rootCtx := rec.StartAgentExecute(ctx)

		childHandler := trace.AsChildAgent(rec, "task-1")
		childCtx := childHandler.StartAgentExecute(rootCtx)

		// Native SubAgent inside child agent
		subCtx := childHandler.StartSubAgent(childCtx, "searcher")
		llmCtx := childHandler.StartLLMCall(subCtx)
		childHandler.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 10}, nil)
		childHandler.EndSubAgent(subCtx, nil)

		childHandler.EndAgentExecute(childCtx, nil)
		rec.EndAgentExecute(rootCtx, nil)

		rootSpan := rec.Trace().RootSpan
		gt.Equal(t, len(rootSpan.Children), 1)

		childSpan := rootSpan.Children[0]
		gt.Equal(t, childSpan.Kind, trace.SpanKindAgentExecute)
		gt.Equal(t, childSpan.Name, "task-1")
		gt.Equal(t, len(childSpan.Children), 1)

		subSpan := childSpan.Children[0]
		gt.Equal(t, subSpan.Kind, trace.SpanKindSubAgent)
		gt.Equal(t, subSpan.Name, "searcher")
		gt.Equal(t, len(subSpan.Children), 1)
		gt.Equal(t, subSpan.Children[0].Kind, trace.SpanKindLLMCall)
	})

	t.Run("multiple parallel AsChildAgents", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()
		rootCtx := rec.StartAgentExecute(ctx)

		var wg sync.WaitGroup
		for i := 0; i < 3; i++ {
			wg.Add(1)
			name := []string{"task-1", "task-2", "task-3"}[i]
			go func(name string) {
				defer wg.Done()
				childHandler := trace.AsChildAgent(rec, name)
				childCtx := childHandler.StartAgentExecute(rootCtx)
				llmCtx := childHandler.StartLLMCall(childCtx)
				childHandler.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 10}, nil)
				childHandler.EndAgentExecute(childCtx, nil)
			}(name)
		}
		wg.Wait()

		rec.EndAgentExecute(rootCtx, nil)

		rootSpan := rec.Trace().RootSpan
		gt.Equal(t, len(rootSpan.Children), 3)

		// All children should be agent_execute
		for _, child := range rootSpan.Children {
			gt.Equal(t, child.Kind, trace.SpanKindAgentExecute)
			gt.Equal(t, len(child.Children), 1)
			gt.Equal(t, child.Children[0].Kind, trace.SpanKindLLMCall)
		}
	})

	t.Run("AsChildAgent mixed with direct operations", func(t *testing.T) {
		// Root(agent_execute) -> LLM Call
		//                     -> ChildAgent("analyzer", agent_execute) -> [LLM Call, Tool Exec]
		//                     -> Tool Exec
		//                     -> LLM Call
		rec := trace.New()
		ctx := context.Background()

		rootCtx := rec.StartAgentExecute(ctx)

		// Direct LLM call on root
		llm1 := rec.StartLLMCall(rootCtx)
		rec.EndLLMCall(llm1, &trace.LLMCallData{InputTokens: 10}, nil)

		// Child agent via AsChildAgent
		childHandler := trace.AsChildAgent(rec, "analyzer")
		childCtx := childHandler.StartAgentExecute(rootCtx)
		childLLM := childHandler.StartLLMCall(childCtx)
		childHandler.EndLLMCall(childLLM, &trace.LLMCallData{InputTokens: 20}, nil)
		childTool := childHandler.StartToolExec(childCtx, "analyze", nil)
		childHandler.EndToolExec(childTool, nil, nil)
		childHandler.EndAgentExecute(childCtx, nil)

		// Direct tool exec on root
		tool1 := rec.StartToolExec(rootCtx, "summarize", nil)
		rec.EndToolExec(tool1, nil, nil)

		// Direct LLM call on root
		llm2 := rec.StartLLMCall(rootCtx)
		rec.EndLLMCall(llm2, &trace.LLMCallData{InputTokens: 30}, nil)

		rec.EndAgentExecute(rootCtx, nil)

		rootSpan := rec.Trace().RootSpan
		gt.Equal(t, len(rootSpan.Children), 4)
		gt.Equal(t, rootSpan.Children[0].Kind, trace.SpanKindLLMCall)
		gt.Equal(t, rootSpan.Children[1].Kind, trace.SpanKindAgentExecute)
		gt.Equal(t, rootSpan.Children[1].Name, "analyzer")
		gt.Equal(t, rootSpan.Children[2].Kind, trace.SpanKindToolExec)
		gt.Equal(t, rootSpan.Children[3].Kind, trace.SpanKindLLMCall)

		// Child agent children
		childSpan := rootSpan.Children[1]
		gt.Equal(t, len(childSpan.Children), 2)
		gt.Equal(t, childSpan.Children[0].Kind, trace.SpanKindLLMCall)
		gt.Equal(t, childSpan.Children[1].Kind, trace.SpanKindToolExec)
	})

	t.Run("deep nesting with AsChildAgent inside SubAgent", func(t *testing.T) {
		rec := trace.New()
		ctx := context.Background()

		rootCtx := rec.StartAgentExecute(ctx)

		// Level 1: AsChildAgent
		child1Handler := trace.AsChildAgent(rec, "level-1")
		child1Ctx := child1Handler.StartAgentExecute(rootCtx)

		// Level 2: native SubAgent inside child1
		sub2Ctx := child1Handler.StartSubAgent(child1Ctx, "level-2-sub")

		// Level 3: AsChildAgent with a new handler using the same recorder
		child3Handler := trace.AsChildAgent(rec, "level-3")
		child3Ctx := child3Handler.StartAgentExecute(sub2Ctx)

		llmCtx := child3Handler.StartLLMCall(child3Ctx)
		child3Handler.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 5}, nil)
		child3Handler.EndAgentExecute(child3Ctx, nil)

		child1Handler.EndSubAgent(sub2Ctx, nil)
		child1Handler.EndAgentExecute(child1Ctx, nil)
		rec.EndAgentExecute(rootCtx, nil)

		rootSpan := rec.Trace().RootSpan
		gt.Equal(t, len(rootSpan.Children), 1)

		level1 := rootSpan.Children[0]
		gt.Equal(t, level1.Kind, trace.SpanKindAgentExecute)
		gt.Equal(t, level1.Name, "level-1")
		gt.Equal(t, len(level1.Children), 1)

		level2 := level1.Children[0]
		gt.Equal(t, level2.Kind, trace.SpanKindSubAgent)
		gt.Equal(t, level2.Name, "level-2-sub")
		gt.Equal(t, len(level2.Children), 1)

		level3 := level2.Children[0]
		gt.Equal(t, level3.Kind, trace.SpanKindAgentExecute)
		gt.Equal(t, level3.Name, "level-3")
		gt.Equal(t, len(level3.Children), 1)
		gt.Equal(t, level3.Children[0].Kind, trace.SpanKindLLMCall)
	})
}

func TestAsChildAgentWithMultiHandler(t *testing.T) {
	rec1 := trace.New()
	rec2 := trace.New()
	multi := trace.Multi(rec1, rec2)

	ctx := context.Background()
	agentCtx := multi.StartAgentExecute(ctx)

	childHandler := trace.AsChildAgent(multi, "task-1")
	childCtx := childHandler.StartAgentExecute(agentCtx)

	llmCtx := childHandler.StartLLMCall(childCtx)
	childHandler.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 10}, nil)

	childHandler.EndAgentExecute(childCtx, nil)
	multi.EndAgentExecute(agentCtx, nil)

	// Both recorders should have the child agent span
	for _, rec := range []*trace.Recorder{rec1, rec2} {
		tr := rec.Trace()
		gt.Value(t, tr).NotNil()
		gt.Equal(t, len(tr.RootSpan.Children), 1)
		gt.Equal(t, tr.RootSpan.Children[0].Kind, trace.SpanKindAgentExecute)
		gt.Equal(t, tr.RootSpan.Children[0].Name, "task-1")
		gt.Equal(t, len(tr.RootSpan.Children[0].Children), 1)
		gt.Equal(t, tr.RootSpan.Children[0].Children[0].Kind, trace.SpanKindLLMCall)
	}
}

func TestAsChildAgentConcurrentRace(t *testing.T) {
	rec := trace.New()
	ctx := context.Background()
	rootCtx := rec.StartAgentExecute(ctx)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			childHandler := trace.AsChildAgent(rec, "task")
			childCtx := childHandler.StartAgentExecute(rootCtx)

			llmCtx := childHandler.StartLLMCall(childCtx)
			childHandler.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 1}, nil)

			toolCtx := childHandler.StartToolExec(childCtx, "tool", nil)
			childHandler.EndToolExec(toolCtx, nil, nil)

			childHandler.AddEvent(childCtx, "event", nil)

			childHandler.EndAgentExecute(childCtx, nil)
		}(i)
	}
	wg.Wait()

	rec.EndAgentExecute(rootCtx, nil)

	rootSpan := rec.Trace().RootSpan
	gt.Equal(t, len(rootSpan.Children), 10)
	for _, child := range rootSpan.Children {
		gt.Equal(t, child.Kind, trace.SpanKindAgentExecute)
		gt.Equal(t, len(child.Children), 3) // LLM + tool + event
	}
}
