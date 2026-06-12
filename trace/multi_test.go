package trace_test

import (
	"context"
	"errors"
	"testing"

	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/gt"
)

func TestMultiHandlerFanOut(t *testing.T) {
	rec1 := trace.New()
	rec2 := trace.New()
	multi := trace.Multi(rec1, rec2)

	ctx := context.Background()
	agentCtx := multi.StartAgentExecute(ctx)

	llmCtx := multi.StartLLMCall(agentCtx)
	multi.EndLLMCall(llmCtx, &trace.LLMCallData{InputTokens: 10}, nil)

	multi.AddEvent(agentCtx, "test_event", nil)

	multi.EndAgentExecute(agentCtx, nil)

	// Both recorders should have the same structure
	for _, rec := range []*trace.Recorder{rec1, rec2} {
		tr := rec.Trace()
		gt.Value(t, tr).NotNil()
		gt.Equal(t, len(tr.RootSpan.Children), 2) // llm_call + event
	}
}

func TestMultiHandlerToolExec(t *testing.T) {
	rec1 := trace.New()
	rec2 := trace.New()
	multi := trace.Multi(rec1, rec2)

	ctx := context.Background()
	agentCtx := multi.StartAgentExecute(ctx)
	toolCtx := multi.StartToolExec(agentCtx, "search", map[string]any{"q": "test"})
	multi.EndToolExec(toolCtx, map[string]any{"found": true}, nil)
	multi.EndAgentExecute(agentCtx, nil)

	for _, rec := range []*trace.Recorder{rec1, rec2} {
		tr := rec.Trace()
		gt.Value(t, tr).NotNil()
		gt.Equal(t, len(tr.RootSpan.Children), 1)
		gt.Equal(t, tr.RootSpan.Children[0].Kind, trace.SpanKindToolExec)
	}
}

func TestMultiHandlerSubAgent(t *testing.T) {
	rec1 := trace.New()
	rec2 := trace.New()
	multi := trace.Multi(rec1, rec2)

	ctx := context.Background()
	agentCtx := multi.StartAgentExecute(ctx)
	subCtx := multi.StartSubAgent(agentCtx, "child")
	multi.EndSubAgent(subCtx, nil)
	multi.EndAgentExecute(agentCtx, nil)

	for _, rec := range []*trace.Recorder{rec1, rec2} {
		tr := rec.Trace()
		gt.Value(t, tr).NotNil()
		gt.Equal(t, len(tr.RootSpan.Children), 1)
		gt.Equal(t, tr.RootSpan.Children[0].Kind, trace.SpanKindSubAgent)
	}
}

type failingFinishHandler struct {
	trace.Recorder
}

func (f *failingFinishHandler) Finish(_ context.Context) error {
	return errors.New("finish failed")
}

func TestMultiHandlerFinishCollectsErrors(t *testing.T) {
	rec := trace.New()
	failing := &failingFinishHandler{}
	multi := trace.Multi(rec, failing)

	err := multi.Finish(context.Background())
	gt.Value(t, err).NotNil()
	gt.B(t, errors.Is(err, errors.New("finish failed"))).False() // errors.Join wraps them
	gt.S(t, err.Error()).Contains("finish failed")
}

func TestMultiHandlerFinishNoErrors(t *testing.T) {
	rec1 := trace.New()
	rec2 := trace.New()
	multi := trace.Multi(rec1, rec2)

	err := multi.Finish(context.Background())
	gt.NoError(t, err)
}
