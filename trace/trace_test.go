package trace_test

import (
	"testing"
	"time"

	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/gt"
)

func TestSpanKindValues(t *testing.T) {
	gt.Equal(t, trace.SpanKindAgentExecute, trace.SpanKind("agent_execute"))
	gt.Equal(t, trace.SpanKindLLMCall, trace.SpanKind("llm_call"))
	gt.Equal(t, trace.SpanKindToolExec, trace.SpanKind("tool_exec"))
	gt.Equal(t, trace.SpanKindSubAgent, trace.SpanKind("sub_agent"))
	gt.Equal(t, trace.SpanKindEvent, trace.SpanKind("event"))
}

func TestSpanStatusValues(t *testing.T) {
	gt.Equal(t, trace.SpanStatusOK, trace.SpanStatus("ok"))
	gt.Equal(t, trace.SpanStatusError, trace.SpanStatus("error"))
}

func TestTraceStructure(t *testing.T) {
	now := time.Now()
	tr := &trace.Trace{
		TraceID: "test-trace-id",
		RootSpan: &trace.Span{
			SpanID:    "root-span-id",
			Kind:      trace.SpanKindAgentExecute,
			Name:      "agent_execute",
			StartedAt: now,
			EndedAt:   now.Add(time.Second),
			Duration:  time.Second,
			Status:    trace.SpanStatusOK,
			Children: []*trace.Span{
				{
					SpanID:    "llm-span-id",
					ParentID:  "root-span-id",
					Kind:      trace.SpanKindLLMCall,
					Name:      "llm_call",
					StartedAt: now,
					EndedAt:   now.Add(500 * time.Millisecond),
					Duration:  500 * time.Millisecond,
					Status:    trace.SpanStatusOK,
					LLMCall: &trace.LLMCallData{
						InputTokens:  100,
						OutputTokens: 50,
						Model:        "test-model",
					},
				},
				{
					SpanID:    "tool-span-id",
					ParentID:  "root-span-id",
					Kind:      trace.SpanKindToolExec,
					Name:      "search",
					StartedAt: now.Add(500 * time.Millisecond),
					EndedAt:   now.Add(800 * time.Millisecond),
					Duration:  300 * time.Millisecond,
					Status:    trace.SpanStatusOK,
					ToolExec: &trace.ToolExecData{
						ToolName: "search",
						Args:     map[string]any{"query": "test"},
						Result:   map[string]any{"found": true},
					},
				},
			},
		},
		Metadata: trace.TraceMetadata{
			Model:    "test-model",
			Strategy: "simple",
			Labels:   map[string]string{"env": "test"},
		},
		StartedAt: now,
		EndedAt:   now.Add(time.Second),
	}

	gt.Equal(t, tr.TraceID, "test-trace-id")
	gt.Equal(t, len(tr.RootSpan.Children), 2)
	gt.Equal(t, tr.RootSpan.Children[0].Kind, trace.SpanKindLLMCall)
	gt.Equal(t, tr.RootSpan.Children[0].LLMCall.InputTokens, 100)
	gt.Equal(t, tr.RootSpan.Children[1].Kind, trace.SpanKindToolExec)
	gt.Equal(t, tr.RootSpan.Children[1].ToolExec.ToolName, "search")
}

func TestEventData(t *testing.T) {
	type customEvent struct {
		Goal string `json:"goal"`
	}

	span := &trace.Span{
		SpanID: "event-span-id",
		Kind:   trace.SpanKindEvent,
		Name:   "plan_created",
		Status: trace.SpanStatusOK,
		Event: &trace.EventData{
			Kind: "plan_created",
			Data: &customEvent{Goal: "implement feature"},
		},
	}

	gt.Equal(t, span.Event.Kind, "plan_created")
	data, ok := span.Event.Data.(*customEvent)
	gt.B(t, ok).True()
	gt.Equal(t, data.Goal, "implement feature")
}
