package trace_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/gt"
)

func TestFileRepositorySave(t *testing.T) {
	dir := t.TempDir()
	repo := trace.NewFileRepository(dir)

	now := time.Now()
	tr := &trace.Trace{
		TraceID: "test-file-repo",
		RootSpan: &trace.Span{
			SpanID:    "root",
			Kind:      trace.SpanKindAgentExecute,
			Name:      "agent_execute",
			StartedAt: now,
			EndedAt:   now.Add(time.Second),
			Duration:  time.Second,
			Status:    trace.SpanStatusOK,
		},
		Metadata: trace.TraceMetadata{
			Model: "test-model",
		},
		StartedAt: now,
		EndedAt:   now.Add(time.Second),
	}

	err := repo.Save(context.Background(), tr)
	gt.NoError(t, err)

	// Verify file exists and content is valid JSON
	filePath := filepath.Join(dir, "test-file-repo.json")
	data, err := os.ReadFile(filePath)
	gt.NoError(t, err)

	var loaded trace.Trace
	err = json.Unmarshal(data, &loaded)
	gt.NoError(t, err)

	gt.Equal(t, loaded.TraceID, "test-file-repo")
	gt.Equal(t, loaded.RootSpan.Kind, trace.SpanKindAgentExecute)
	gt.Equal(t, loaded.Metadata.Model, "test-model")
}

func TestFileRepositoryCreatesDirectory(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nested", "dir")
	repo := trace.NewFileRepository(dir)

	now := time.Now()
	tr := &trace.Trace{
		TraceID: "test-nested-dir",
		RootSpan: &trace.Span{
			SpanID:    "root",
			Kind:      trace.SpanKindAgentExecute,
			Name:      "agent_execute",
			StartedAt: now,
			EndedAt:   now.Add(time.Second),
			Duration:  time.Second,
			Status:    trace.SpanStatusOK,
		},
		StartedAt: now,
		EndedAt:   now.Add(time.Second),
	}

	err := repo.Save(context.Background(), tr)
	gt.NoError(t, err)

	// Verify directory was created and file exists
	filePath := filepath.Join(dir, "test-nested-dir.json")
	_, err = os.Stat(filePath)
	gt.NoError(t, err)
}

func TestFileRepositoryWithChildren(t *testing.T) {
	dir := t.TempDir()
	repo := trace.NewFileRepository(dir)

	now := time.Now()
	tr := &trace.Trace{
		TraceID: "test-with-children",
		RootSpan: &trace.Span{
			SpanID:    "root",
			Kind:      trace.SpanKindAgentExecute,
			Name:      "agent_execute",
			StartedAt: now,
			EndedAt:   now.Add(2 * time.Second),
			Duration:  2 * time.Second,
			Status:    trace.SpanStatusOK,
			Children: []*trace.Span{
				{
					SpanID:    "llm-1",
					ParentID:  "root",
					Kind:      trace.SpanKindLLMCall,
					Name:      "llm_call",
					StartedAt: now,
					EndedAt:   now.Add(time.Second),
					Duration:  time.Second,
					Status:    trace.SpanStatusOK,
					LLMCall: &trace.LLMCallData{
						InputTokens:  200,
						OutputTokens: 100,
						Model:        "claude-3",
						Request: &trace.LLMRequest{
							SystemPrompt: "You are helpful.",
							Messages:     []trace.Message{{Role: "user", Contents: []trace.MessageContent{trace.NewTextContent("hello")}}},
							Tools:        []trace.ToolSpec{{Name: "search", Description: "Search tool"}},
						},
						Response: &trace.LLMResponse{
							Texts: []string{"Hi there!"},
							FunctionCalls: []*trace.FunctionCall{
								{ID: "call-1", Name: "search", Arguments: map[string]any{"q": "test"}},
							},
						},
					},
				},
				{
					SpanID:    "tool-1",
					ParentID:  "root",
					Kind:      trace.SpanKindToolExec,
					Name:      "search",
					StartedAt: now.Add(time.Second),
					EndedAt:   now.Add(2 * time.Second),
					Duration:  time.Second,
					Status:    trace.SpanStatusOK,
					ToolExec: &trace.ToolExecData{
						ToolName: "search",
						Args:     map[string]any{"q": "test"},
						Result:   map[string]any{"items": []any{"result1"}},
					},
				},
			},
		},
		StartedAt: now,
		EndedAt:   now.Add(2 * time.Second),
	}

	err := repo.Save(context.Background(), tr)
	gt.NoError(t, err)

	// Read back and verify
	data, err := os.ReadFile(filepath.Join(dir, "test-with-children.json"))
	gt.NoError(t, err)

	var loaded trace.Trace
	err = json.Unmarshal(data, &loaded)
	gt.NoError(t, err)

	gt.Equal(t, len(loaded.RootSpan.Children), 2)
	gt.Equal(t, loaded.RootSpan.Children[0].LLMCall.InputTokens, 200)
	gt.Equal(t, loaded.RootSpan.Children[0].LLMCall.Request.SystemPrompt, "You are helpful.")
	gt.Equal(t, loaded.RootSpan.Children[1].ToolExec.ToolName, "search")
}
