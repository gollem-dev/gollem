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

func TestFileRepositoryTraceIDAsFileName(t *testing.T) {
	type testCase struct {
		traceID string
		valid   bool
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			// The trace directory is nested so that a "../" path would land
			// in parent, which the test can inspect.
			parent := t.TempDir()
			dir := filepath.Join(parent, "traces")
			repo := trace.NewFileRepository(dir)

			err := repo.Save(context.Background(), &trace.Trace{TraceID: tc.traceID})

			if tc.valid {
				gt.NoError(t, err)
				_, err := os.Stat(filepath.Join(dir, tc.traceID+".json"))
				gt.NoError(t, err)
				return
			}

			gt.Error(t, err)
			var written []string
			walkErr := filepath.WalkDir(parent, func(path string, d os.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if !d.IsDir() {
					written = append(written, path)
				}
				return nil
			})
			gt.NoError(t, walkErr)
			gt.A(t, written).Length(0)
		}
	}

	t.Run("parent directory traversal is rejected", runTest(testCase{traceID: "../escape"}))
	t.Run("deep parent directory traversal is rejected", runTest(testCase{traceID: "../../config/app"}))
	t.Run("subdirectory is rejected", runTest(testCase{traceID: "sub/trace"}))
	t.Run("backslash is rejected", runTest(testCase{traceID: `..\escape`}))
	t.Run("absolute path is rejected", runTest(testCase{traceID: "/tmp/escape"}))
	t.Run("empty trace ID is rejected", runTest(testCase{traceID: ""}))
	t.Run("UUID is accepted", runTest(testCase{traceID: "0192f6c4-7b5e-7c3a-9d2e-1a2b3c4d5e6f", valid: true}))
	t.Run("dots inside the trace ID are accepted", runTest(testCase{traceID: "req..v1.2", valid: true}))
}

func TestFileRepositorySymlink(t *testing.T) {
	type testCase struct {
		// linkTarget returns the symbolic link target for a link created in
		// parent/traces, where parent/outside/target.json and
		// parent/traces/inside/target.json already exist.
		linkTarget func(parent string) string
		valid      bool
	}

	runTest := func(tc testCase) func(t *testing.T) {
		return func(t *testing.T) {
			parent := t.TempDir()
			dir := filepath.Join(parent, "traces")
			outside := filepath.Join(parent, "outside", "target.json")
			inside := filepath.Join(dir, "inside", "target.json")
			for _, p := range []string{outside, inside} {
				gt.NoError(t, os.MkdirAll(filepath.Dir(p), 0750))
				gt.NoError(t, os.WriteFile(p, []byte("original"), 0600))
			}
			gt.NoError(t, os.Symlink(tc.linkTarget(parent), filepath.Join(dir, "linked.json")))

			repo := trace.NewFileRepository(dir)
			err := repo.Save(context.Background(), &trace.Trace{TraceID: "linked"})

			outsideData, readErr := os.ReadFile(outside)
			gt.NoError(t, readErr)
			gt.Equal(t, string(outsideData), "original")

			if !tc.valid {
				gt.Error(t, err)
				return
			}

			gt.NoError(t, err)
			var loaded trace.Trace
			insideData, readErr := os.ReadFile(inside)
			gt.NoError(t, readErr)
			gt.NoError(t, json.Unmarshal(insideData, &loaded))
			gt.Equal(t, loaded.TraceID, "linked")
		}
	}

	t.Run("absolute link outside the directory is rejected", runTest(testCase{
		linkTarget: func(parent string) string { return filepath.Join(parent, "outside", "target.json") },
	}))
	t.Run("relative link outside the directory is rejected", runTest(testCase{
		linkTarget: func(string) string { return filepath.Join("..", "outside", "target.json") },
	}))
	t.Run("relative link inside the directory is followed", runTest(testCase{
		linkTarget: func(string) string { return filepath.Join("inside", "target.json") },
		valid:      true,
	}))
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
