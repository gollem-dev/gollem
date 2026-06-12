package main_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	main "github.com/gollem-dev/gollem/cmd/gollem"
	"github.com/m-mizutani/gt"
)

func TestLocalSourceList(t *testing.T) {
	ctx := context.Background()

	t.Run("list all entries at root", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		resp := gt.R1(src.List(ctx, "", 10, "")).NoError(t)
		// Expect: 1 directory ("sub") + 3 files (trace-001, trace-002, trace-003)
		gt.Equal(t, 4, len(resp.Entries))
		// Directories come first, then files alphabetically.
		gt.Equal(t, "sub", resp.Entries[0].Name)
		gt.Equal(t, main.EntryKindDir, resp.Entries[0].Kind)
		gt.Equal(t, "trace-001", resp.Entries[1].Name)
		gt.Equal(t, main.EntryKindFile, resp.Entries[1].Kind)
		gt.Equal(t, "trace-002", resp.Entries[2].Name)
		gt.Equal(t, "trace-003", resp.Entries[3].Name)
		gt.Equal(t, "", resp.NextPageToken)
	})

	t.Run("list entries inside subdirectory", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		resp := gt.R1(src.List(ctx, "sub", 10, "")).NoError(t)
		gt.Equal(t, 1, len(resp.Entries))
		gt.Equal(t, "trace-sub-001", resp.Entries[0].Name)
		gt.Equal(t, main.EntryKindFile, resp.Entries[0].Kind)
	})

	t.Run("pagination first page", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		resp := gt.R1(src.List(ctx, "", 2, "")).NoError(t)
		gt.Equal(t, 2, len(resp.Entries))
		gt.Equal(t, "sub", resp.Entries[0].Name)
		gt.Equal(t, "trace-001", resp.Entries[1].Name)
		gt.True(t, resp.NextPageToken != "")
	})

	t.Run("pagination second page", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		resp1 := gt.R1(src.List(ctx, "", 2, "")).NoError(t)

		resp2 := gt.R1(src.List(ctx, "", 2, resp1.NextPageToken)).NoError(t)
		gt.Equal(t, 2, len(resp2.Entries))
		gt.Equal(t, "trace-002", resp2.Entries[0].Name)
		gt.Equal(t, "trace-003", resp2.Entries[1].Name)
		gt.Equal(t, "", resp2.NextPageToken)
	})

	t.Run("empty directory", func(t *testing.T) {
		dir := t.TempDir()
		src := main.NewLocalSource(dir)
		resp := gt.R1(src.List(ctx, "", 10, "")).NoError(t)
		gt.Equal(t, 0, len(resp.Entries))
	})

	t.Run("non-existent directory", func(t *testing.T) {
		src := main.NewLocalSource("/nonexistent")
		_, err := src.List(ctx, "", 10, "")
		gt.Error(t, err)
	})

	t.Run("default page size", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		resp := gt.R1(src.List(ctx, "", 0, "")).NoError(t)
		gt.Equal(t, 4, len(resp.Entries))
	})

	t.Run("rejects path with parent traversal", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		_, err := src.List(ctx, "../etc", 10, "")
		gt.Error(t, err)
	})

	t.Run("rejects absolute path", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		_, err := src.List(ctx, "/etc", 10, "")
		gt.Error(t, err)
	})
}

func TestLocalSourceGet(t *testing.T) {
	ctx := context.Background()

	t.Run("get existing trace at root", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		tr := gt.R1(src.Get(ctx, "trace-001")).NoError(t)
		gt.Equal(t, "trace-001", tr.TraceID)
		gt.Equal(t, "gpt-4", tr.Metadata.Model)
		gt.NotEqual(t, nil, tr.RootSpan)
		gt.Equal(t, 2, len(tr.RootSpan.Children))
	})

	t.Run("get trace inside subdirectory", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		tr := gt.R1(src.Get(ctx, "sub/trace-sub-001")).NoError(t)
		gt.Equal(t, "trace-sub-001", tr.TraceID)
	})

	t.Run("get non-existent trace", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		_, err := src.Get(ctx, "non-existent")
		gt.Error(t, err)
	})

	t.Run("get trace with error status", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		tr := gt.R1(src.Get(ctx, "trace-002")).NoError(t)
		gt.Equal(t, "trace-002", tr.TraceID)
		gt.Equal(t, "error", string(tr.RootSpan.Status))
		gt.Equal(t, "tool execution failed", tr.RootSpan.Error)
	})

	t.Run("invalid json file", func(t *testing.T) {
		dir := t.TempDir()
		err := os.WriteFile(filepath.Join(dir, "bad.json"), []byte("not json"), 0644)
		gt.NoError(t, err)

		src := main.NewLocalSource(dir)
		_, err = src.Get(ctx, "bad")
		gt.Error(t, err)
	})

	t.Run("rejects parent traversal in trace path", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		_, err := src.Get(ctx, "../etc/passwd")
		gt.Error(t, err)
	})

	t.Run("rejects empty path", func(t *testing.T) {
		src := main.NewLocalSource("testdata")
		_, err := src.Get(ctx, "")
		gt.Error(t, err)
	})
}

func TestCleanRelativePath(t *testing.T) {
	t.Run("empty is valid root", func(t *testing.T) {
		got := gt.R1(main.CleanRelativePath("")).NoError(t)
		gt.Equal(t, "", got)
	})

	t.Run("simple relative path", func(t *testing.T) {
		got := gt.R1(main.CleanRelativePath("foo/bar")).NoError(t)
		gt.Equal(t, "foo/bar", got)
	})

	t.Run("trailing slash trimmed", func(t *testing.T) {
		got := gt.R1(main.CleanRelativePath("foo/bar/")).NoError(t)
		gt.Equal(t, "foo/bar", got)
	})

	t.Run("absolute path rejected", func(t *testing.T) {
		_, err := main.CleanRelativePath("/foo")
		gt.Error(t, err)
	})

	t.Run("parent reference rejected", func(t *testing.T) {
		_, err := main.CleanRelativePath("foo/../bar")
		gt.Error(t, err)
	})

	t.Run("dot segment rejected", func(t *testing.T) {
		_, err := main.CleanRelativePath("./foo")
		gt.Error(t, err)
	})

	t.Run("backslash rejected", func(t *testing.T) {
		_, err := main.CleanRelativePath("foo\\bar")
		gt.Error(t, err)
	})

	t.Run("consecutive slashes rejected", func(t *testing.T) {
		_, err := main.CleanRelativePath("foo//bar")
		gt.Error(t, err)
	})
}
