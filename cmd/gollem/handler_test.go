package main_test

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	main "github.com/gollem-dev/gollem/cmd/gollem"
	"github.com/m-mizutani/gt"
)

func TestHandleHealth(t *testing.T) {
	src := main.NewLocalSource("testdata")
	s := main.NewServer(main.WithTestSource(src))

	req := httptest.NewRequest(http.MethodGet, "/api/health", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)

	gt.Equal(t, http.StatusOK, rec.Code)

	var resp map[string]string
	gt.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	gt.Equal(t, "ok", resp["status"])
}

func TestHandleListTraces(t *testing.T) {
	src := main.NewLocalSource("testdata")
	s := main.NewServer(main.WithTestSource(src))

	t.Run("list root entries", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/traces", nil)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		gt.Equal(t, http.StatusOK, rec.Code)

		var resp main.ListEntriesResponse
		gt.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
		// 1 directory ("sub") + 3 files
		gt.Equal(t, 4, len(resp.Entries))
		gt.Equal(t, "", resp.Path)
		gt.Equal(t, main.EntryKindDir, resp.Entries[0].Kind)
		gt.Equal(t, "sub", resp.Entries[0].Name)
	})

	t.Run("list inside subdirectory", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/traces?path=sub", nil)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		gt.Equal(t, http.StatusOK, rec.Code)

		var resp main.ListEntriesResponse
		gt.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
		gt.Equal(t, 1, len(resp.Entries))
		gt.Equal(t, "sub", resp.Path)
		gt.Equal(t, "trace-sub-001", resp.Entries[0].Name)
		gt.Equal(t, main.EntryKindFile, resp.Entries[0].Kind)
	})

	t.Run("with page size", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/traces?page_size=2", nil)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		gt.Equal(t, http.StatusOK, rec.Code)

		var resp main.ListEntriesResponse
		gt.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
		gt.Equal(t, 2, len(resp.Entries))
		gt.True(t, resp.NextPageToken != "")
	})

	t.Run("invalid page size", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/traces?page_size=abc", nil)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		gt.Equal(t, http.StatusBadRequest, rec.Code)
	})

	t.Run("path with parent traversal rejected", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/traces?path=../etc", nil)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		gt.Equal(t, http.StatusBadRequest, rec.Code)
	})

	t.Run("absolute path rejected", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/traces?path=/etc", nil)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		gt.Equal(t, http.StatusBadRequest, rec.Code)
	})
}

func TestHandleGetTrace(t *testing.T) {
	src := main.NewLocalSource("testdata")
	s := main.NewServer(main.WithTestSource(src))

	t.Run("get existing trace at root", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/traces/trace-001", nil)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		gt.Equal(t, http.StatusOK, rec.Code)

		var resp map[string]any
		gt.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
		gt.Equal(t, "trace-001", resp["trace_id"])
	})

	t.Run("get trace inside subdirectory", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/traces/sub/trace-sub-001", nil)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		gt.Equal(t, http.StatusOK, rec.Code)

		var resp map[string]any
		gt.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
		gt.Equal(t, "trace-sub-001", resp["trace_id"])
	})

	t.Run("get non-existent trace", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/traces/nonexistent", nil)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		gt.Equal(t, http.StatusNotFound, rec.Code)
	})

	t.Run("path with parent traversal rejected", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/traces/../etc/passwd", nil)
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)

		// http.ServeMux normalizes "../" so the request is redirected before reaching the
		// handler. This is acceptable because the redirect target is harmless. We only
		// require that no 200 is returned.
		gt.True(t, rec.Code != http.StatusOK)
	})
}
