package main

import (
	"context"
	"strings"
	"time"

	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
)

// entryKind distinguishes between files and directories in a listing.
type entryKind string

const (
	entryKindFile entryKind = "file"
	entryKindDir  entryKind = "dir"
)

// entrySummary represents a single entry (file or directory) in a listing.
// For files, Name is the trace ID (without ".json" suffix) and Size/UpdatedAt are populated.
// For directories, Name is the directory name and Size/UpdatedAt are zero.
type entrySummary struct {
	Name      string    `json:"name"`
	Kind      entryKind `json:"kind"`
	Size      int64     `json:"size,omitempty"`
	UpdatedAt time.Time `json:"updated_at,omitzero"`
}

type listRequest struct {
	// path is the relative path from the virtual root.
	// "/" separated, no leading or trailing slash. Empty means the virtual root itself.
	path      string
	pageSize  int
	pageToken string
}

type listResponse struct {
	entries       []entrySummary
	nextPageToken string
}

// traceSource provides access to trace data from various backends.
type traceSource interface {
	// List returns entries (files and subdirectories) under the given path.
	// path is relative to the virtual root configured at construction time.
	List(ctx context.Context, req listRequest) (*listResponse, error)
	// Get returns the trace stored at the given path.
	// path is the relative path from the virtual root, without ".json" suffix.
	Get(ctx context.Context, path string) (*trace.Trace, error)
}

// cleanRelativePath validates that path is a safe relative path under the
// virtual root and returns its canonical form: forward-slash separated, no
// leading/trailing slash, no "." or ".." segments, no empty segments.
// Empty input is valid and returned as-is, representing the virtual root.
func cleanRelativePath(path string) (string, error) {
	if path == "" {
		return "", nil
	}
	// Reject absolute paths (Unix-style or Windows-style backslashes).
	if strings.HasPrefix(path, "/") || strings.HasPrefix(path, "\\") {
		return "", goerr.New("path must not be absolute", goerr.Value("path", path))
	}
	// Reject backslashes outright; we only accept forward slashes.
	if strings.Contains(path, "\\") {
		return "", goerr.New("path must not contain backslashes", goerr.Value("path", path))
	}
	// Trim a single trailing slash for convenience but reject if it doubles up.
	trimmed := strings.TrimSuffix(path, "/")
	if strings.HasSuffix(trimmed, "/") {
		return "", goerr.New("path must not contain consecutive slashes", goerr.Value("path", path))
	}
	parts := strings.Split(trimmed, "/")
	for _, p := range parts {
		switch p {
		case "", ".", "..":
			return "", goerr.New("path must not contain empty, '.' or '..' segments", goerr.Value("path", path))
		}
	}
	return strings.Join(parts, "/"), nil
}
