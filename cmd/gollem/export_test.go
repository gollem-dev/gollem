package main

import (
	"context"
	"net/http"

	"github.com/gollem-dev/gollem/trace"
)

// ListEntriesResponse is exported for testing.
type ListEntriesResponse = listEntriesResponse

// Exported constructors for testing
var NewServer = newServer

// Exported server options for testing
var WithSource = withSource
var WithAddr = withAddr
var WithNoBrowser = withNoBrowser

// Handler returns the server's HTTP handler for testing.
func (s *server) Handler() http.Handler {
	return s.handler()
}

// EntrySummaryExported is an exported version of entrySummary for testing.
type EntrySummaryExported = entrySummary

// EntryKindExported is an exported alias of entryKind for testing.
type EntryKindExported = entryKind

const (
	EntryKindFile = entryKindFile
	EntryKindDir  = entryKindDir
)

// ListResult holds the exported result of a List call.
type ListResult struct {
	Entries       []EntrySummaryExported
	NextPageToken string
}

// TestableSource wraps a traceSource for external test access.
type TestableSource struct {
	src traceSource
}

// NewLocalSource creates a TestableSource backed by localSource.
func NewLocalSource(dir string) *TestableSource {
	return &TestableSource{src: newLocalSource(dir)}
}

// List calls the underlying source's List with exported types.
func (ts *TestableSource) List(ctx context.Context, path string, pageSize int, pageToken string) (*ListResult, error) {
	resp, err := ts.src.List(ctx, listRequest{
		path:      path,
		pageSize:  pageSize,
		pageToken: pageToken,
	})
	if err != nil {
		return nil, err
	}
	return &ListResult{
		Entries:       resp.entries,
		NextPageToken: resp.nextPageToken,
	}, nil
}

// Get calls the underlying source's Get.
func (ts *TestableSource) Get(ctx context.Context, path string) (*trace.Trace, error) {
	return ts.src.Get(ctx, path)
}

// AsTraceSource returns the underlying traceSource for use with server options.
func (ts *TestableSource) AsTraceSource() traceSource {
	return ts.src
}

// WithTestSource creates a server option from a TestableSource.
func WithTestSource(ts *TestableSource) serverOption {
	return withSource(ts.src)
}

// ParseGSURI is exported for testing.
var ParseGSURI = parseGSURI

// CleanRelativePath is exported for testing.
var CleanRelativePath = cleanRelativePath
