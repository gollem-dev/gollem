package trace

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"

	"github.com/m-mizutani/goerr/v2"
)

// Repository is the interface for persisting trace data.
type Repository interface {
	Save(ctx context.Context, trace *Trace) error
}

// FileRepository persists trace data as JSON files.
type FileRepository struct {
	dir string
}

// NewFileRepository creates a new FileRepository that writes to the given directory.
func NewFileRepository(dir string) *FileRepository {
	return &FileRepository{dir: dir}
}

// Save writes the trace as JSON to {dir}/{trace_id}.json.
// The trace ID must be usable as a single file name in dir: it must not be
// empty, must not contain a path separator, and must be accepted by
// filepath.IsLocal. Otherwise Save returns an error and writes nothing.
func (r *FileRepository) Save(_ context.Context, trace *Trace) error {
	fileName, err := traceFileName(trace.TraceID)
	if err != nil {
		return err
	}

	if err := os.MkdirAll(r.dir, 0750); err != nil {
		return goerr.Wrap(err, "failed to create trace directory", goerr.V("dir", r.dir))
	}

	data, err := json.MarshalIndent(trace, "", "  ")
	if err != nil {
		return goerr.Wrap(err, "failed to marshal trace")
	}

	// os.Root also rejects a symbolic link in dir that points outside of it,
	// which validating the file name alone cannot detect.
	root, err := os.OpenRoot(r.dir)
	if err != nil {
		return goerr.Wrap(err, "failed to open trace directory", goerr.V("dir", r.dir))
	}
	writeErr := root.WriteFile(fileName, data, 0600)
	closeErr := root.Close()
	if writeErr != nil {
		return goerr.Wrap(writeErr, "failed to write trace file", goerr.V("dir", r.dir), goerr.V("file", fileName))
	}
	if closeErr != nil {
		return goerr.Wrap(closeErr, "failed to close trace directory", goerr.V("dir", r.dir))
	}

	return nil
}

// traceFileName returns the file name for traceID. The trace ID can come from
// an untrusted source such as an HTTP request header, so it is rejected
// unless it forms a single local file name.
func traceFileName(traceID string) (string, error) {
	fileName := traceID + ".json"
	if traceID == "" || strings.ContainsAny(traceID, `/\`) || !filepath.IsLocal(fileName) {
		return "", goerr.New("trace ID cannot be used as a file name", goerr.V("trace_id", traceID))
	}
	return fileName, nil
}
