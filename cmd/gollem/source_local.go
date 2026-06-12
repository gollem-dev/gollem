package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
)

type localSource struct {
	dir string
}

func newLocalSource(dir string) traceSource {
	return &localSource{dir: dir}
}

func (s *localSource) List(ctx context.Context, req listRequest) (*listResponse, error) {
	rel, err := cleanRelativePath(req.path)
	if err != nil {
		return nil, goerr.Wrap(err, "invalid path")
	}

	// os.Root scopes directory access to s.dir, matching the defense-in-depth
	// approach used by Get.
	root, err := os.OpenRoot(s.dir)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to open root", goerr.Value("dir", s.dir))
	}
	defer func() { _ = root.Close() }()

	relDir := "."
	if rel != "" {
		relDir = filepath.FromSlash(rel)
	}
	dir, err := root.Open(relDir)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to open directory", goerr.Value("path", rel))
	}
	entries, err := dir.ReadDir(-1)
	_ = dir.Close()
	if err != nil {
		return nil, goerr.Wrap(err, "failed to read directory", goerr.Value("path", rel))
	}

	type fileEntry struct {
		name string
		kind entryKind
		info os.FileInfo
	}
	var collected []fileEntry
	for _, e := range entries {
		// Skip hidden entries.
		if strings.HasPrefix(e.Name(), ".") {
			continue
		}
		if e.IsDir() {
			collected = append(collected, fileEntry{
				name: e.Name(),
				kind: entryKindDir,
			})
			continue
		}
		if !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		info, err := e.Info()
		if err != nil {
			continue
		}
		collected = append(collected, fileEntry{
			name: strings.TrimSuffix(e.Name(), ".json"),
			kind: entryKindFile,
			info: info,
		})
	}

	// Sort: directories first, then files; each group alphabetically.
	sort.Slice(collected, func(i, j int) bool {
		if collected[i].kind != collected[j].kind {
			return collected[i].kind == entryKindDir
		}
		return collected[i].name < collected[j].name
	})

	startIdx := 0
	if req.pageToken != "" {
		lastKey, err := decodePageToken(req.pageToken)
		if err != nil {
			return nil, goerr.Wrap(err, "invalid page token")
		}
		startIdx = sort.Search(len(collected), func(i int) bool {
			return entrySortKey(collected[i].kind, collected[i].name) > lastKey
		})
		if startIdx >= len(collected) {
			return &listResponse{}, nil
		}
	}

	pageSize := req.pageSize
	if pageSize <= 0 {
		pageSize = 20
	}

	endIdx := min(startIdx+pageSize, len(collected))

	resp := &listResponse{}
	for _, f := range collected[startIdx:endIdx] {
		entry := entrySummary{
			Name: f.name,
			Kind: f.kind,
		}
		if f.kind == entryKindFile && f.info != nil {
			entry.Size = f.info.Size()
			entry.UpdatedAt = f.info.ModTime()
		}
		resp.entries = append(resp.entries, entry)
	}

	if endIdx < len(collected) {
		last := collected[endIdx-1]
		resp.nextPageToken = encodePageToken(entrySortKey(last.kind, last.name))
	}

	return resp, nil
}

func (s *localSource) Get(ctx context.Context, path string) (*trace.Trace, error) {
	rel, err := cleanRelativePath(path)
	if err != nil {
		return nil, goerr.Wrap(err, "invalid path")
	}
	if rel == "" {
		return nil, goerr.New("trace path is required")
	}

	// os.Root scopes file access to s.dir, preventing any directory traversal
	// even if cleanRelativePath were ever bypassed.
	root, err := os.OpenRoot(s.dir)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to open root", goerr.Value("dir", s.dir))
	}
	defer func() { _ = root.Close() }()

	relFile := filepath.FromSlash(rel) + ".json"
	f, err := root.Open(relFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, goerr.Wrap(err, "trace not found", goerr.Value("path", rel))
		}
		return nil, goerr.Wrap(err, "failed to open trace file", goerr.Value("path", rel))
	}
	defer func() { _ = f.Close() }()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to read trace file", goerr.Value("path", rel))
	}

	var t trace.Trace
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, goerr.Wrap(err, "failed to parse trace file", goerr.Value("path", rel))
	}

	return &t, nil
}

// entrySortKey produces a sort key matching the List ordering: directories first.
func entrySortKey(kind entryKind, name string) string {
	if kind == entryKindDir {
		return "0:" + name
	}
	return "1:" + name
}

func encodePageToken(key string) string {
	return base64.URLEncoding.EncodeToString([]byte(key))
}

func decodePageToken(token string) (string, error) {
	b, err := base64.URLEncoding.DecodeString(token)
	if err != nil {
		return "", goerr.Wrap(err, "failed to decode page token")
	}
	return string(b), nil
}
