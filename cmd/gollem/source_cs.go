package main

import (
	"context"
	"encoding/json"
	"io"
	"sort"
	"strings"
	"time"

	"cloud.google.com/go/storage"
	"github.com/gollem-dev/gollem/trace"
	"github.com/m-mizutani/goerr/v2"
	"google.golang.org/api/iterator"
)

type csSource struct {
	bucket string
	prefix string
	client *storage.Client
}

func newCSSource(ctx context.Context, bucket, prefix string) (traceSource, error) {
	client, err := storage.NewClient(ctx)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to create Cloud Storage client")
	}
	return &csSource{
		bucket: bucket,
		prefix: prefix,
		client: client,
	}, nil
}

func (s *csSource) List(ctx context.Context, req listRequest) (*listResponse, error) {
	rel, err := cleanRelativePath(req.path)
	if err != nil {
		return nil, goerr.Wrap(err, "invalid path")
	}

	queryPrefix := s.prefix
	if rel != "" {
		queryPrefix += rel + "/"
	}

	query := &storage.Query{
		Prefix:    queryPrefix,
		Delimiter: "/",
	}

	bkt := s.client.Bucket(s.bucket)
	it := bkt.Objects(ctx, query)

	type collectedEntry struct {
		name      string
		kind      entryKind
		size      int64
		updatedAt time.Time
	}
	var collected []collectedEntry

	for {
		attr, err := it.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return nil, goerr.Wrap(err, "failed to list objects",
				goerr.Value("bucket", s.bucket),
				goerr.Value("prefix", queryPrefix),
			)
		}

		// When Delimiter is set, the iterator yields either an object (attr.Name)
		// or a synthetic directory marker (attr.Prefix).
		if attr.Prefix != "" {
			dirName := strings.TrimSuffix(strings.TrimPrefix(attr.Prefix, queryPrefix), "/")
			if dirName == "" {
				continue
			}
			collected = append(collected, collectedEntry{
				name: dirName,
				kind: entryKindDir,
			})
			continue
		}

		if !strings.HasSuffix(attr.Name, ".json") {
			continue
		}
		// The object's name relative to queryPrefix; should be a single path segment.
		rel := strings.TrimPrefix(attr.Name, queryPrefix)
		if rel == "" || strings.Contains(rel, "/") {
			continue
		}
		traceID := strings.TrimSuffix(rel, ".json")
		collected = append(collected, collectedEntry{
			name:      traceID,
			kind:      entryKindFile,
			size:      attr.Size,
			updatedAt: attr.Updated,
		})
	}

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
	for _, e := range collected[startIdx:endIdx] {
		entry := entrySummary{
			Name: e.name,
			Kind: e.kind,
		}
		if e.kind == entryKindFile {
			entry.Size = e.size
			entry.UpdatedAt = e.updatedAt
		}
		resp.entries = append(resp.entries, entry)
	}

	if endIdx < len(collected) {
		last := collected[endIdx-1]
		resp.nextPageToken = encodePageToken(entrySortKey(last.kind, last.name))
	}

	return resp, nil
}

func (s *csSource) Get(ctx context.Context, path string) (*trace.Trace, error) {
	rel, err := cleanRelativePath(path)
	if err != nil {
		return nil, goerr.Wrap(err, "invalid path")
	}
	if rel == "" {
		return nil, goerr.New("trace path is required")
	}

	objectName := s.prefix + rel + ".json"
	reader, err := s.client.Bucket(s.bucket).Object(objectName).NewReader(ctx)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to read trace object",
			goerr.Value("bucket", s.bucket),
			goerr.Value("object", objectName),
		)
	}
	defer func() { _ = reader.Close() }()

	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, goerr.Wrap(err, "failed to read trace data",
			goerr.Value("bucket", s.bucket),
			goerr.Value("object", objectName),
		)
	}

	var t trace.Trace
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, goerr.Wrap(err, "failed to parse trace data",
			goerr.Value("bucket", s.bucket),
			goerr.Value("object", objectName),
		)
	}

	return &t, nil
}
