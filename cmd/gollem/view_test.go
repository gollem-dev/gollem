package main_test

import (
	"testing"

	main "github.com/gollem-dev/gollem/cmd/gollem"
	"github.com/m-mizutani/gt"
)

func TestParseGSURI(t *testing.T) {
	t.Run("bucket only", func(t *testing.T) {
		bucket, prefix, err := main.ParseGSURI("gs://my-bucket")
		gt.NoError(t, err)
		gt.Equal(t, "my-bucket", bucket)
		gt.Equal(t, "", prefix)
	})

	t.Run("bucket with trailing slash", func(t *testing.T) {
		bucket, prefix, err := main.ParseGSURI("gs://my-bucket/")
		gt.NoError(t, err)
		gt.Equal(t, "my-bucket", bucket)
		gt.Equal(t, "", prefix)
	})

	t.Run("bucket and prefix", func(t *testing.T) {
		bucket, prefix, err := main.ParseGSURI("gs://my-bucket/path/to/traces/")
		gt.NoError(t, err)
		gt.Equal(t, "my-bucket", bucket)
		gt.Equal(t, "path/to/traces/", prefix)
	})

	t.Run("bucket and prefix without trailing slash", func(t *testing.T) {
		bucket, prefix, err := main.ParseGSURI("gs://my-bucket/traces")
		gt.NoError(t, err)
		gt.Equal(t, "my-bucket", bucket)
		gt.Equal(t, "traces/", prefix)
	})

	t.Run("missing gs:// prefix", func(t *testing.T) {
		_, _, err := main.ParseGSURI("s3://my-bucket")
		gt.Error(t, err)
	})

	t.Run("empty URI", func(t *testing.T) {
		_, _, err := main.ParseGSURI("gs://")
		gt.Error(t, err)
	})
}
