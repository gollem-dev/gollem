package gemini_test

import (
	"os"
	"testing"

	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/m-mizutani/gt"
)

func TestGenerateEmbedding(t *testing.T) {
	projectID, ok := os.LookupEnv("TEST_GCP_PROJECT_ID")
	if !ok {
		t.Skip("TEST_GCP_PROJECT_ID is not set")
	}

	location, ok := os.LookupEnv("TEST_GCP_LOCATION")
	if !ok {
		t.Skip("TEST_GCP_LOCATION is not set")
	}

	ctx := t.Context()
	client, err := gemini.New(ctx, projectID, location)
	if err != nil {
		t.Fatalf("failed to create client: %v", err)
	}

	embeddings, err := client.GenerateEmbedding(ctx, 256, []string{"not, SANE", "Five timeless words"})
	if err != nil {
		t.Fatalf("failed to generate embedding: %v", err)
	}

	gt.A(t, embeddings).Length(2).
		At(0, func(t testing.TB, v []float64) {
			gt.A(t, v).Longer(0)
		}).
		At(1, func(t testing.TB, v []float64) {
			gt.A(t, v).Longer(0)
		})
}
