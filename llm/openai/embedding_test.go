package openai_test

import (
	"os"
	"testing"

	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/m-mizutani/gt"
)

func TestGenerateEmbedding(t *testing.T) {
	apiKey, ok := os.LookupEnv("TEST_OPENAI_API_KEY")
	if !ok {
		t.Skip("TEST_OPENAI_API_KEY is not set")
	}

	ctx := t.Context()
	client, err := openai.New(ctx, apiKey)
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
