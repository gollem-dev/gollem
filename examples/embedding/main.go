//go:build examples

package main

import (
	"context"
	"fmt"
	"os"

	"github.com/gollem-dev/gollem/llm/openai"
)

func main() {
	ctx := context.Background()

	// Create OpenAI client
	client, err := openai.New(ctx, os.Getenv("OPENAI_API_KEY"))
	if err != nil {
		panic(err)
	}

	embedding, err := client.GenerateEmbedding(ctx, 100, []string{"Hello, world!", "This is a test"})
	if err != nil {
		panic(err)
	}
	fmt.Println("embedding:", embedding)
}
