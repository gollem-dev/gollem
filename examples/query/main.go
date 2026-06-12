//go:build examples

package main

import (
	"context"
	"fmt"
	"os"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/claude"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/gollem-dev/gollem/llm/openai"
)

// AnalysisResult is the structured response from the LLM.
type AnalysisResult struct {
	Summary  string   `json:"summary" description:"brief summary of the analysis"`
	Keywords []string `json:"keywords" description:"key terms extracted from the input"`
	Score    int      `json:"score" description:"relevance score from 1 to 10" min:"1" max:"10"`
}

func main() {
	ctx := context.Background()

	if len(os.Args) != 4 {
		fmt.Println("Usage: go run main.go <gemini|claude|openai> <model_name> <prompt>")
		os.Exit(1)
	}

	llmProvider := os.Args[1]
	model := os.Args[2]
	prompt := os.Args[3]

	var client gollem.LLMClient
	var err error

	switch llmProvider {
	case "gemini":
		client, err = gemini.New(ctx, os.Getenv("GEMINI_PROJECT_ID"), os.Getenv("GEMINI_LOCATION"), gemini.WithModel(model))
	case "claude":
		client, err = claude.New(ctx, os.Getenv("ANTHROPIC_API_KEY"), claude.WithModel(model))
	case "openai":
		client, err = openai.New(ctx, os.Getenv("OPENAI_API_KEY"), openai.WithModel(model))
	}

	if err != nil {
		panic(err)
	}

	resp, err := gollem.Query[AnalysisResult](ctx, client, prompt,
		gollem.WithQuerySystemPrompt("You are an expert analyst. Analyze the given text and return structured results."),
	)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Summary:  %s\n", resp.Data.Summary)
	fmt.Printf("Keywords: %v\n", resp.Data.Keywords)
	fmt.Printf("Score:    %d\n", resp.Data.Score)
	fmt.Printf("Tokens:   %d input, %d output\n", resp.InputToken, resp.OutputToken)
}
