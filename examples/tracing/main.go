package main

import (
	"context"
	"fmt"
	"os"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/gollem-dev/gollem/trace"
)

// WeatherTool is a simple tool for demonstration.
type WeatherTool struct{}

func (t *WeatherTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "get_weather",
		Description: "Get the current weather for a location",
		Parameters: map[string]*gollem.Parameter{
			"location": {
				Type:        gollem.TypeString,
				Description: "City name",
				Required:    true,
			},
		},
	}
}

func (t *WeatherTool) Run(_ context.Context, args map[string]any) (map[string]any, error) {
	location, ok := args["location"].(string)
	if !ok {
		return nil, fmt.Errorf("location is required")
	}
	return map[string]any{
		"location":    location,
		"temperature": "22C",
		"condition":   "sunny",
	}, nil
}

func main() {
	ctx := context.Background()

	// Create OpenAI client
	client, err := openai.New(ctx, os.Getenv("OPENAI_API_KEY"))
	if err != nil {
		panic(err)
	}

	// Create a trace recorder with file repository
	rec := trace.New(
		trace.WithStackTrace(),
		trace.WithRepository(trace.NewFileRepository("./traces")),
		trace.WithMetadata(trace.TraceMetadata{
			Labels: map[string]string{"env": "example"},
		}),
	)

	// Create agent with trace handler
	agent := gollem.New(client,
		gollem.WithTools(&WeatherTool{}),
		gollem.WithSystemPrompt("You are a helpful assistant."),
		gollem.WithTrace(rec),
	)

	// Execute - trace is automatically recorded and saved
	result, err := agent.Execute(ctx, gollem.Text("What is the weather in Tokyo?"))
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	if result != nil {
		fmt.Printf("Result: %s\n", result.String())
	}

	fmt.Println("Trace saved to ./traces/ directory")
}
