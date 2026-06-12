package main

import (
	"bufio"
	"context"
	"fmt"
	"os"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/gemini"
)

// WeatherTool is a simple tool that returns weather information
type WeatherTool struct{}

func (t *WeatherTool) Spec() gollem.ToolSpec {
	return gollem.ToolSpec{
		Name:        "weather",
		Description: "Returns weather information for a city",
		Parameters: map[string]*gollem.Parameter{
			"city": {
				Type:        gollem.TypeString,
				Description: "City name",
				Required:    true,
			},
		},
	}
}

func (t *WeatherTool) Run(ctx context.Context, args map[string]any) (map[string]any, error) {
	city, ok := args["city"].(string)
	if !ok {
		return nil, fmt.Errorf("city is required")
	}

	// Simulate weather data
	weather := map[string]string{
		"tokyo":    "sunny, 25°C",
		"london":   "cloudy, 18°C",
		"new york": "rainy, 22°C",
		"paris":    "partly cloudy, 20°C",
		"sydney":   "sunny, 28°C",
	}

	if w, exists := weather[city]; exists {
		return map[string]any{
			"city":    city,
			"weather": w,
			"message": fmt.Sprintf("The weather in %s is %s.", city, w),
		}, nil
	}

	return map[string]any{
		"city":    city,
		"weather": "sunny, 23°C",
		"message": fmt.Sprintf("The weather in %s is sunny, 23°C (default).", city),
	}, nil
}

func main() {
	ctx := context.Background()

	// Initialize Gemini client
	client, err := gemini.New(ctx, os.Getenv("GEMINI_PROJECT_ID"), os.Getenv("GEMINI_LOCATION"))
	if err != nil {
		panic(err)
	}

	// Create agent with streaming response and tools
	agent := gollem.New(client,
		gollem.WithResponseMode(gollem.ResponseModeStreaming),
		gollem.WithTools(&WeatherTool{}),
		gollem.WithSystemPrompt("You are a helpful weather assistant. Use the weather tool to provide accurate weather information."),
	)

	fmt.Println("🌤️  Weather Chat Assistant")
	fmt.Println("💡 Ask me about the weather in any city!")
	fmt.Println("🔄 Conversation history is automatically managed")
	fmt.Println("📝 Type 'quit' to exit")
	fmt.Println("")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		input := scanner.Text()
		if input == "quit" || input == "exit" {
			fmt.Print("\n👋 Goodbye!")
			break
		}

		if input == "" {
			continue
		}

		fmt.Printf("🤖 ")

		// Execute with automatic session management
		// No need to manually handle history - it's managed automatically!
		result, err := agent.Execute(ctx, gollem.Text(input))
		if err != nil {
			fmt.Printf("\n❌ Error: %v\n", err)
			continue
		}

		// Display conclusion if available
		if result != nil && !result.IsEmpty() {
			fmt.Printf("\n💭 %s", result.String())
		}

		// Optional: Show conversation statistics
		if history, err := agent.Session().History(); err == nil && history != nil {
			fmt.Printf("\n📊 (Total messages: %d)\n", history.ToCount())
		}
		fmt.Println()
	}
}
