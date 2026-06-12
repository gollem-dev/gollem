package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/openai"
)

// This example demonstrates how SubAgent middleware can access session history
// for post-execution processing like memory extraction and metrics collection.

func main() {
	// Create LLM client
	ctx := context.Background()
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY environment variable is required")
	}

	client, err := openai.New(ctx, apiKey)
	if err != nil {
		log.Fatalf("Failed to create OpenAI client: %v", err)
	}

	// Create a SubAgent with middleware that:
	// 1. Pre-execution: Injects context (timestamp, user info)
	// 2. Post-execution: Analyzes session history and extracts metrics
	subagent := gollem.NewSubAgent(
		"research_assistant",
		"A research assistant that provides detailed answers to queries",
		func() (*gollem.Agent, error) {
			return gollem.New(client), nil
		},
		gollem.WithSubAgentMiddleware(createMiddleware()),
	)

	// Create parent agent that uses the SubAgent
	parentAgent := gollem.New(
		client,
		gollem.WithSubAgents(subagent),
		gollem.WithLoopLimit(5),
	)

	// Execute
	result, err2 := parentAgent.Execute(ctx, gollem.Text("Use the research_assistant to explain what quantum computing is"))
	if err2 != nil {
		log.Fatalf("Execution failed: %v", err2)
	}

	fmt.Println("\n=== Final Result ===")
	fmt.Println(result.String())
}

// createMiddleware creates a middleware that demonstrates both pre and post execution processing
func createMiddleware() func(gollem.SubAgentHandler) gollem.SubAgentHandler {
	return func(next gollem.SubAgentHandler) gollem.SubAgentHandler {
		return func(ctx context.Context, args map[string]any) (gollem.SubAgentResult, error) {
			// === Pre-execution: Inject context ===
			startTime := time.Now()
			args["_execution_time"] = startTime.Format(time.RFC3339)
			args["_user_context"] = "Example user from middleware"

			fmt.Println("\n=== Middleware: Pre-execution ===")
			fmt.Printf("Injected execution time: %s\n", args["_execution_time"])
			fmt.Printf("Injected user context: %s\n", args["_user_context"])

			// Execute the SubAgent
			result, err := next(ctx, args)
			if err != nil {
				return gollem.SubAgentResult{}, err
			}

			// === Post-execution: Analyze session history ===
			fmt.Println("\n=== Middleware: Post-execution ===")

			// Access session history
			history, historyErr := result.Session.History()
			if historyErr != nil {
				fmt.Printf("Warning: Could not access history: %v\n", historyErr)
			} else if history != nil {
				// Extract metrics from history
				messageCount := len(history.Messages)
				executionDuration := time.Since(startTime)

				fmt.Printf("Session history contains %d messages\n", messageCount)
				fmt.Printf("Execution took %v\n", executionDuration)

				// Add metrics to result
				result.Data["metrics"] = map[string]any{
					"message_count":      messageCount,
					"execution_duration": executionDuration.String(),
					"execution_time":     args["_execution_time"],
				}

				// Simulate memory extraction (in real use, this would save to a database)
				fmt.Println("\nExtracted insights for memory:")
				for i, msg := range history.Messages {
					if len(msg.Contents) > 0 {
						// For simplicity, just show the role and number of content blocks
						fmt.Printf("  Message %d (%s): %d content blocks\n", i+1, msg.Role, len(msg.Contents))
					}
				}
			}

			// Cleanup temporary context fields
			delete(args, "_execution_time")
			delete(args, "_user_context")

			return result, nil
		}
	}
}
