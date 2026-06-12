package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/gemini"
	"github.com/gollem-dev/gollem/strategy/reflexion"
)

func main() {
	ctx := context.Background()

	// Initialize Gemini client
	projectID := os.Getenv("GEMINI_PROJECT_ID")
	location := os.Getenv("GEMINI_LOCATION")
	if projectID == "" || location == "" {
		log.Fatal("GEMINI_PROJECT_ID and GEMINI_LOCATION environment variables are required")
	}

	client, err := gemini.New(ctx, projectID, location)
	if err != nil {
		log.Fatalf("Failed to create Gemini client: %v", err)
	}

	// Create a custom evaluator that checks if the answer contains specific keywords
	evaluator := reflexion.Evaluator(func(ctx context.Context, trajectory *reflexion.Trajectory) (*reflexion.EvaluationResult, error) {
		// Simple evaluation: check if response contains "correct" keyword
		// In real scenarios, you would implement more sophisticated evaluation logic
		response := ""
		if len(trajectory.FinalResponse) > 0 {
			response = trajectory.FinalResponse[0]
		}

		// For demo purposes, we'll use a simple heuristic
		// In production, you might use another LLM call, rule-based checks, or external validation
		success := len(response) > 100 // Simple check: response should be reasonably detailed
		score := 0.0
		if success {
			score = 1.0
		}

		feedback := ""
		if !success {
			feedback = "Response is too brief. Please provide more detailed explanation."
		}

		return &reflexion.EvaluationResult{
			Success:  success,
			Score:    score,
			Feedback: feedback,
		}, nil
	})

	// Create custom hooks to observe the reflection process
	hooks := &reflectionHooks{}

	// Create Reflexion strategy
	strategy := reflexion.New(
		client,
		reflexion.WithMaxTrials(3),
		reflexion.WithMemorySize(2),
		reflexion.WithEvaluator(evaluator),
		reflexion.WithHooks(hooks),
	)

	// Create agent with Reflexion strategy
	agent := gollem.New(client, gollem.WithStrategy(strategy))

	// Execute a task that might require multiple attempts
	question := gollem.Text("Explain the concept of recursion in programming with examples.")
	response, err := agent.Execute(ctx, question)
	if err != nil {
		log.Fatalf("Execution failed: %v", err)
	}

	fmt.Println("\n=== Final Response ===")
	for _, text := range response.Texts {
		fmt.Println(text)
	}
}

// reflectionHooks implements reflexion.Hooks to observe the reflection process
type reflectionHooks struct{}

func (h *reflectionHooks) OnTrialStart(ctx context.Context, trialNum int) error {
	fmt.Printf("\n--- Trial %d started ---\n", trialNum)
	return nil
}

func (h *reflectionHooks) OnTrialEnd(ctx context.Context, trialNum int, evaluation *reflexion.EvaluationResult) error {
	fmt.Printf("Trial %d ended - Success: %v, Score: %.2f\n", trialNum, evaluation.Success, evaluation.Score)
	if evaluation.Feedback != "" {
		fmt.Printf("Feedback: %s\n", evaluation.Feedback)
	}
	return nil
}

func (h *reflectionHooks) OnReflectionGenerated(ctx context.Context, trialNum int, reflection string) error {
	fmt.Printf("\nReflection for trial %d:\n%s\n", trialNum, reflection)
	return nil
}
