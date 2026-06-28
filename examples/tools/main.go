//go:build examples

package main

import (
	"context"
	"log"
	"os"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/gemini"
)

func main() {
	ctx := context.Background()

	// Initialize Gemini client
	client, err := gemini.New(ctx, os.Getenv("GEMINI_PROJECT_ID"), os.Getenv("GEMINI_LOCATION"))
	if err != nil {
		log.Fatal(err)
	}

	// Register tools defined with typed handlers. The schema is inferred from the
	// input struct and the handler receives a decoded value, so there is no manual
	// type assertion on a map[string]any.
	tools := []gollem.Tool{
		gollem.MustNewTool("add", "Adds two numbers together", add),
		gollem.MustNewTool("multiply", "Multiplies two numbers together", multiply),
	}

	// Create agent with tools
	agent := gollem.New(client,
		gollem.WithTools(tools...),
		gollem.WithSystemPrompt("You are a helpful calculator assistant. Use the available tools to perform mathematical operations."),
		gollem.WithContentBlockMiddleware(func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
			return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
				resp, err := next(ctx, req)
				if err == nil && len(resp.Texts) > 0 {
					for _, text := range resp.Texts {
						log.Printf("🤖 %s", text)
					}
				}
				return resp, err
			}
		}),
		gollem.WithToolMiddleware(func(next gollem.ToolHandler) gollem.ToolHandler {
			return func(ctx context.Context, req *gollem.ToolExecRequest) (*gollem.ToolExecResponse, error) {
				log.Printf("⚡ Using tool: %s", req.Tool.Name)
				return next(ctx, req)
			}
		}),
	)

	query := "Add 5 and 3, then multiply the result by 2"
	log.Printf("📝 Query: %s", query)

	// Execute with automatic session management
	if _, err := agent.Execute(ctx, gollem.Text(query)); err != nil {
		log.Fatal(err)
	}

	log.Printf("✅ Calculation completed!")
}

// operands is the typed input shared by the calculator tools.
type operands struct {
	A float64 `json:"a" description:"First number" required:"true"`
	B float64 `json:"b" description:"Second number" required:"true"`
}

// calcResult is the typed output of the calculator tools.
type calcResult struct {
	Result float64 `json:"result" description:"Calculation result"`
}

func add(_ context.Context, in operands) (calcResult, error) {
	result := in.A + in.B
	log.Printf("🔢 Add: %.2f + %.2f = %.2f", in.A, in.B, result)
	return calcResult{Result: result}, nil
}

func multiply(_ context.Context, in operands) (calcResult, error) {
	result := in.A * in.B
	log.Printf("🔢 Multiply: %.2f × %.2f = %.2f", in.A, in.B, result)
	return calcResult{Result: result}, nil
}
