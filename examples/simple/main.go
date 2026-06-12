//go:build examples

package main

import (
	"bufio"
	"context"
	"fmt"
	"os"

	"github.com/gollem-dev/gollem"
	"github.com/gollem-dev/gollem/llm/openai"
	"github.com/gollem-dev/gollem/mcp"
)

func main() {
	ctx := context.Background()

	// Create OpenAI client
	client, err := openai.New(ctx, os.Getenv("OPENAI_API_KEY"))
	if err != nil {
		panic(err)
	}

	// Create MCP client with local server (with custom client info)
	mcpLocal, err := mcp.NewStdio(ctx, "./mcp-server", []string{"arg1", "arg2"},
		mcp.WithStdioClientInfo("gollem-simple-example", "1.0.0"))
	if err != nil {
		panic(err)
	}
	defer mcpLocal.Close()

	// Create gollem agent with MCP tools
	agent := gollem.New(client,
		gollem.WithToolSets(mcpLocal),
		gollem.WithSystemPrompt("You are a helpful assistant with access to MCP tools."),
		gollem.WithContentBlockMiddleware(func(next gollem.ContentBlockHandler) gollem.ContentBlockHandler {
			return func(ctx context.Context, req *gollem.ContentRequest) (*gollem.ContentResponse, error) {
				resp, err := next(ctx, req)
				if err == nil && len(resp.Texts) > 0 {
					for _, text := range resp.Texts {
						fmt.Printf("🤖 %s\n", text)
					}
				}
				return resp, err
			}
		}),
	)

	fmt.Println("🚀 Simple Gollem Agent with MCP Tools")
	fmt.Println("💡 Enter your message below:")
	fmt.Print("> ")

	scanner := bufio.NewScanner(os.Stdin)
	if scanner.Scan() {
		input := scanner.Text()

		// Execute with automatic session management
		if err := agent.Execute(ctx, input); err != nil {
			fmt.Printf("❌ Error: %v\n", err)
		} else {
			fmt.Println("✅ Task completed!")
		}
	}
}
